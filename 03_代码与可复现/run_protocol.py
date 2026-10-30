"""成员 A 的协议运行包装器：把一次 TimesNet / ETTm2 运行做成成员 B 要求的可审计记录。

它做四件事
----------
1. 按 `run_plan.csv` 里的 run_id 取出该次运行的条件（L / H / 缩放 / 种子），
   用**官方 ETTm2 脚本的设置**拼出 `run.py` 的命令行；
2. 用成员 B 自己的 `08_reproduction_comparison.py new-run` 建运行目录模板，
   保证目录结构、`run_metadata.json` 字段、起报索引与他完全一致；
3. 训练前先把 `command.txt` / `hyperparameters.json` / `environment.txt` /
   `patches.diff` 落盘（这样即使训练中途崩了，也留有真实记录），
   训练日志直接 tee 成 `train.log`；
4. 训练完成后把 checkpoint 拷进运行目录、补齐 `run_metadata.json` 的
   代码版本字段，并导出验证集预测张量。

为什么必须这么做
----------------
`exp.vali()` 只算损失、**不保存任何数组**，而且 `data_provider` 里验证集默认
`shuffle=True`。所以验证预测必须用固定顺序单独导出，否则起报行号会与
`expected_origins_validation_H*.npy` 整体错位——而错位之后指标仍然"看起来正常"，
极难发现。

协议顺序
--------
    train   → 训练 + 验证 + 导出验证预测（不做测试推理，靠 P6 的 --skip_test）
    [B]     → score-validation、seal
    test    → 加载 checkpoint 做测试推理并导出 pred/true/origin_rows
    [B]     → evaluate

用法
----
    python tools/run_protocol.py train --run-id TimesNet_L96_H96_zscore_s2021
    python tools/run_protocol.py test  --run-id TimesNet_L96_H96_zscore_s2021
    python tools/run_protocol.py train --run-id ... --dry-run     # 只打印命令
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                            # memberA_timesnet/
WORK = ROOT / "work"                          # 成员 B 交接包的工作副本（project root）
REPO = ROOT.parent / "_tsl"                   # 打过补丁的官方仓库（目录名必须短，见 MAX_PATH）
# _tsl 是纯副本、没有 .git，commit / dirty 必须从带历史的原始 clone 取
GIT_REPO = ROOT / "code" / "Time-Series-Library"
COMMIT_FILE = HERE / "repo_commit.txt"
SCRIPTS_B = WORK / "scripts"
PRED_DIR = WORK / "experiments/predictions"
PATCH_DIFF = ROOT / "_smoke_test/patches.diff"

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
CONTINUOUS = WORK / "data/processed/v1/continuous"

# 官方 scripts/long_term_forecast/ETT_script/TimesNet_ETTm2.sh 的逐 H 设置。
# 注意其中两个"怪癖"：H=192 与 H=720 带 --train_epochs 1；H=720 用 d_model 16。
OFFICIAL = {
    96: dict(train_epochs=10, d_model=32),
    192: dict(train_epochs=1, d_model=32),
    336: dict(train_epochs=10, d_model=32),
    720: dict(train_epochs=1, d_model=16),
}

# 验证段在原始序列里的起始行（border1s[1] = 34560 - seq_len），
# 首样本最后一个已观测行 = 34560 - seq_len + seq_len - 1 = 34559
VAL_LAST_OBSERVED_BASE = 34559

# argparse 的 store_true 开关：只能出现、不能带值
FLAG_KEYS = {"skip_test"}


# ---------------------------------------------------------------- 计划与参数
def load_plan() -> pd.DataFrame:
    p = WORK / "artifacts/evaluation/v1/run_plan.csv"
    if not p.is_file():
        raise SystemExit(f"找不到运行计划：{p}")
    return pd.read_csv(p)


def plan_row(run_id: str, plan: pd.DataFrame):
    sel = plan.loc[plan.run_id.eq(run_id)]
    if len(sel) != 1:
        raise SystemExit(
            f"run-id 不在固定计划中，或重复出现：{run_id}\n"
            f"请查 {WORK / 'artifacts/evaluation/v1/run_plan.csv'}"
        )
    return sel.iloc[0]


def short_model_id(row) -> str:
    """checkpoint 目录名要短（Windows MAX_PATH=260），但必须唯一区分
    L / H / 缩放 / 种子，否则同 L/H/缩放 的不同种子会互相覆盖检查点。"""
    sc = "z" if row.input_scaling == "zscore" else "m"
    return f"L{int(row.input_length)}H{int(row.horizon)}{sc}_s{int(row.seed)}"


def build_params(row, num_workers: int, epochs: int | None) -> dict:
    h = int(row.horizon)
    official = OFFICIAL[h]
    scaling = str(row.input_scaling)
    # zscore：给原始连续数据，由加载器用训练段拟合的 StandardScaler 变换一次
    # minmax ：给已预缩放数据，并关闭加载器二次缩放（P7 的 --scale 0）
    if scaling == "zscore":
        data_path, scale = "ETTm2_raw.csv", 1
    elif scaling == "minmax":
        data_path, scale = "ETTm2_minmax.csv", 0
    else:
        raise SystemExit(f"未知的 input_scaling: {scaling}")

    return dict(
        task_name="long_term_forecast",
        is_training=1,
        root_path=str(CONTINUOUS).replace("\\", "/"),
        data_path=data_path,
        model_id=short_model_id(row),
        model="TimesNet",
        data="ETTm2",
        features="M",
        target="OT",
        freq="h",                       # 官方脚本没传，默认 h
        seq_len=int(row.input_length),
        label_len=48,
        pred_len=h,
        e_layers=2,
        d_layers=1,
        factor=3,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=official["d_model"],
        d_ff=32,
        top_k=5,
        num_kernels=6,
        embed="timeF",
        dropout=0.1,
        train_epochs=official["train_epochs"] if epochs is None else int(epochs),
        batch_size=32,
        num_workers=num_workers,
        learning_rate=0.0001,
        loss="MSE",
        lradj="type1",
        patience=3,
        itr=1,
        des="a",
        fix_seed=int(row.seed),
        scale=scale,
        skip_test=1,
    )


def params_to_argv(p: dict) -> list[str]:
    argv = ["run.py"]
    for k, v in p.items():
        # store_true 型开关不能带值，否则 argparse 会把值当成多余的位置参数
        if k in FLAG_KEYS:
            if v:
                argv.append(f"--{k}")
            continue
        argv += [f"--{k}", str(v)]
    return argv


def params_to_command_text(p: dict, indent: str = "  ") -> str:
    lines = ["python -u run.py"]
    for k, v in p.items():
        if k in FLAG_KEYS:
            if v:
                lines.append(f"{indent}--{k}")
            continue
        lines.append(f"{indent}--{k} {v}")
    return " \\\n".join(lines)


# ---------------------------------------------------------------- 运行目录
def ensure_run_dir(run_id: str) -> Path:
    base = PRED_DIR / run_id
    if base.is_dir():
        return base
    cmd = [
        sys.executable, str(SCRIPTS_B / "08_reproduction_comparison.py"), "new-run",
        "--run-id", run_id, "--project-root", str(WORK),
    ]
    print("[1/6] 建运行目录模板（调用 B 的 new-run）…")
    r = subprocess.run(cmd, cwd=str(WORK), capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"new-run 失败：\n{r.stdout}\n{r.stderr}")
    print("      " + r.stdout.strip().splitlines()[-2])
    return base


def repo_commit() -> tuple[str, bool]:
    """返回 (40 位 commit, 是否 dirty)。

    commit 来自带 git 历史的原始 clone（`_tsl` 是纯副本，没有 .git）。
    dirty 只要满足其一即为 True：clone 有未提交改动，或 `_tsl` 打过补丁
    （补丁本身就意味着交付的代码与原版不同，必须如实标记）。
    """
    def git(*a):
        return subprocess.run(
            ["git", "-C", str(GIT_REPO), *a], capture_output=True, text=True, encoding="utf-8"
        )

    head = ""
    clone_dirty = False
    try:
        head = git("rev-parse", "HEAD").stdout.strip()
        clone_dirty = bool(git("status", "--short").stdout.strip())
    except OSError:
        pass

    if len(head) != 40:
        # 兜底：读第一次成功取得时记下的 commit，避免因为环境问题让整次运行失败
        if COMMIT_FILE.is_file():
            head = COMMIT_FILE.read_text(encoding="utf-8").strip()
        else:
            raise SystemExit(
                f"取不到 git commit（{GIT_REPO} 不是 git 仓库？），也不要写 {COMMIT_FILE}"
            )
    elif not COMMIT_FILE.is_file():
        COMMIT_FILE.write_text(head + "\n", encoding="utf-8")

    if len(head) != 40:
        raise SystemExit(f"commit 不是 40 位：{head!r}")

    patched = PATCH_DIFF.is_file() and PATCH_DIFF.stat().st_size > 0
    return head, (clone_dirty or patched)


def write_environment(path: Path, extra: dict | None = None) -> None:
    import torch
    lines = [
        "=== 运行环境（成员 A / TimesNet / ETTm2）===",
        f"time_hong_kong : {pd.Timestamp.now(tz='Asia/Hong_Kong').isoformat(timespec='seconds')}",
        f"python         : {sys.version.split()[0]}  ({sys.executable})",
        f"platform       : {platform.platform()}",
        f"machine        : {platform.machine()}",
        f"torch          : {torch.__version__}",
        f"torch.cuda     : {torch.version.cuda}",
        f"cuda_available : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            lines.append(
                f"gpu[{i}]        : {pr.name}, {pr.total_memory / 1024**3:.2f} GiB, "
                f"cc {pr.major}.{pr.minor}"
            )
    lines.append("")
    lines.append("--- 关键依赖版本 ---")
    # import 名与发行名不同的几个要显式映射
    pkg = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
    }
    for modname, distname in pkg.items():
        try:
            mod = __import__(modname)
            lines.append(f"{distname:<14}: {getattr(mod, '__version__', 'unknown')}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"{distname:<14}: MISSING ({type(e).__name__})")
    if extra:
        lines.append("")
        lines.append("--- 运行参数补充 ---")
        for k, v in extra.items():
            lines.append(f"{k:<14}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fill_metadata(base: Path, row, params: dict, setting: str | None) -> dict:
    """补齐/纠正 run_metadata.json 里由我们负责的字段。"""
    p = base / "run_metadata.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    commit, dirty = repo_commit()
    scaling = str(row.input_scaling)

    meta.update(
        {
            "code_repository_url": "https://github.com/thuml/Time-Series-Library",
            "code_commit": commit,
            "code_dirty": dirty,
            "code_diff_file": "patches.diff",
            # 用真实文件名覆盖模板里不存在的 model_checkpoint.bin
            "model_artifact_file": "checkpoint.pth",
            "loader_source": "raw_continuous" if scaling == "zscore" else "pre_scaled_continuous",
            "loader_scale": scaling == "zscore",
            "prediction_space": scaling,
            "checkpoint_setting": setting,
            # 短码含种子，不能进 hyperparameters.json（会让同组种子哈希不一致），
            # 但它是定位 checkpoint 的钥匙，所以记在 metadata 里保留可追溯性
            "checkpoint_model_id": params.get("model_id"),
            "notes": (
                "TimesNet 内部归一化保留；zscore 路径用训练段 StandardScaler 恰好变换一次，"
                "minmax 路径用预缩放文件并以 --scale 0 关闭加载器二次缩放。"
                "训练与测试分离（--skip_test），测试推理在 seal 之后执行。"
            ),
        }
    )
    # B 的 load_meta 会做严格一致性检查（loader_scale 必须与 loader_source 逻辑相符）
    assert meta["loader_scale"] is (meta["loader_source"] == "raw_continuous")
    p.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return meta


def find_setting_dir(model_id: str) -> str | None:
    ck = REPO / "checkpoints"
    if not ck.is_dir():
        return None
    hits = [d.name for d in ck.iterdir() if d.is_dir() and f"_{model_id}_TimesNet_" in d.name]
    if not hits:
        return None
    return max(hits, key=lambda n: (ck / n).stat().st_mtime)


# ---------------------------------------------------------------- 训练进度显示
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RE_ITERS = re.compile(r"iters:\s*(\d+),\s*epoch:\s*(\d+)\s*\|\s*loss:\s*([-\d.eE+]+)")
_RE_SPEED = re.compile(r"speed:\s*([\d.]+)s/iter;\s*left time:\s*([\d.]+)s")
_RE_COST = re.compile(r"Epoch:\s*(\d+)\s+cost time:\s*([\d.]+)")
_RE_SUMMARY = re.compile(
    r"Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([-\d.eE+]+)\s*"
    r"Vali Loss:\s*([-\d.eE+]+)\s*Test Loss:\s*([-\d.eE+]+)"
)
_RE_BEST = re.compile(r"Validation loss decreased\s*\(([-\d.eE+]+)\s*-->\s*([-\d.eE+]+)\)")
_RE_PATIENCE = re.compile(r"EarlyStopping counter:\s*(\d+)\s+out of\s+(\d+)")
_RE_EPOCH_SUMMARY = re.compile(r"Epoch:\s*(\d+),\s*Steps:")


def _fmt_dur(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ProgressTail(threading.Thread):
    """实时读 train.log，在控制台打印可读的训练进度。

    为什么放在包装器里、而不是给官方代码加打印
    ------------------------------------------
    `run_metadata.json` 记录 `patches.diff` 的哈希（`code_diff_sha256`）。
    如果为了加打印再改官方仓库，已跑完的运行与后续运行代码哈希就不一致，
    成员 B 的 `aggregate_metrics` 会拒绝把同组种子聚合。
    本类只读日志文件，对训练进程零侵入，也不改变任何被记录的哈希。
    """

    # 官方每 100 步刷一次的噪音行，不再往控制台重复；其余原样透传（便于看报错）
    _NOISE = ("iters:", "speed:", "cost time:")
    _BEAT_SECONDS = 45

    def __init__(self, log_path: Path, total_epochs: int):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.total_epochs = total_epochs
        self._halt = threading.Event()
        self.epochs: list[dict] = []
        self.cur_epoch = 0
        self.cur_iter = 0
        self.cur_loss = float("nan")
        self.left_time: float | None = None
        self.epoch_time: float | None = None
        self.best = (float("inf"), 0)
        self.patience = (0, 3)
        self.early_stopped = False
        self._in_args = False
        self._last_beat = time.time()

    # -------------------------------------------------- 线程主体
    def run(self) -> None:
        fh = None
        deadline = time.time() + 90
        while not self._halt.is_set() and time.time() < deadline:
            if self.log_path.is_file():
                fh = self.log_path.open("r", encoding="utf-8", errors="replace")
                break
            self._halt.wait(0.5)
        if fh is None:
            return

        buffer = ""
        with fh:
            while not self._halt.is_set():
                chunk = fh.read()
                if chunk:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        self._handle(line.rstrip("\r"))
                else:
                    self._heartbeat()
                    self._halt.wait(1.0)
            # 停止信号到达后把剩余内容读完，保证最后一轮不丢
            try:
                rest = fh.read()
            except Exception:  # noqa: BLE001
                rest = ""
            if rest:
                buffer += rest
            for line in buffer.split("\n"):
                if line.strip():
                    self._handle(line.rstrip("\r"))

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=10)

    # -------------------------------------------------- 解析
    def _handle(self, raw: str) -> None:
        line = _ANSI.sub("", raw)

        # 官方开头那张 50 行的参数表只打印一次，控制台里折叠掉（完整内容仍在 train.log）
        if "Args in experiment:" in line:
            self._in_args = True
            print("  │ （官方参数表已折叠，完整内容见 train.log）")
            return
        if self._in_args:
            if line.startswith("Use GPU:") and "cuda" in line:
                self._in_args = False
            return

        m = _RE_SUMMARY.search(line)
        if m:
            self.cur_epoch = int(m.group(1))
            self._record_epoch(int(m.group(2)), float(m.group(3)),
                               float(m.group(4)), float(m.group(5)))
            return

        m = _RE_ITERS.search(line)
        if m:
            self.cur_iter = int(m.group(1))
            self.cur_epoch = int(m.group(2))
            self.cur_loss = float(m.group(3))
            self._heartbeat()
            return

        m = _RE_SPEED.search(line)
        if m:
            self.left_time = float(m.group(2))
            return

        m = _RE_COST.search(line)
        if m:
            self.epoch_time = float(m.group(2))
            return

        m = _RE_BEST.search(line)
        if m:
            self.best = (float(m.group(2)), self.cur_epoch)
            return

        m = _RE_PATIENCE.search(line)
        if m:
            self.patience = (int(m.group(1)), int(m.group(2)))
            return

        if "Early stopping" in line and len(line.strip()) < 40:
            self.early_stopped = True
            print("  ⏹  触发 EarlyStopping，训练提前结束")
            return

        if any(k in line for k in self._NOISE):
            return
        if line.strip():
            print("  │ " + line)

    def _record_epoch(self, steps: int, train: float, valid: float, test: float) -> None:
        self.epochs.append({
            "epoch": self.cur_epoch, "steps": steps,
            "train": train, "vali": valid, "test": test,
            "cost": self.epoch_time,
        })
        # 注意：官方的 "Validation loss decreased" 与 "EarlyStopping counter" 两种提示
        # 都打印在本行**之后**，所以不能用回调值，直接由已收集的轮次自己算，
        # 否则显示的永远是上一轮的状态。
        best = min(self.epochs, key=lambda x: x["vali"])
        since_best = self.cur_epoch - best["epoch"]          # 等价于 patience 计数器
        costs = [e["cost"] for e in self.epochs if e.get("cost")]
        mean_cost = sum(costs) / len(costs) if costs else None
        remaining = self.total_epochs - self.cur_epoch
        eta = (f"   预计再 {_fmt_dur(mean_cost * remaining)}"
               if remaining > 0 and mean_cost else "")
        cost_txt = f"{self.epoch_time:.1f}s" if self.epoch_time else "—"
        print()
        print(f"  ┌─ 第 {self.cur_epoch}/{self.total_epochs} 轮 " + "─" * 44)
        print(f"  │ 本轮用时 {cost_txt}   步数 {steps}")
        print(f"  │ Train {train:.6f}    Vali {valid:.6f}    Test {test:.6f}")
        print(f"  │ 历史最佳 Vali {best['vali']:.6f}（第 {best['epoch']} 轮）"
              f"   patience {since_best}/{self.patience[1]}{eta}")
        print("  └" + "─" * 56)
        self._last_beat = time.time()

    def _heartbeat(self) -> None:
        if time.time() - self._last_beat < self._BEAT_SECONDS:
            return
        if not self.cur_epoch:
            return
        left = _fmt_dur(self.left_time) if self.left_time else "--:--"
        print(f"  ·  第 {self.cur_epoch}/{self.total_epochs} 轮  "
              f"迭代 {self.cur_iter}  loss {self.cur_loss:.6f}  本轮剩余 {left}")
        self._last_beat = time.time()

    # -------------------------------------------------- 收尾
    def summary(self) -> str:
        if not self.epochs:
            return ""
        out = ["", "  ══ 训练汇总 " + "═" * 48,
               f"  {'轮':>3}  {'用时':>8}  {'Train':>10}  {'Vali':>10}  {'Test':>10}",
               "  " + "-" * 48]
        for e in self.epochs:
            cost = f"{e['cost']:.1f}s" if e.get("cost") else "—"
            out.append(f"  {e['epoch']:>3}  {cost:>8}  {e['train']:>10.6f}  "
                       f"{e['vali']:>10.6f}  {e['test']:>10.6f}")
        best = min(self.epochs, key=lambda x: x["vali"])
        total = sum(e["cost"] for e in self.epochs if e.get("cost"))
        out.append("  " + "-" * 48)
        out.append(f"  最佳 Vali {best['vali']:.6f}（第 {best['epoch']} 轮）"
                   f"   共 {len(self.epochs)} 轮 / {_fmt_dur(total)}"
                   + ("   EarlyStopping 提前结束" if self.early_stopped else ""))
        out.append("  " + "═" * 56)
        return "\n".join(out)


# ---------------------------------------------------------------- 训练
def hyperparameters_payload(params: dict) -> dict:
    """生成 `hyperparameters.json` 的内容。

    **必须与随机种子无关。** 成员 B 的 `summarize` 会在同一个
    (model, input_length, horizon, input_scaling) 分组内比对
    `hyperparameters_sha256`，只要混进任何随种子变化的字段，三个种子就会被
    判为"配置不一致"并抛出
    `同一条件的随机种子运行混入不同配置／代码 ...；hyperparameters_sha256`。

    所以这里剔除三样东西：
      - `fix_seed`  —— 种子本身
      - `model_id`  —— checkpoint 短码里编码了种子（如 L96H96z_s2021）
      - `run_id`    —— 同样含种子
    种子仍然完整记录在 `run_metadata.json`（由 B 的 new-run 按运行计划写入）。
    """
    hp = dict(params)
    for k in ("root_path", "fix_seed", "model_id", "run_id"):
        hp.pop(k, None)
    hp.update(
        {
            "model": "TimesNet",
            "feature_order": FEATURES,
            "official_setting_source": "scripts/long_term_forecast/ETT_script/TimesNet_ETTm2.sh",
            "used_official_dmodel16_only_for_H720": params["pred_len"] == 720,
            "used_official_train_epochs_1_only_for_H192_H720": params["pred_len"] in (192, 720),
            "note_freq": "--freq 官方脚本未传，默认 'h'，即 15 分钟数据使用小时级时间特征（保持官方原样）",
            "note_seed": "随机种子按 B 的协议只写在 run_metadata.json，不写入本文件，"
                         "以免同组种子的超参数哈希不一致而无法聚合",
        }
    )
    return hp


def training_complete(log_path: Path, train_epochs: int) -> bool:
    """判断 train.log 是否显示训练**正常跑完**。

    只看「有没有日志文件」是不够的：进程被中途杀掉时日志也在，但 checkpoint
    只是某一轮的最好结果，复用它会错误地声称"按官方设置跑完了"。
    因此要求出现 `Early stopping`，或者已跑满名义轮数。
    """
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Early stopping" in text:
        return True
    done = len(_RE_EPOCH_SUMMARY.findall(text))
    return done >= train_epochs


def _training_already_done(base: Path, params: dict) -> bool:
    """该运行目录对应的训练是否已完成，可直接复用 _tsl 里的 checkpoint。"""
    setting = find_setting_dir(params["model_id"])
    if not setting:
        return False
    ckpt = REPO / "checkpoints" / setting / "checkpoint.pth"
    if not ckpt.is_file():
        return False
    return training_complete(base / "train.log", params["train_epochs"])


def cmd_train(args) -> int:
    plan = load_plan()
    row = plan_row(args.run_id, plan)
    params = build_params(row, args.num_workers, args.epochs)
    argv = params_to_argv(params)
    base = PRED_DIR / args.run_id

    print(f"run_id      : {args.run_id}")
    print(f"条件        : L={int(row.input_length)}  H={int(row.horizon)}  "
          f"缩放={row.input_scaling}  种子={int(row.seed)}")
    print(f"官方设置    : train_epochs={params['train_epochs']}  d_model={params['d_model']}  "
          f"batch={params['batch_size']}")
    print(f"数据        : {params['data_path']}  (--scale {params['scale']})")
    print(f"仓库        : {REPO}")
    print()

    if args.dry_run:
        print("=== 将要执行的命令（cwd = %s）===" % REPO)
        print(params_to_command_text(params))
        return 0

    base = ensure_run_dir(args.run_id)

    # 训练阶段已完整做完就不重跑（否则手动重复执行会白白再训 20+ 分钟）
    if (base / "validation_pred.npy").is_file() and (base / "checkpoint.pth").is_file():
        print(f"该运行的训练阶段已完成（validation_pred.npy 与 checkpoint.pth 都在），无需重跑。")
        print(f"下一步：score-validation → seal → test → evaluate，"
              f"可直接用 run_batch.py --limit 1。")
        return 0

    # ---- 先落盘记录，再训练（中途崩了也留有真实记录）----
    print("[2/6] 落盘 command.txt / hyperparameters.json / environment.txt / patches.diff …")
    (base / "command.txt").write_text(
        "=== 实际执行的 argv（无 shell，逐 token）===\n"
        + " ".join(argv)
        + "\n\n=== 等价多行形式 ===\n"
        + params_to_command_text(params)
        + f"\n\n# cwd = {REPO}\n",
        encoding="utf-8",
    )
    (base / "hyperparameters.json").write_text(
        json.dumps(hyperparameters_payload(params), ensure_ascii=False,
                   indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_environment(base / "environment.txt", {"root_path": str(CONTINUOUS), "repo": str(REPO)})
    if PATCH_DIFF.is_file():
        shutil.copy2(PATCH_DIFF, base / "patches.diff")
    fill_metadata(base, row, params, setting=None)

    # ---- 训练 ----
    if _training_already_done(base, params):
        print("[3/6] 检测到该运行的训练已**完整结束**（train.log 有 Early stopping 或已跑满轮数），")
        print("      复用 _tsl 里的 checkpoint，跳过训练；train.log 原样保留。")
    else:
        print(f"[3/6] 开始训练（官方 {params['train_epochs']} epochs）→ {base.name}/train.log")
        print("      下面实时显示每轮 loss 与用时；原始输出同时完整写入 train.log\n")
        log = (base / "train.log").open("w", encoding="utf-8")
        tail = ProgressTail(base / "train.log", params["train_epochs"])
        tail.start()
        try:
            r = subprocess.run(
                [sys.executable, "-u", *argv], cwd=str(REPO),
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        finally:
            log.close()
            tail.stop()
        summary = tail.summary()
        if summary:
            print(summary)
        if r.returncode != 0:
            print(f"FAIL: 训练退出码 {r.returncode}，见 {base / 'train.log'}")
            return r.returncode

    # ---- 定位并拷贝 checkpoint ----
    print("[4/6] 定位 checkpoint …")
    setting = find_setting_dir(params["model_id"])
    if not setting:
        print("FAIL: 找不到 checkpoint 目录")
        return 1
    ckpt = REPO / "checkpoints" / setting / "checkpoint.pth"
    if not ckpt.is_file():
        print(f"FAIL: 找不到 {ckpt}")
        return 1
    shutil.copy2(ckpt, base / "checkpoint.pth")
    print(f"      {setting} -> checkpoint.pth ({ckpt.stat().st_size / 1024**2:.1f} MB)")
    fill_metadata(base, row, params, setting=setting)

    # ---- 导出验证预测 ----
    print("[5/6] 导出验证集预测（固定顺序，关闭 shuffle）…")
    n = export_validation(base, row, params, setting, base / "checkpoint.pth")
    print(f"      验证样本数 = {n}")

    # ---- 自检 ----
    print("[6/6] 与 B 的起报索引逐行核对 …")
    if not verify_origins(base, "validation", int(row.horizon), VAL_LAST_OBSERVED_BASE):
        return 1

    print()
    print("=" * 68)
    print(f"训练阶段完成：{args.run_id}")
    print(f"运行目录：{base}")
    print("下一步：交给成员 B 执行 score-validation 与 seal；之后用 `test` 子命令做测试推理。")
    print("=" * 68)
    return 0


# ---------------------------------------------------------------- 验证集导出
def export_validation(base: Path, row, params: dict, setting: str, ckpt: Path) -> int:
    """用固定顺序前向一遍验证集，导出 [N,H,7] 的预测与真值。

    直接构造 Dataset_ETT_minute(flag='val') 并手写 DataLoader(shuffle=False,
    drop_last=False)，避免走 data_provider 里 val 默认 shuffle=True 的分支。
    """
    import torch
    sys.path.insert(0, str(REPO))
    from data_provider.data_loader import Dataset_ETT_minute  # noqa: E402
    from models import TimesNet  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    cfg = Namespace(
        task_name="long_term_forecast",
        seq_len=params["seq_len"], label_len=params["label_len"], pred_len=params["pred_len"],
        e_layers=params["e_layers"], enc_in=params["enc_in"], dec_in=params["dec_in"],
        c_out=params["c_out"], d_model=params["d_model"], d_ff=params["d_ff"],
        top_k=params["top_k"], num_kernels=params["num_kernels"], embed=params["embed"],
        freq=params["freq"], dropout=params["dropout"], augmentation_ratio=0,
    )

    ds = Dataset_ETT_minute(
        args=cfg, root_path=params["root_path"], flag="val",
        size=[cfg.seq_len, cfg.label_len, cfg.pred_len], features="M", target="OT",
        data_path=params["data_path"], scale=bool(params["scale"]),
        timeenc=1 if params["embed"] == "timeF" else 0, freq=params["freq"],
    )
    dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=False,
                    num_workers=0, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesNet.Model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds, trues = [], []
    H = cfg.pred_len
    with torch.no_grad():
        for x, y, xm, ym in dl:
            # 必须显式 .float()：Dataset 返回的是 numpy float64，直接 to(device)
            # 会得到 DoubleTensor，与模型参数（float32）类型不符而报错。
            # 官方 exp_long_term_forecasting 的训练循环里也有同样的 .float()，
            # 这里手写导出循环时必须自己补上。
            x, y, xm, ym = [t.float().to(device) for t in (x, y, xm, ym)]
            out = model(x, xm, y, ym)[:, -H:, :]
            preds.append(out.cpu().numpy())
            trues.append(y[:, -H:, :].cpu().numpy())

    pred = np.concatenate(preds, axis=0).astype(np.float32)
    true = np.concatenate(trues, axis=0).astype(np.float32)
    origin = (VAL_LAST_OBSERVED_BASE + np.arange(len(ds), dtype=np.int64))

    np.save(base / "validation_pred.npy", pred, allow_pickle=False)
    np.save(base / "validation_true.npy", true, allow_pickle=False)
    np.save(base / "validation_origin_rows.npy", origin, allow_pickle=False)

    # 权重文件里含绝对路径信息，交付前不必保留临时目录名
    print(f"      validation_pred {pred.shape} {pred.dtype}, "
          f"true {true.shape}, origin {origin.shape}")
    return len(ds)


def verify_origins(base: Path, split: str, horizon: int, base_origin: int) -> bool:
    name = "validation_origin_rows.npy" if split == "validation" else "origin_rows.npy"
    got = np.load(base / name)
    exp_path = WORK / f"artifacts/evaluation/v1/expected_origins_{split}_H{horizon}.npy"
    want = np.load(exp_path)
    if got.shape != want.shape:
        print(f"FAIL: {name} 形状 {got.shape} 与 B 的 {want.shape} 不一致")
        return False
    if not np.array_equal(got, want):
        bad = np.nonzero(got != want)[0][:5]
        print(f"FAIL: {name} 与 B 的索引不一致，前几处位置 {bad.tolist()}")
        return False
    print(f"      OK  {name} 与 B 的 expected_origins_{split}_H{horizon}.npy "
          f"完全一致（{len(want)} 行，首个 {int(want[0])}，末个 {int(want[-1])}）")
    return True


# ---------------------------------------------------------------- 测试阶段
def cmd_test(args) -> int:
    plan = load_plan()
    row = plan_row(args.run_id, plan)
    base = PRED_DIR / args.run_id
    if not (base / "locked_selection.json").is_file():
        raise SystemExit(
            "尚未封存：运行目录里没有 locked_selection.json。\n"
            "B 的协议要求先 score-validation、再 seal，之后才允许测试推理。"
        )
    if (base / "pred.npy").is_file():
        raise SystemExit("pred.npy 已存在，不覆盖已有测试预测。")

    params = build_params(row, args.num_workers, args.epochs)
    params["is_training"] = 0        # 只做测试推理，加载已有 checkpoint
    params.pop("skip_test", None)
    argv = params_to_argv(params)
    setting = find_setting_dir(params["model_id"])
    if not setting:
        raise SystemExit("找不到 checkpoint 目录，无法做测试推理。")

    print(f"测试推理：{args.run_id}")
    if args.dry_run:
        print(params_to_command_text(params))
        return 0

    log = (base / "test.log").open("w", encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, "-u", *argv], cwd=str(REPO),
                           stdout=log, stderr=subprocess.STDOUT, text=True)
    finally:
        log.close()
    if r.returncode != 0:
        print(f"FAIL: 测试退出码 {r.returncode}，见 {base / 'test.log'}")
        return r.returncode

    res = REPO / "results" / setting
    for src, dst in (("pred.npy", "pred.npy"), ("true.npy", "true.npy")):
        if not (res / src).is_file():
            print(f"FAIL: 官方测试循环没有产出 {res / src}")
            return 1
        shutil.copy2(res / src, base / dst)

    H = int(row.horizon)
    # 测试段 border1s[2] = 34560 + 11520 - seq_len，首样本最后已观测行 = 46079
    n_test = int(np.load(base / "pred.npy", mmap_mode="r").shape[0])
    origin = 46079 + np.arange(n_test, dtype=np.int64)

    # origin_rows.npy 由 B 的 new-run 预置，且会在 seal 时被写进哈希。
    # 只要数值一致就保留原文件字节不动，避免封存后哈希变化导致 evaluate 失败。
    target = base / "origin_rows.npy"
    if target.is_file():
        old = np.load(target)
        if old.shape != origin.shape or not np.array_equal(old, origin):
            print("FAIL: 计算出的测试起报行号与预置索引不一致，说明测试导出顺序有问题；"
                  "不覆盖已封存文件。")
            return 1
        print(f"      origin_rows.npy 与预置索引一致（{n_test} 行），保留原文件不改动")
    else:
        np.save(target, origin, allow_pickle=False)
        print(f"      origin_rows.npy 已生成（{n_test} 行）")
    print(f"      pred/true 已拷入运行目录")

    if not verify_origins(base, "test", H, 46079):
        return 1
    print("\n测试阶段完成，下一步交给成员 B 执行 evaluate。")
    return 0


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["train", "test"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker 数（官方默认 10；Windows 上 4 更稳）")
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆盖官方 epochs，仅用于排错；正式运行不要传")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = ap.parse_args()
    return cmd_train(args) if args.phase == "train" else cmd_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
