"""批量跑完成员 A 的全部运行，每一次都走完整的协议链路。

顺序（每一次运行内部）
----------------------
    train          训练 + 验证 + 导出验证预测（不做测试推理）
    score-validation  用 B 的工具算验证集分数
    seal           封存配置（此后不得改动训练产物）
    test           加载 checkpoint 做测试推理并导出 pred/true
    evaluate       用 B 的工具算测试指标并产出图表数据

为什么每次都要跑完整链路
------------------------
只在最后统一评分的话，万一测试导出或评分环节有系统性问题，会等到全部
跑完才发现。每次立即闭环，出错时只损失一次运行。

可恢复
------
每一步都先检查产物是否存在，已完成的步骤自动跳过。中断后重跑本脚本
会从断点继续，不会重复训练。

用法
----
    python tools/run_batch.py                 # 跑所有未完成的 A 运行
    python tools/run_batch.py --limit 3       # 只跑接下来 3 次（试水）
    python tools/run_batch.py --only RUN_ID[,RUN_ID...]
    python tools/run_batch.py --dry-run       # 只列出将要跑什么
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                            # memberA_timesnet/
WORK = ROOT / "work"
PRED_DIR = WORK / "experiments/predictions"
RUNS_DIR = WORK / "reports/08_reproduction/runs"
SCRIPTS_B = WORK / "scripts/08_reproduction_comparison.py"
RUN_PROTOCOL = HERE / "run_protocol.py"
LOG_DIR = ROOT / "logs"


# ---------------------------------------------------------------- 阻止睡眠
def keep_awake() -> bool:
    """在本次进程存活期间阻止系统自动睡眠。

    为什么需要：这台机器"交流电下 15 分钟睡眠"。Windows 的睡眠计时只看
    用户输入空闲，不看 CPU/GPU 是否在算，所以一个跑 13 小时的训练会被
    系统睡眠直接掐死。

    用 Windows 官方的 SetThreadExecutionState 而不是改电源计划：
    前者只在本进程存活期间生效、退出即释放，不改动任何持久设置。

    注意：**合盖仍会睡眠**（受"合盖动作"策略控制），这一条程序管不了。
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        # 不使用 ES_DISPLAY_REQUIRED，允许关显示器（关屏不影响训练）
        prev = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        return prev != 0
    except Exception:  # noqa: BLE001
        return False


def release_awake() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
    except Exception:  # noqa: BLE001
        pass


def sh(cmd: list[str], log_path: Path) -> int:
    """执行子命令，输出**同时**追加到日志文件和当前控制台。

    用逐行流式读取而不是一次性 capture，这样批量跑的时候也能实时看到
    每一轮的 loss/用时（子进程用 -u 启动，否则管道下 print 会被块缓冲，
    输出要等很久才成批出现）。
    """
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 78 + "\n")
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        fh.write("=" * 78 + "\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT.parent), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            print(line, end="", flush=True)
        proc.wait()
    return proc.returncode


def step_done(run_id: str) -> dict[str, bool]:
    base = PRED_DIR / run_id
    return {
        "train": (base / "validation_pred.npy").is_file()
                 and (base / "checkpoint.pth").is_file(),
        "score": (base / "validation_score.json").is_file(),
        "seal": (base / "locked_selection.json").is_file(),
        "test": (base / "pred.npy").is_file() and (base / "true.npy").is_file(),
        "eval": (RUNS_DIR / run_id / "metrics.json").is_file(),
    }


def run_one(run_id: str, py: str, idx: int, total: int) -> tuple[bool, str]:
    log = LOG_DIR / f"{run_id}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    done = step_done(run_id)
    t0 = time.time()

    print(f"\n[{idx}/{total}] {run_id}")
    print(f"        已完成步骤: " +
          (", ".join(k for k, v in done.items() if v) or "无"))

    plan = [
        ("train", [py, "-u", str(RUN_PROTOCOL), "train", "--run-id", run_id]),
        ("score", [py, "-u", str(SCRIPTS_B), "score-validation", "--run-id", run_id,
                   "--project-root", str(WORK)]),
        ("seal", [py, "-u", str(SCRIPTS_B), "seal", "--run-id", run_id,
                  "--project-root", str(WORK)]),
        ("test", [py, "-u", str(RUN_PROTOCOL), "test", "--run-id", run_id]),
        ("eval", [py, "-u", str(SCRIPTS_B), "evaluate", "--run-id", run_id,
                  "--project-root", str(WORK)]),
    ]

    for name, cmd in plan:
        if done[name]:
            print(f"        跳过 {name}（已完成）")
            continue
        # 后续步骤的前置条件
        if name in ("score", "seal", "test", "eval") and not done["train"] and name != "train":
            print(f"        FAIL: 前置步骤 train 未完成，停止本运行")
            return False, "前置 train 未完成"
        if name in ("seal", "test", "eval") and not done["score"]:
            print(f"        FAIL: 前置步骤 score-validation 未完成")
            return False, "前置 score 未完成"

        print(f"        → {name} …", end="", flush=True)
        rc = sh(cmd, log)
        if rc != 0:
            print(f" 失败 (exit {rc})")
            print(f"        详见 {log}")
            return False, f"{name} exit {rc}"
        done[name] = True
        print(" ok")

    dur = time.time() - t0
    print(f"        完成，用时 {dur / 60:.1f} 分钟")
    return True, f"{dur / 60:.1f} min"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="最多跑几次")
    ap.add_argument("--only", type=str, default=None, help="逗号分隔的 run_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = pd.read_csv(WORK / "artifacts/evaluation/v1/run_plan.csv")
    mine = plan.loc[plan.owner == "A"].copy()

    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        mine = mine.loc[mine.run_id.isin(wanted)]
        missing = set(wanted) - set(mine.run_id)
        if missing:
            raise SystemExit(f"不在 A 的运行计划里：{sorted(missing)}")

    todo, finished = [], []
    for rid in mine.run_id:
        d = step_done(rid)
        (finished if all(d.values()) else todo).append(rid)

    if args.limit:
        todo = todo[: args.limit]

    print(f"A 的运行共 {len(mine)} 次：已完成 {len(finished)}，待跑 {len(todo)}")
    if finished:
        print("  已完成：" + ", ".join(finished))
    if not todo:
        print("没有待跑的运行。")
        return 0

    print("\n待跑顺序：")
    for i, rid in enumerate(todo, 1):
        print(f"  {i:>2}. {rid}")

    if args.dry_run:
        print("\n（--dry-run，不执行）")
        return 0

    py = sys.executable
    started = datetime.now()
    print(f"\n开始时间 {started:%Y-%m-%d %H:%M:%S}   解释器 {py}")

    awake = keep_awake()
    print("系统睡眠抑制 : " + ("已启用（仅本进程存活期间生效，退出自动释放）"
                              if awake else "未能启用 —— 请手动把电源计划设为“从不睡眠”"))
    print("注意         : 合盖仍会睡眠，跑批期间请不要合上笔记本")

    results: list[tuple[str, bool, str]] = []
    try:
        for i, rid in enumerate(todo, 1):
            ok, msg = run_one(rid, py, i, len(todo))
            results.append((rid, ok, msg))
            if not ok:
                print(f"        ⚠ 本运行失败，按设计继续下一条（可事后重跑本脚本续跑）")
    finally:
        release_awake()

    elapsed = (datetime.now() - started).total_seconds()
    ok_n = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 78)
    print(f"批量结束：成功 {ok_n} / {len(results)}，总用时 {elapsed / 3600:.2f} 小时")
    print("=" * 78)
    for rid, ok, msg in results:
        print(f"  {'✓' if ok else '✗'} {rid:<44} {msg}")
    if ok_n != len(results):
        print("\n失败项可用 `python tools/run_batch.py` 重跑，已完成的步骤会自动跳过。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
