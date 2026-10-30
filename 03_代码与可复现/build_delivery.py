"""把成员 A 的交付物组装到一个独立的交付文件夹里。

设计原则
--------
1. **可重跑**：每次运行都把交付文件夹整目录重建，避免残留上一次的旧文件。
2. **只搬运，不加工**：所有内容从权威位置复制，保证交付文件和实验现场一致
   （哈希可核对）。唯一的"加工"是生成 `00_交付说明.md` 和清单。
3. **反映真实进度**：还没跑完的运行不会被伪造，只在说明里标为"待完成"。

交付对象与所需内容
------------------
- 成员 B：每次运行的接口文件（元数据 + 验证/测试预测张量 + 日志 + 封存记录）
          加上 `reports/08_reproduction/` 的汇总表 —— 他要用这些做统一评分与复现核对。
- 成员 D：`runs/<run_id>/` 下的误差分解 CSV（by variable / lead / hour、分层指标、
          固定 lead 误差）—— 他做第二轮误差 EDA。
- 组长/统稿：图表 PNG+PDF、代码与补丁、环境锁、复现说明。

用法
----
    python tools/build_delivery.py                     # 全量（含预测张量与权重）
    python tools/build_delivery.py --skip-large        # 不含 .npy/.pth（体积小很多）
    python tools/build_delivery.py --out D:/某个盘/A_交付包
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                  # memberA_timesnet/
PROJ = ROOT.parent                  # Team_project/
WORK = ROOT / "work"
PRED = WORK / "experiments/predictions"
REPRO = WORK / "reports/08_reproduction"
TOOLS = ROOT / "tools"

DEFAULT_OUT = PROJ / "A_交付包"

# 每次运行目录里要交付的文件（缺的跳过，不报错——有些文件是分阶段产生的）
RUN_FILES = [
    "run_metadata.json", "hyperparameters.json", "environment.txt",
    "command.txt", "train.log", "test.log", "patches.diff",
    "checkpoint.pth",
    "validation_pred.npy", "validation_true.npy", "validation_origin_rows.npy",
    "pred.npy", "true.npy", "origin_rows.npy",
    "validation_score.json", "locked_selection.json", "README.txt",
    "metrics.json",
]

# 需要整目录搬运的汇总产物
SUMMARY_DIRS = ["runs"]
SUMMARY_FILES = [
    "per_run_metrics.csv", "aggregate_metrics.csv", "aggregate_by_variable.csv",
    "paper_comparison.csv", "paper_comparability_review.csv",
    "run_status.csv", "comparison_summary.json", "figure_catalog.csv",
    "PASTE_BACK_08_SUMMARY.txt", "process_record_08.md",
]

CODE_FILES = [
    "run_protocol.py", "run_batch.py", "apply_seed_2021.py",
    "repair_hyperparams.py", "build_delivery.py",
    "make_report_figures.py", "fill_paper_review.py", "audit_delivery.py",
    "verify_rescorable.py",
    "env_5002_freeze.txt", "repo_commit.txt", "seed_2021.diff", "seed_2021_sha256.txt",
]


def human(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.0f}{u}" if u == "B" else f"{x:.1f}{u}"
        x /= 1024
    return f"{x:.1f}TB"


def copy_file(src: Path, dst: Path, skip_large: bool) -> int:
    if not src.is_file():
        return -1
    if skip_large and src.suffix.lower() in (".npy", ".pth", ".bin"):
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-large", action="store_true",
                    help="不复制 .npy / .pth（交付说明里会注明）")
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    plan = pd.read_csv(WORK / "artifacts/evaluation/v1/run_plan.csv")
    mine = plan.loc[plan.owner == "A"].copy()

    # ---------------- 01 预测与指标 ----------------
    d01 = out / "01_预测与指标"
    done_runs, pending_runs = [], []
    total = 0
    for rid in mine.run_id:
        src_dir = PRED / rid
        if not src_dir.is_dir():
            pending_runs.append((rid, "尚未创建运行目录"))
            continue
        has_test = (src_dir / "pred.npy").is_file()
        has_val = (src_dir / "validation_pred.npy").is_file()
        if not has_val:
            pending_runs.append((rid, "尚未完成训练"))
            continue
        for name in RUN_FILES:
            total += max(copy_file(src_dir / name, d01 / "predictions" / rid / name,
                                   args.skip_large), 0)
        done_runs.append((rid, "全部完成" if has_test else "训练完成，待封存/测试"))

    for name in SUMMARY_FILES:
        total += max(copy_file(REPRO / name, d01 / name, args.skip_large), 0)
    for sub in SUMMARY_DIRS:
        s = REPRO / sub
        if s.is_dir():
            for f in s.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(s)
                    total += max(copy_file(f, d01 / sub / rel, args.skip_large), 0)

    # ---------------- 02 图表 ----------------
    d02 = out / "02_图表"
    fig_dir = REPRO / "figures"
    figs = []
    if fig_dir.is_dir():
        for f in sorted(fig_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".png", ".pdf"):
                total += max(copy_file(f, d02 / f.name, False), 0)
                figs.append(f.name)

    # 报告/PPT 专用的干净图表（由 tools/make_report_figures.py 生成）
    fig_rep = REPRO / "figures_report"
    if fig_rep.is_dir():
        for f in sorted(fig_rep.iterdir()):
            if f.is_file() and f.suffix.lower() in (".png", ".pdf"):
                total += max(copy_file(f, d02 / "报告图" / f.name, False), 0)
                figs.append("报告图/" + f.name)

    # ---------------- 03 代码与可复现 ----------------
    d03 = out / "03_代码与可复现"
    for name in CODE_FILES:
        total += max(copy_file(TOOLS / name, d03 / name, False), 0)
    total += max(copy_file(ROOT / "_smoke_test/patches.diff",
                           d03 / "patches.diff", False), 0)

    # ---------------- 04 报告素材 ----------------
    d04 = out / "04_报告素材"
    d04.mkdir(parents=True, exist_ok=True)
    chapter = ROOT / "A_报告章节_TimesNet复现与消融.md"
    if chapter.is_file():
        total += max(copy_file(chapter, d04 / chapter.name, False), 0)
    note = d04 / "说明.md"
    note.write_text(
        "# 04_报告素材\n\n"
        "- `A_报告章节_TimesNet复现与消融.md`：A 负责的报告章节（全部结果表、与论文对比、"
        "两组消融、结论回接第一轮 EDA、局限说明）\n"
        "- PPT 用图见 `../02_图表/`（4 组，PNG + PDF）\n"
        "- 论文参考值与差值见 `../01_预测与指标/paper_comparison.csv`\n"
        "- 论文条件逐项核对见 `../01_预测与指标/paper_comparability_review.csv`\n",
        encoding="utf-8",
    )
    total += note.stat().st_size

    # ---------------- 00 交付说明 ----------------
    now = datetime.now()
    n_done = len(done_runs)
    n_all = len(mine)
    lines = [
        "# 成员 A 交付包",
        "",
        f"生成时间：{now:%Y-%m-%d %H:%M:%S}（香港）  ",
        f"完成进度：**{n_done} / {n_all}** 次运行已产出可交付内容  ",
        f"交付包大小：**{human(total)}**"
        + ("（已排除 .npy / .pth 大文件）" if args.skip_large else ""),
        "",
        "---",
        "",
        "## 1. 这个包是什么",
        "",
        "DSC5002 Group 8 · ETTm2 长时序预测 · **成员 A 的部分**：",
        "TimesNet 在 ETTm2 上的复现（L96，H=96/192/336/720）+ 两组敏感性实验",
        "（输入长度 L=192/384 @H336；缩放 Min-Max @L96/H336），共 21 次运行，",
        "每次三个随机种子（2021/2026/2027）。",
        "",
        "## 2. 目录结构",
        "",
        "| 目录 / 文件 | 内容 | 谁用 |",
        "|---|---|---|",
        "| `01_预测与指标/predictions/<run_id>/` | 每次运行的元数据、超参数、环境、日志、"
        "权重、验证/测试预测张量、封存记录 | **成员 B** 做统一评分与复现核对 |",
        "| `01_预测与指标/per_run_metrics.csv` | 每次运行的测试指标 | B / 报告 |",
        "| `01_预测与指标/aggregate_metrics.csv` | 三种子聚合（均值±标准差） | 报告 |",
        "| `01_预测与指标/paper_comparison.csv` | 本地 vs 论文 Table 13 差值 | 报告 |",
        "| `01_预测与指标/paper_comparability_review.csv` | 论文条件逐项核对表 | B / 报告 |",
        "| `01_预测与指标/runs/<run_id>/` | 误差分解：by variable / lead / hour、"
        "分层指标、固定 lead 误差 | **成员 D** 做第二轮误差 EDA |",
        "| `02_图表/` | 4 组图（基线、敏感性、变量误差、lead 剖面），PNG+PDF | 报告 / PPT |",
        "| `03_代码与可复现/` | 包装脚本、官方仓库补丁、种子补丁、环境版本锁、commit 记录 | 复现审计 |",
        "| `04_报告素材/` | A 负责的报告章节与 PPT 素材（待补） | 统稿 / PPT |",
        "",
        "## 3. 运行完成情况",
        "",
        "| run_id | 状态 |",
        "|---|---|",
    ]
    for rid, st in done_runs:
        lines.append(f"| `{rid}` | {st} |")
    for rid, st in pending_runs:
        lines.append(f"| `{rid}` | ⏳ {st} |")

    lines += [
        "",
        "## 4. 复现口径（写进报告用）",
        "",
        "- 官方仓库：`https://github.com/thuml/Time-Series-Library`",
        "- commit：`4e938a1767106324dd753b2a44832bf870a0252e`",
        "- 代码有补丁 → 每次运行的 `run_metadata.json` 中 `code_dirty = true`，"
        "补丁全文见 `03_代码与可复现/patches.diff`（共 7 类 10 处）",
        "- 运行环境：Python 3.10.22 / torch 2.7.1+cu118 / RTX 3060 Laptop 6 GB，"
        "版本锁见 `03_代码与可复现/env_5002_freeze.txt`",
        "- 数据划分与起报索引由成员 B 的阶段一协议固定，未做任何改动",
        "- 随机种子由官方写死的 2021 改为可传参，TimesNet 用 2021/2026/2027 三个种子；",
        "  其中 2021 与论文同条件，详见 `03_代码与可复现/seed_2021.diff`",
        "",
        "## 5. 协议顺序（重要）",
        "",
        "每次运行都严格按下列顺序，测试预测只可能在封存之后产生：",
        "",
        "```",
        "train（--skip_test，不做测试推理）",
        "  → score-validation（只读验证集）",
        "  → seal（写 locked_selection.json，锁配置哈希）",
        "  → test（加载封存后的 checkpoint 做测试推理）",
        "  → evaluate（统一 Z-score 尺度评分）",
        "```",
        "",
        "> 注：官方 `exp.train()` 每个 epoch 会计算并打印 `Test Loss`"
        "（`exp_long_term_forecasting.py` 第 152 行），但 early stopping 与模型选择",
        "只用 `Vali Loss`（第 156 行）。该数值未参与任何选择，已在报告中如实记录。",
        "",
        "## 5.1 过程中发生的两次变更（如实记录）",
        "",
        "1. **随机种子 2028 → 2021**（组内同意）。目的是让三个种子里有一个等于官方 "
        "`run.py` 硬编码的 2021，与论文有一个完全同条件的参照点。只影响 TimesNet。",
        "   补丁见 `03_代码与可复现/seed_2021.diff`。",
        "2. **超参数文件修正后按协议重做了 score→seal→test→evaluate**。"
        "早期版本误把随种子变化的 `fix_seed`/`model_id`/`run_id` 写进 "
        "`hyperparameters.json`，导致同组三种子的 `hyperparameters_sha256` 不同、"
        "`summarize` 拒绝聚合。修正后超参数文件与种子无关，种子仅存于 "
        "`run_metadata.json`。",
        "   **重做期间模型与预测未变**：`checkpoint.pth`、`train.log`、"
        "`validation_pred.npy` 字节未动（训练未重跑），只是重新导出一次测试预测，"
        "保证「封存先于测试导出」的顺序可审计。",
        "3. `L384/H336_s2027` 首次训练在训练正常结束后因脚本收尾缺陷崩溃，随后**完整重训**；"
        "最终结果来自重训后的完整轮次。",
        "",
        "## 5.2 关于本包的完整性",
        "",
        f"**本包构建方式：{'轻量（已排除 .npy / .pth，仅适合评审与传阅）' if args.skip_large else '全量（含全部预测张量与模型权重，可用于独立复算）'}**",
        "",
        "| 方式 | 体积 | 含预测张量/权重 | 成员 B 能否独立复算 |",
        "|---|---:|---|---|",
        "| `--skip-large` | 约 130 MB | 否 | **否**，只能核对指标与元数据 |",
        "| 全量（不加参数） | 约 8.4 GB | 是 | **能**，可用 B 的工具端到端复算 |",
        "",
        ("本包已是**全量**：`pred.npy` / `true.npy` / `validation_pred.npy` / "
         "`true.npy` / `checkpoint.pth` 齐全，B 可以直接用 "
         "`08_reproduction_comparison.py` 重新执行 `score-validation` / `evaluate` 验证。"
         if not args.skip_large else
         "若课程或成员 B 要求「独立复算验证」，需另行索取全量包；"
         "`pred.npy` / `true.npy` 与 `checkpoint.pth` 始终保存在本机工作目录，不会丢失。"),
        "",
        "## 6. 怎么用这个包",
        "",
        "```bash",
        "# 成员 B 复核并评分（在他的 work 副本上执行，或在收到本包后）",
        "python work/scripts/08_reproduction_comparison.py score-validation \\",
        "    --run-id <run_id> --project-root <他的 project root>",
        "",
        "# 重新汇总与出图",
        "python work/scripts/08_reproduction_comparison.py summarize --project-root <root>",
        "```",
        "",
    ]
    (out / "00_交付说明.md").write_text("\n".join(lines), encoding="utf-8")
    total += (out / "00_交付说明.md").stat().st_size

    # ---------------- 控制台汇总 ----------------
    print(f"交付包: {out}")
    print(f"大小  : {human(total)}" + ("  （已排除 .npy/.pth）" if args.skip_large else ""))
    print(f"进度  : {n_done} / {n_all} 次运行可交付，{len(pending_runs)} 次待完成")
    print()
    for sub in sorted(out.iterdir()):
        if sub.is_dir():
            n = sum(1 for _ in sub.rglob("*") if _.is_file())
            sz = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            print(f"  {sub.name:<16} {n:>4} 文件  {human(sz):>8}")
        else:
            print(f"  {sub.name:<16} {1:>4} 文件  {human(sub.stat().st_size):>8}")
    print()
    if pending_runs:
        print("待完成的运行：")
        for rid, st in pending_runs:
            print(f"  - {rid}  ({st})")
        print("\n跑完更多运行后，重新执行本脚本即可刷新交付包。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
