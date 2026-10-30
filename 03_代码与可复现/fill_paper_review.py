"""填写 `paper_comparability_review.csv`：逐项核对论文条件与本地条件。

为什么必须填
------------
成员 B 的 `paper_comparison.csv` 只报"数值差"，并明确标注
`NUMERICAL_DIFFERENCE_ONLY; consult paper_comparability_review.csv`。
也就是说**数值接近不等于复现成功**，必须先证明两边条件可比。
这张表就是那一步的正式依据。

填表原则
--------
- `paper_setting`：只在论文或官方代码里**能查到**的才写；查不到就写
  `NOT_STATED_IN_PAPER`，**不猜**。
- `local_setting`：写真实执行条件，与每次运行的
  `command.txt` / `hyperparameters.json` / `environment.txt` 一致。
- `status`：用固定词表，便于检索与统计。
- `difference_and_implication`：说清差异**会不会**影响可比性。

用法：
    python tools/fill_paper_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "work"
CSV = WORK / "reports/08_reproduction/paper_comparability_review.csv"


def build(item: str, h: int) -> tuple[str, str, str, str]:
    """返回 (paper_setting, local_setting, status, difference_and_implication)。"""

    if item == "source_and_split":
        return (
            "ETTm2 benchmark split: 12/4/4 month-blocks, 30-day month, chronological, no shuffling",
            "Same split, inherited unchanged from member B's phase-1 protocol: "
            "train [0,34560), validation [34560,46080), test [46080,57600), "
            "12,080 tail rows unused",
            "MATCH",
            "No difference. Boundaries are identical to the official Dataset_ETT_minute borders.",
        )

    if item == "task_features_M":
        return (
            "Multivariate -> multivariate, 7 variables (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)",
            "Same: --features M, --enc_in/--dec_in/--c_out 7, target OT",
            "MATCH",
            "No difference.",
        )

    if item == "lookback_and_horizon":
        return (
            "Input length 96; forecast lengths 96 / 192 / 336 / 720",
            f"Input length 96 (unchanged); this run forecasts H={h}",
            "MATCH",
            "No difference in lookback or horizon definition.",
        )

    if item == "origin_set":
        return (
            "NOT_STATED_IN_PAPER (the paper reports aggregate MSE/MAE only and does not "
            "document how many forecast origins are evaluated)",
            "Official loader borders, which borrow seq_len points of history before each "
            "split boundary. Origin counts: 11,425 / 11,329 / 11,185 / 10,801 for "
            "H=96/192/336/720; verified row-by-row against member B's "
            "expected_origins_test_H*.npy",
            "NOT_VERIFIABLE_FROM_PAPER",
            "Cannot be confirmed as identical. Local follows the official implementation exactly, "
            "so the comparison is 'same code path' rather than 'proven same origins'.",
        )

    if item == "external_scaling":
        return (
            "NOT_STATED_IN_PAPER explicitly for ETTm2; the official code applies a "
            "StandardScaler fitted on the training rows",
            "StandardScaler fitted on training rows only, applied exactly once by the data "
            "loader (--scale 1). No second transform anywhere downstream.",
            "CONSISTENT_WITH_OFFICIAL_CODE",
            "Matches the official code path. The paper text does not state it, so this is "
            "verified against implementation rather than against the paper.",
        )

    if item == "internal_normalization":
        return (
            "TimesNet normalises each input window by its own mean and standard deviation "
            "inside forecast() (Non-stationary Transformer style)",
            "Preserved unchanged; the second normalisation layer is active in every run",
            "MATCH",
            "No difference. This is also why external scaling choices have little effect "
            "(see the scaling ablation).",
        )

    if item == "model_architecture":
        dm = 16 if h == 720 else 32
        note = (
            "H=720 uses d_model 16 instead of 32, exactly as the official ETTm2 script does"
            if h == 720
            else "d_model 32"
        )
        return (
            "TimesNet: 2 TimesBlocks, d_model 32, d_ff 32, top_k 5, num_kernels 6 "
            "('Embedding' layer plus Inception-style 2D convolutions)",
            f"Same structure: e_layers 2, d_ff 32, top_k 5, num_kernels 6, {note}",
            "MATCH_WITH_OFFICIAL_SCRIPT",
            ("Architecture follows the official ETTm2 script for this horizon. "
             + ("The d_model 16 choice for H=720 comes from that script; whether the paper "
                "table used the same setting cannot be verified from the text."
                if h == 720 else "No architectural difference.")),
        )

    if item == "optimization_and_early_stopping":
        ep = 1 if h in (192, 720) else 10
        return (
            "Adam, learning rate 1e-4, MSE loss, batch size 32, lradj 'type1', "
            "early stopping with patience 3. The paper does not state a per-setting epoch budget.",
            f"Identical optimiser/loss/lr/batch/patience. train_epochs = {ep} for H={h}, "
            f"taken verbatim from the official ETTm2 script"
            + (" (which caps this horizon at a single epoch)" if ep == 1 else ""),
            "MATCH_WITH_OFFICIAL_SCRIPT",
            ("The epoch budget is script-defined rather than paper-defined. "
             + ("H=192 and H=720 are capped at 1 epoch by the official script, so those two "
                "numbers reflect an under-trained official setting, not the model's ceiling."
                if ep == 1 else
                "Early stopping decided the actual number of epochs in every run.")),
        )

    if item == "code_version_differences":
        return (
            "Code as of the ICLR 2023 release of thuml/Time-Series-Library",
            "Commit 4e938a1767106324dd753b2a44832bf870a0252e (2026-04) plus 10 documented "
            "patches; run_metadata.json has code_dirty = true and patches.diff is shipped "
            "with every run",
            "DIFFERENT",
            "Implementation has evolved between releases. The 10 patches only touch dependency "
            "loading (lazy imports), seed parameterisation, separating training from test "
            "inference, and a --scale switch for the Min-Max variant. No change to model math, "
            "loss or optimiser. This is the main residual risk for the reproduction claim.",
        )

    if item == "seed_and_hardware_differences":
        return (
            "NOT_STATED_IN_PAPER per setting (single aggregate numbers are reported; "
            "seeding and hardware are not disclosed)",
            "Three seeds: 2021 (the value hard-coded in the official run.py), 2026 and 2027. "
            "Single RTX 3060 Laptop GPU (6 GB). CUDA 11.8, torch 2.7.1+cu118, cuDNN 90100.",
            "PARTIALLY_UNKNOWN",
            "Seed handling cannot be aligned with the paper. Reporting a 3-seed mean +/- sample SD "
            "instead of a single number is a stricter, not looser, comparison. cuDNN "
            "non-determinism means bit-identical cross-machine reproduction is not expected.",
        )

    return (
        "The paper reports MSE and MAE over the test set; the exact averaging is not documented",
        "MSE/MAE recomputed by member B's evaluator: predictions and truth are inverse-"
        "transformed to raw units, then rescaled by the training-period standard deviation. "
        "Every origin-lead-variable triple is weighted equally. OT errors are additionally "
        "reported in raw units.",
        "LIKELY_EQUIVALENT_BUT_UNVERIFIED",
        "Equal weighting per triple can differ slightly from an implementation that averages "
        "batch means. This is one plausible source of the sub-percent residual differences.",
    )


def main() -> int:
    if not CSV.is_file():
        raise SystemExit(f"找不到 {CSV}")

    df = pd.read_csv(CSV)
    required = {"horizon", "item", "paper_setting", "local_setting", "status",
                "difference_and_implication"}
    if not required.issubset(df.columns):
        raise SystemExit(f"列不符，实际列：{list(df.columns)}")

    for i, row in df.iterrows():
        paper, local, status, diff = build(str(row["item"]), int(row["horizon"]))
        df.at[i, "paper_setting"] = paper
        df.at[i, "local_setting"] = local
        df.at[i, "status"] = status
        df.at[i, "difference_and_implication"] = diff

    df.to_csv(CSV, index=False, float_format="%.17g", lineterminator="\n",
              encoding="utf-8")

    print(f"已填写 {len(df)} 行 -> {CSV}")
    print()
    print("按状态统计：")
    print(df.status.value_counts().to_string())
    print()
    print("按条目统计（是否还有未填）：")
    pend = df[(df.local_setting.astype(str).str.startswith("PENDING"))
              | (df.status.astype(str).str.startswith("PENDING"))]
    print(f"  仍为 PENDING 的行数: {len(pend)}")
    return 0 if len(pend) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
