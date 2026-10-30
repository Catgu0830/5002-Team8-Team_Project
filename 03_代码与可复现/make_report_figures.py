"""为报告与 PPT 生成干净的图表（与成员 B 的 `08_*` 图并存，互不覆盖）。

为什么另出一套
--------------
B 的 `08_02_timesnet_sensitivity` 左右两列（MSE / MAE）子图标题相同，看图时
容易混；而该脚本的 sha256 已记进每次运行的 `input_verification.json`，
事后修改会让证据链不自洽。所以本脚本**只读**汇总结果，另存到独立目录，
不动 B 的任何产物。

输出目录：work/reports/08_reproduction/figures_report/

使用的数据（全部来自 B 的统一评分工具，未经二次加工）：
  aggregate_metrics.csv / aggregate_by_variable.csv / per_run_metrics.csv
  paper_comparison.csv
  各运行目录下的 train.log（逐轮损失）

用法：
    python tools/make_report_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "work"
REPRO = WORK / "reports/08_reproduction"
PRED = WORK / "experiments/predictions"
OUT = REPRO / "figures_report"

DPI = 240
C_BASE = "#4C78A8"      # 主色
C_ALT = "#E45756"       # 强调/次色
C_MUTED = "#9E9E9E"
C_GOOD = "#54A24B"
C_BAD = "#D62728"


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        tmp = p.with_name(p.name + ".tmp")
        fig.savefig(tmp, format=ext, dpi=DPI, bbox_inches="tight")
        tmp.replace(p)
    plt.close(fig)
    print(f"  已生成 {stem}.png / .pdf")


def agg(metric: str) -> pd.DataFrame:
    return pd.read_csv(REPRO / "aggregate_metrics.csv")


def bar_with_err(ax, xs, means, stds, color=C_BASE, labels=None, width=0.55):
    pos = np.arange(len(xs))
    ax.bar(pos, means, width, yerr=stds, color=color, capsize=6,
           error_kw={"elinewidth": 1.4, "ecolor": "#333333", "capthick": 1.4})
    ax.set_xticks(pos)
    ax.set_xticklabels(labels or xs)
    return pos


# ---------------------------------------------------------------- 图 1
def fig_paper_vs_local() -> None:
    pc = pd.read_csv(REPRO / "paper_comparison.csv").dropna(subset=["local_mse_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, metric, paper_col, local_col in (
        (axes[0], "MSE", "paper_mse", "local_mse_mean"),
        (axes[1], "MAE", "paper_mae", "local_mae_mean"),
    ):
        x = np.arange(len(pc))
        w = 0.36
        ax.bar(x - w / 2, pc[paper_col], w, label="Paper (Table 13)", color=C_MUTED)
        ax.bar(x + w / 2, pc[local_col], w, label="Local (3-seed mean)", color=C_BASE)
        for i, r in enumerate(pc.itertuples()):
            p, l = getattr(r, paper_col), getattr(r, local_col)
            diff = (l - p) / p * 100
            ax.annotate(f"{diff:+.2f}%", (i, max(p, l)), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=C_GOOD if abs(diff) < 1 else C_ALT, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H={int(h)}" for h in pc.horizon])
        ax.set_xlabel("Forecast horizon (15-min steps)")
        ax.set_ylabel(f"{metric} (common train z-score scale)")
        ax.set_title(f"{metric}: paper vs local reproduction")
        ax.legend(fontsize=9, frameon=False)
        ax.set_ylim(0, max(pc[paper_col].max(), pc[local_col].max()) * 1.18)
    fig.suptitle("TimesNet on ETTm2 — reproduction against paper Table 13 (input length 96)",
                 fontsize=13)
    save(fig, "R01_paper_vs_local")


# ---------------------------------------------------------------- 图 2
def fig_input_length() -> None:
    a = pd.read_csv(REPRO / "aggregate_metrics.csv")
    s = a[(a.horizon == 336) & (a.input_scaling == "zscore")].sort_values("input_length")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, metric in ((axes[0], "mse_common_zscore"), (axes[1], "mae_common_zscore")):
        m = s[metric + "_mean"].to_numpy()
        sd = s[metric + "_sample_std"].to_numpy()
        pos = bar_with_err(ax, s.input_length, m, sd,
                           color=[C_BASE, C_GOOD, C_BASE],
                           labels=[f"L={int(v)}\n({int(v)*15/60/24:.0f} day{'s' if v>96 else ''})"
                                   for v in s.input_length])
        base = m[0]
        for i, v in enumerate(m):
            pct = (v - base) / base * 100
            txt = "baseline" if i == 0 else f"{pct:+.1f}%"
            ax.annotate(txt, (pos[i], v + sd[i]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=10,
                        color="black" if i == 0 else (C_GOOD if pct < 0 else C_BAD),
                        fontweight="bold")
        ax.set_ylabel(metric.replace("_common_zscore", "").upper()
                      + " (common train z-score scale)")
        ax.set_title(metric.startswith("mse") and "MSE vs input length (H=336)"
                     or "MAE vs input length (H=336)")
        ax.set_ylim(m.min() * 0.88, (m + sd).max() * 1.12)
    fig.suptitle("Sensitivity 1 — input window length (error bars = sample SD across 3 seeds)",
                 fontsize=13)
    save(fig, "R02_input_length")


# ---------------------------------------------------------------- 图 3
def fig_scaling() -> None:
    a = pd.read_csv(REPRO / "aggregate_metrics.csv")
    s = a[(a.horizon == 336) & (a.input_length == 96)].sort_values("input_scaling")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), layout="constrained")
    for ax, metric in ((axes[0], "mse_common_zscore"), (axes[1], "mae_common_zscore")):
        m = s[metric + "_mean"].to_numpy()
        sd = s[metric + "_sample_std"].to_numpy()
        pos = bar_with_err(ax, s.input_scaling, m, sd,
                           color=[C_BASE, C_BASE],
                           labels=[v.capitalize() for v in s.input_scaling])
        diff = abs(m[1] - m[0])
        ax.annotate(f"|diff| = {diff:.4f}\nseed SD = {max(sd):.4f}\n"
                    + ("diff < SD → indistinguishable"
                       if diff < max(sd) else "diff > SD"),
                    (0.5, 0.06), xycoords="axes fraction", ha="center", fontsize=9.5,
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF4E5",
                          "edgecolor": "#E0A030"})
        ax.set_ylabel(metric.replace("_common_zscore", "").upper()
                      + " (common train z-score scale)")
        ax.set_title(metric.startswith("mse") and "MSE vs external scaling (L=96, H=336)"
                     or "MAE vs external scaling (L=96, H=336)")
        ax.set_ylim(m.min() * 0.9, (m + sd).max() * 1.1)
    fig.suptitle("Sensitivity 2 — external scaling: difference is smaller than seed noise",
                 fontsize=13)
    save(fig, "R03_scaling")


# ---------------------------------------------------------------- 图 4
def fig_per_variable() -> None:
    v = pd.read_csv(REPRO / "aggregate_by_variable.csv")
    sel = v[(v.horizon == 336) & (v.input_scaling == "zscore")]
    piv = sel.pivot_table(index="variable", columns="input_length",
                          values="mse_common_zscore_mean")
    order = piv[96].sort_values(ascending=False).index
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    x = np.arange(len(piv))
    w = 0.26
    cols = [96, 192, 384]
    colors = [C_MUTED, C_GOOD, C_ALT]
    for i, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (i - 1) * w, piv[c], w, label=f"L={c}", color=col)
    for j, var in enumerate(piv.index):
        d192 = (piv.loc[var, 192] / piv.loc[var, 96] - 1) * 100
        d384 = (piv.loc[var, 384] / piv.loc[var, 96] - 1) * 100
        ax.annotate(f"{d384:+.1f}%", (x[j] + w, piv.loc[var, 384]),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8.5, fontweight="bold",
                    color=C_BAD if d384 > 0 else C_GOOD)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("MSE (common train z-score scale)")
    ax.set_title("Per-variable effect of input length at H=336 "
                 "(labels = change of L=384 vs L=96)", fontsize=12)
    ax.legend(fontsize=9, frameon=False, ncol=3)
    # 标出 LUFL 异常
    if "LUFL" in piv.index:
        j = list(piv.index).index("LUFL")
        ax.annotate("LUFL: the only variable\nthat gets worse with\nlonger history",
                    (x[j] + w, piv.loc["LUFL", 384]),
                    xytext=(x[j] - 0.1, piv.loc["LUFL", 384] * 1.55),
                    textcoords="data", fontsize=9, color=C_BAD,
                    arrowprops={"arrowstyle": "->", "color": C_BAD, "lw": 1.2})
    save(fig, "R04_per_variable_input_length")


# ---------------------------------------------------------------- 图 5
def fig_training_curves() -> None:
    """逐轮验证损失。

    重要：两个子图的 y 轴**不在同一空间**。Z-score 组的加载器用 StandardScaler，
    验证损失在 z-score 空间（H336 约 0.22）；Min-Max 组关闭了加载器缩放，
    验证损失在 Min-Max 空间（约 0.0038）。因此**不能跨子图比较水平高低**，
    本图只用于展示"到达各自最优所需轮数"的差异。
    """
    rx = re.compile(r"Epoch: (\d+), Steps: \d+ \| Train Loss: ([-\d.eE+]+) "
                    r"Vali Loss: ([-\d.eE+]+)")
    groups = {
        ("L96_H336 zscore", "validation loss in z-score space"):
            [f"TimesNet_L96_H336_zscore_s{s}" for s in (2021, 2026, 2027)],
        ("L96_H336 minmax", "validation loss in Min-Max space"):
            [f"TimesNet_L96_H336_minmax_s{s}" for s in (2021, 2026, 2027)],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, ((title, space), rids) in zip(axes, groups.items()):
        bests, stops = [], []
        for rid in rids:
            log = PRED / rid / "train.log"
            if not log.is_file():
                continue
            rows = [(int(m.group(1)), float(m.group(3)))
                    for m in rx.finditer(log.read_text(encoding="utf-8", errors="replace"))]
            if not rows:
                continue
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            seed = rid.rsplit("_s", 1)[1]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=f"seed {seed}", alpha=0.85)
            bi = int(np.argmin(ys))
            ax.plot(xs[bi], ys[bi], marker="*", ms=15, color=C_GOOD, zorder=5)
            bests.append(f"{seed}: ep{xs[bi]}")
            stops.append(f"{seed}: ep{xs[-1]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss (MSE)")
        ax.set_title(f"{title}\n({space}; patience = 3)", fontsize=11)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        ax.grid(alpha=0.25)
        # 底部留出空白带，避免信息框盖住低值区的星标
        lo, hi = ax.get_ylim()
        span = hi - lo
        ax.set_ylim(lo - span * 0.20, hi + span * 0.05)
        # 用文字框统一列出，避免逐点标注在低值区互相重叠
        ax.text(0.02, 0.02,
                "★ best epoch — " + " | ".join(bests) + "\n"
                "early stop at — " + " | ".join(stops),
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                      "edgecolor": "#BBBBBB", "alpha": 0.95})
    fig.suptitle("Validation loss per epoch — Min-Max keeps improving marginally and only "
                 "stops at the 10-epoch cap (seeds 2021/2026),\nwhile z-score stops at "
                 "epoch 4-5; each panel is in its own external scale, so do NOT compare "
                 "levels across panels", fontsize=11)
    save(fig, "R05_training_curves")


def main() -> int:
    print(f"输出目录: {OUT}")
    fig_paper_vs_local()
    fig_input_length()
    fig_scaling()
    fig_per_variable()
    fig_training_curves()
    n = len(list(OUT.glob("*")))
    print(f"\n完成，共 {n} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
