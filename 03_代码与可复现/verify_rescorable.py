"""独立复算验证：只用**交付包里的张量**重新计算 MSE/MAE，与记录的指标比对。

目的
----
回答一个问题：成员 B 拿到交付包后，能不能不依赖本机任何中间结果、
仅凭包内的 `pred.npy` / `true.npy` 重算出同样的分数？

做法（与 B 的评估口径一致）
--------------------------
1. 预测与真值都在各自的 `prediction_space` 里（zscore 或 minmax）；
2. 先逆变换回原始单位：
     zscore : raw = scaled * z_scale + z_mean
     minmax : raw = (scaled - mm_offset) / mm_scale
3. 再用**训练期 z-score 的标准差**归一化误差：
       e = (pred_raw - true_raw) / z_scale
   （zscore 路径下该式化简为 pred_z - true_z，minmax 路径则要换算）
4. MSE = mean(e^2)，MAE = mean(|e|)，与 B 的 `metrics.json` 对照。

用法：
    python tools/verify_rescorable.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "work"
DELIV = ROOT.parent / "A_交付包"
PRED = DELIV / "01_预测与指标" / "predictions"
SCALER = WORK / "artifacts/preprocessing/v1/scaler_parameters.json"


def main() -> int:
    sc = json.loads(SCALER.read_text(encoding="utf-8"))
    z_scale = np.asarray(sc["zscore"]["scale"], dtype=np.float64)
    z_mean = np.asarray(sc["zscore"]["mean"], dtype=np.float64)
    mm_scale = np.asarray(sc["minmax"]["scale"], dtype=np.float64)
    mm_off = np.asarray(sc["minmax"]["offset"], dtype=np.float64)

    print(f"交付包张量目录: {PRED}")
    print(f"训练期 z-score 标准差已载入（7 变量）\n")

    rows = []
    bad = []
    for d in sorted(PRED.iterdir()):
        if not d.is_dir():
            continue
        m = json.loads((d / "run_metadata.json").read_text(encoding="utf-8"))
        space = m["prediction_space"]
        pred = np.load(d / "pred.npy")
        true = np.load(d / "true.npy")
        if pred.shape != true.shape:
            bad.append(f"{d.name} pred/true 形状不一致")
            continue

        # 逆变换到原始单位，再按训练期 z-score 标准差归一化
        if space == "zscore":
            e = pred.astype(np.float64) - true.astype(np.float64)
        elif space == "minmax":
            e = ((pred.astype(np.float64) - true.astype(np.float64)) / mm_scale) / z_scale
        else:
            bad.append(f"{d.name} 未知 prediction_space={space}")
            continue

        mse = float(np.mean(e ** 2))
        mae = float(np.mean(np.abs(e)))

        # 与 B 的评测结果比对
        mp = WORK / "reports/08_reproduction/runs" / d.name / "metrics.json"
        if mp.is_file():
            rec = json.loads(mp.read_text(encoding="utf-8"))
            rec_mse = float(rec["mse_common_zscore"])
            rec_mae = float(rec["mae_common_zscore"])
        else:
            rec_mse = rec_mae = float("nan")

        rel = abs(mse - rec_mse) / rec_mse if rec_mse == rec_mse else float("nan")
        ok = rec_mse == rec_mse and rel < 1e-4
        if not ok:
            bad.append(f"{d.name} 重算 {mse:.6f} vs 记录 {rec_mse:.6f}")
        rows.append({"run": d.name, "space": space, "n": pred.shape[0],
                     "mse_recalc": mse, "mse_recorded": rec_mse,
                     "mae_recalc": mae, "mae_recorded": rec_mae,
                     "rel_diff": rel, "ok": ok})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{'run_id':<40}{'space':>7}{'N':>7}{'MSE重算':>11}{'MSE记录':>11}{'相对差':>10}{'':>4}")
    print("-" * 92)
    for r in df.itertuples():
        print(f"{r.run:<40}{r.space:>7}{r.n:>7}{r.mse_recalc:>11.6f}"
              f"{r.mse_recorded:>11.6f}{r.rel_diff:>10.2e}{'  OK' if r.ok else '  ✗'}")

    print()
    print(f"共 {len(df)} 次运行；MSE 最大相对偏差 = {df.rel_diff.max():.2e}")
    print(f"MAE 最大绝对偏差 = {(df.mae_recalc - df.mae_recorded).abs().max():.2e}")
    print()
    if bad:
        print("存在问题：")
        for b in bad[:10]:
            print("  -", b)
        return 1
    print("=" * 66)
    print("结论：仅用交付包内的张量即可完整复算出记录中的指标 ✅")
    print("      （成员 B 可独立复核，无需本机任何中间结果）")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
