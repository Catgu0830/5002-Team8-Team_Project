"""交付前审计：逐项检查 21 次运行的产物与交付包是否完整、自洽、可复现。

检查项
------
A 结构   每个运行目录的文件是否齐（按阶段）
B 张量   形状 / dtype / 有限值 / 起报索引是否与 B 的期望逐行一致
C 元数据 commit 是否 40 位、loader 声明是否自洽、prediction_space 是否匹配
D 超参数 是否含种子相关字段、同组哈希是否唯一
E 封存   locked_selection 状态与时间顺序
F 评分   validation_score 状态、是否误读测试集
G 汇总   汇总表是否齐全、paper reviewing 是否还有 PENDING
H 交付包 章节与图片引用路径是否存在、有无 .tmp 残留
I 一致性 重新计算关键数字，与报告章节里的数字对照

用法：
    python tools/audit_delivery.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "work"
PRED = WORK / "experiments/predictions"
REPRO = WORK / "reports/08_reproduction"
DELIV = ROOT.parent / "A_交付包"
CHAPTER = ROOT / "A_报告章节_TimesNet复现与消融.md"

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
H_EXPECT_N = {96: 11425, 192: 11329, 336: 11185, 720: 10801}

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((bool(ok), name, detail))


def main() -> int:
    plan = pd.read_csv(WORK / "artifacts/evaluation/v1/run_plan.csv")
    mine = plan.loc[plan.owner == "A"]
    print(f"A 的运行: {len(mine)} 次\n")

    # ---------------- A 结构 ----------------
    print("=== A. 运行目录结构 ===")
    need_train = ["run_metadata.json", "hyperparameters.json", "environment.txt",
                  "command.txt", "train.log", "checkpoint.pth", "patches.diff",
                  "validation_pred.npy", "validation_true.npy",
                  "validation_origin_rows.npy", "validation_score.json"]
    need_test = ["pred.npy", "true.npy", "origin_rows.npy", "locked_selection.json"]
    missing_total = 0
    for rid in mine.run_id:
        d = PRED / rid
        if not d.is_dir():
            check(False, f"缺少运行目录 {rid}")
            missing_total += 1
            continue
        lack = [f for f in need_train + need_test if not (d / f).is_file()]
        empty = [f for f in need_train + need_test
                 if (d / f).is_file() and (d / f).stat().st_size == 0]
        if lack or empty:
            check(False, f"{rid} 文件不全",
                  f"缺 {lack} 空 {empty}")
            missing_total += 1
    check(missing_total == 0, "21 个运行目录文件齐全", f"问题 {missing_total} 个")

    # ---------------- B 张量 ----------------
    print("=== B. 预测张量与起报索引 ===")
    bad = []
    for _, row in mine.iterrows():
        rid = row.run_id
        H = int(row.horizon)
        d = PRED / rid
        exp_n = H_EXPECT_N[H]
        for name, expect_rows in (("validation_pred.npy", True),
                                  ("validation_true.npy", True),
                                  ("pred.npy", True), ("true.npy", True)):
            p = d / name
            if not p.is_file():
                continue
            a = np.load(p, mmap_mode="r")
            if a.shape != (exp_n, H, 7):
                bad.append(f"{rid}/{name} shape={a.shape} 期望 {(exp_n, H, 7)}")
            if a.dtype != np.float32:
                bad.append(f"{rid}/{name} dtype={a.dtype}")
            # 抽样检查有限值（避免读整个大张量）
            seg = np.asarray(a[:200])
            if not np.isfinite(seg).all():
                bad.append(f"{rid}/{name} 前 200 行含非有限值")
        for name, split in (("validation_origin_rows.npy", "validation"),
                            ("origin_rows.npy", "test")):
            p = d / name
            if not p.is_file():
                continue
            got = np.load(p)
            want = np.load(WORK / f"artifacts/evaluation/v1/"
                                   f"expected_origins_{split}_H{H}.npy")
            if got.shape != want.shape or not np.array_equal(got, want):
                bad.append(f"{rid}/{name} 与 B 的期望索引不一致")
    check(not bad, "全部张量形状/dtype/有限值/起报索引正确",
          "; ".join(bad[:3]) if bad else "")

    # ---------------- C 元数据 ----------------
    print("=== C. run_metadata 自洽性 ===")
    bad = []
    for _, row in mine.iterrows():
        rid = row.run_id
        m = json.loads((PRED / rid / "run_metadata.json").read_text(encoding="utf-8"))
        if not re.fullmatch(r"[0-9a-f]{40}", str(m.get("code_commit", ""))):
            bad.append(f"{rid} code_commit 非 40 位")
        if not isinstance(m.get("code_dirty"), bool):
            bad.append(f"{rid} code_dirty 非布尔")
        scaling = str(row.input_scaling)
        if m.get("prediction_space") != scaling:
            bad.append(f"{rid} prediction_space={m.get('prediction_space')} != {scaling}")
        want_src = "raw_continuous" if scaling == "zscore" else "pre_scaled_continuous"
        if m.get("loader_source") != want_src:
            bad.append(f"{rid} loader_source={m.get('loader_source')} 期望 {want_src}")
        if m.get("loader_scale") is not (want_src == "raw_continuous"):
            bad.append(f"{rid} loader_scale 与 loader_source 逻辑不符")
        if int(m.get("horizon", -1)) != int(row.horizon):
            bad.append(f"{rid} horizon 不符")
        if int(m.get("seed", -1)) != int(row.seed):
            bad.append(f"{rid} seed 不符")
    check(not bad, "全部 metadata 字段合法且自洽", "; ".join(bad[:3]) if bad else "")

    # ---------------- D 超参数 ----------------
    print("=== D. 超参数文件（必须与种子无关）===")
    bad = []
    hashes: dict[tuple, set] = {}
    for _, row in mine.iterrows():
        rid = row.run_id
        p = PRED / rid / "hyperparameters.json"
        raw = p.read_bytes()
        hp = json.loads(raw.decode("utf-8"))
        for k in ("fix_seed", "run_id", "model_id"):
            if k in hp:
                bad.append(f"{rid} 超参数含 {k}")
        h = hashlib.sha256(raw).hexdigest()
        key = (int(row.input_length), int(row.horizon), str(row.input_scaling))
        hashes.setdefault(key, set()).add(h)
    for key, hs in hashes.items():
        if len(hs) != 1:
            bad.append(f"种组 {key} 有 {len(hs)} 种超参数哈希")
    check(not bad, f"{len(hashes)} 个种组超参数哈希各自唯一、无种子字段",
          "; ".join(bad[:3]) if bad else "")

    # ---------------- E/F 封存与评分 ----------------
    print("=== E/F. 封存与评分 ===")
    bad = []
    for rid in mine.run_id:
        d = PRED / rid
        vs = json.loads((d / "validation_score.json").read_text(encoding="utf-8"))
        if vs.get("status") != "PASS":
            bad.append(f"{rid} validation_score 非 PASS")
        if vs.get("test_selection_metric") or vs.get("test_values_used"):
            bad.append(f"{rid} 评分记录显示用了测试集")
        lk = json.loads((d / "locked_selection.json").read_text(encoding="utf-8"))
        if lk.get("status") != "LOCKED_BEFORE_TEST_EXPORT":
            bad.append(f"{rid} 封存状态异常 {lk.get('status')}")
        t_vs = vs.get("time_hong_kong", "")
        t_lk = lk.get("time_hong_kong", "")
        if not (t_vs and t_lk and t_vs <= t_lk):
            bad.append(f"{rid} 时间顺序异常 评分={t_vs} 封存={t_lk}")
    check(not bad, "21 次运行：评分 PASS、封存状态正确、时间顺序合规",
          "; ".join(bad[:3]) if bad else "")

    # ---------------- G 汇总 ----------------
    print("=== G. 汇总产物 ===")
    for name in ("per_run_metrics.csv", "aggregate_metrics.csv", "paper_comparison.csv",
                 "paper_comparability_review.csv", "run_status.csv",
                 "aggregate_by_variable.csv", "comparison_summary.json"):
        check((REPRO / name).is_file(), f"汇总文件存在 {name}")
    pr = pd.read_csv(REPRO / "per_run_metrics.csv")
    check(len(pr) == 21, "per_run_metrics 有 21 行", f"实际 {len(pr)}")
    ag = pd.read_csv(REPRO / "aggregate_metrics.csv")
    check(len(ag) == 7, "aggregate_metrics 有 7 个种组", f"实际 {len(ag)}")
    check(bool(ag.seed_group_complete.all()), "7 个种组都是完整三种子")
    rev = pd.read_csv(REPRO / "paper_comparability_review.csv")
    pend = rev[rev.local_setting.astype(str).str.contains("PENDING")
               | rev.status.astype(str).str.contains("PENDING")]
    check(len(pend) == 0, "paper_comparability_review 已全部填写", f"仍 PENDING {len(pend)} 行")
    rs = pd.read_csv(REPRO / "run_status.csv")
    a_eval = rs[(rs.owner == "A") & (rs.status == "EVALUATED")]
    check(len(a_eval) == 21, "run_status 里 A 的 21 次均标 EVALUATED", f"实际 {len(a_eval)}")

    # ---------------- H 交付包 ----------------
    print("=== H. 交付包 ===")
    check(DELIV.is_dir(), "交付包目录存在")
    check((DELIV / "00_交付说明.md").is_file(), "交付说明存在")
    check((DELIV / "04_报告素材" / CHAPTER.name).is_file(), "报告章节已进交付包")
    for sub in ("01_预测与指标", "02_图表", "03_代码与可复现", "04_报告素材"):
        check((DELIV / sub).is_dir(), f"交付子目录存在 {sub}")
    n_pred = len(list((DELIV / "01_预测与指标/predictions").glob("*")))
    check(n_pred == 21, "交付包含 21 个运行目录", f"实际 {n_pred}")
    tmps = list(DELIV.rglob("*.tmp"))
    check(not tmps, "交付包内无 .tmp 残留", f"{len(tmps)} 个")
    # 章节里的图片引用是否都能在交付包里解析
    if CHAPTER.is_file():
        refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", CHAPTER.read_text(encoding="utf-8"))
        miss = []
        for r in refs:
            target = (DELIV / "04_报告素材" / r).resolve()
            if not target.is_file():
                miss.append(r)
        check(not miss, f"章节引用的 {len(refs)} 张图在交付包内均可解析",
              f"缺失 {miss}" if miss else "")
    else:
        check(False, "报告章节文件存在")

    # ---------------- I 数字复核 ----------------
    print("=== I. 关键数字复核（对照报告章节）===")
    base = ag[(ag.input_length == 96) & (ag.input_scaling == "zscore")]
    recomputed = {
        "H96": float(base[base.horizon == 96].mse_common_zscore_mean.iloc[0]),
        "H192": float(base[base.horizon == 192].mse_common_zscore_mean.iloc[0]),
        "H336": float(base[base.horizon == 336].mse_common_zscore_mean.iloc[0]),
        "H720": float(base[base.horizon == 720].mse_common_zscore_mean.iloc[0]),
        "L192_MSE": float(ag[(ag.input_length == 192) & (ag.horizon == 336)]
                          .mse_common_zscore_mean.iloc[0]),
        "L384_MSE": float(ag[(ag.input_length == 384) & (ag.horizon == 336)]
                          .mse_common_zscore_mean.iloc[0]),
        "minmax_MSE": float(ag[(ag.input_scaling == "minmax") & (ag.horizon == 336)]
                            .mse_common_zscore_mean.iloc[0]),
    }
    l96 = recomputed["H336"]
    recomputed["L192_pct"] = (recomputed["L192_MSE"] / l96 - 1) * 100
    recomputed["L384_pct"] = (recomputed["L384_MSE"] / l96 - 1) * 100
    recomputed["minmax_pct"] = (recomputed["minmax_MSE"] / l96 - 1) * 100
    for k, v in recomputed.items():
        print(f"  {k:12} = {v:.4f}" if "pct" not in k else f"  {k:12} = {v:+.2f}%")

    # 章节里的关键数字是否与之吻合
    # 注意：绝对值的容差是绝对误差，百分比的容差是"百分点"
    text = CHAPTER.read_text(encoding="utf-8") if CHAPTER.is_file() else ""
    abs_pairs = [
        ("0.1877", recomputed["H96"]), ("0.2504", recomputed["H192"]),
        ("0.3167", recomputed["H336"]), ("0.4166", recomputed["H720"]),
        ("0.3016", recomputed["L192_MSE"]), ("0.3146", recomputed["L384_MSE"]),
        ("0.3140", recomputed["minmax_MSE"]),
    ]
    pct_pairs = [
        ("−4.77%", recomputed["L192_pct"]),
        ("−0.65%", recomputed["L384_pct"]),
        ("−0.84%", recomputed["minmax_pct"]),
    ]
    bad = []
    for token, val in abs_pairs:
        if token not in text:
            bad.append(f"章节未出现 {token}")
        elif abs(float(token) - val) > 2e-4:
            bad.append(f"{token} 与重算 {val:.4f} 不符")
    for token, val in pct_pairs:
        if token not in text:
            bad.append(f"章节未出现 {token}")
            continue
        num = float(token.replace("−", "-").replace("%", ""))
        if abs(num - val) > 0.05:      # 容差 0.05 个百分点
            bad.append(f"{token} 与重算 {val:+.2f}% 不符")
    check(not bad, f"{len(abs_pairs) + len(pct_pairs)} 个关键数字与重算结果一致",
          "; ".join(bad[:3]) if bad else "")

    # ---------------- 汇总 ----------------
    ok_n = sum(1 for ok, _, _ in results if ok)
    print()
    print("=" * 74)
    print(f"审计结果：{ok_n} / {len(results)} 项通过")
    print("=" * 74)
    for ok, name, detail in results:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}"
        if detail:
            line += f"  —— {detail}"
        print(line)
    return 0 if ok_n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
