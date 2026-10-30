"""把 TimesNet 的第三个随机种子 2028 换成 2021（组内已同意）。

为什么必须改代码而不只是改 CSV
--------------------------------
成员 B 的 `scripts/member_b_next_common.py::experiment_plan()` 每次调用都会
**重新生成**运行计划；`08_reproduction_comparison.py` 的 `init` 和 `summarize`
都用这个函数而不是读现成的 CSV。只改 `run_plan.csv` 会导致
`run_status.csv` 与实际不一致，而且 `new_run` / `load_meta` 的 run_id 查表
仍然按 CSV 走，两边会错位。

所以本工具同时做两件事：
  1. 改 `work/scripts/member_b_next_common.py` 里三处硬编码的种子列表；
  2. 用改后的 `experiment_plan()` 重新生成 `work/artifacts/evaluation/v1/run_plan.csv`。

范围
----
只改 **TimesNet**（成员 A）。C 的 XGBoost / Informer 仍是 {2026, 2027, 2028}，
因为只有 TimesNet 需要与论文 Table 13 对标，2021 正是官方 `run.py` 里写死的种子。

不改动
------
`evaluation_protocol.json` 只锁 data_protocol / scaler_parameters /
detection_rules / hypotheses 四个哈希（见 `contract()`），不锁脚本也不锁
run_plan，所以不重跑 `init` 也不会被判为不一致。

产出
----
- `work/scripts/member_b_next_common.py`（改后）
- `work/artifacts/evaluation/v1/run_plan.csv`（重新生成）
- `tools/seed_2021.diff`（给成员 B 的补丁，便于他那边字节一致地跟进）
- `tools/seed_2021_sha256.txt`（改后文件的 SHA-256，供双方核对）

用法（必须用装了 matplotlib/pandas 的环境）：
    python tools/apply_seed_2021.py
"""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # memberA_timesnet/
WORK = ROOT / "work"
HELPER = WORK / "scripts" / "member_b_next_common.py"
PLAN_CSV = WORK / "artifacts/evaluation/v1/run_plan.csv"
# 基线必须取成员 B 的**原始交接包**，而不是已经改过的 work/ 副本，
# 否则第二次运行时 diff 会变成空文件（无法作为给 B 的补丁）。
PRISTINE = (
    ROOT.parent
    / "DSC5002_MemberB_Phase1_Handover_v1"
    / "DSC5002_MemberB_Phase1_Handover_v1"
    / "scripts"
    / "member_b_next_common.py"
)
OUT_DIFF = HERE / "seed_2021.diff"
OUT_SHA = HERE / "seed_2021_sha256.txt"

NEW_SEEDS = "[2021,2026,2027]"

# (原字符串, 新字符串, 说明)
EDITS: list[tuple[str, str, str]] = [
    (
        "        seeds=[None] if model=='ARIMA' else [2026,2027,2028]",
        "        # [DSC5002-A SEED-2021] TimesNet 换用 2021（官方 run.py 写死的种子），"
        "与论文同条件\n"
        "        seeds=[None] if model=='ARIMA' else ([2021,2026,2027] if model=='TimesNet' "
        "else [2026,2027,2028])",
        "第125行：按模型区分种子（ARIMA 确定性 / TimesNet 2021,2026,2027 / 其余不变）",
    ),
    (
        "    for seed in [2026,2027,2028]: rows.append({'model':'TimesNet','seed':seed,"
        "'input_length':length,'horizon':336,'input_scaling':'zscore',"
        "'experiment':'input_length','owner':'A'})",
        "    for seed in [2021,2026,2027]: rows.append({'model':'TimesNet','seed':seed,"
        "'input_length':length,'horizon':336,'input_scaling':'zscore',"
        "'experiment':'input_length','owner':'A'})",
        "第131行：输入长度消融（L192/384 @H336）改用 2021,2026,2027",
    ),
    (
        "    for seed in [2026,2027,2028]: rows.append({'model':'TimesNet','seed':seed,"
        "'input_length':96,'horizon':336,'input_scaling':'minmax',"
        "'experiment':'scaling','owner':'A'})",
        "    for seed in [2021,2026,2027]: rows.append({'model':'TimesNet','seed':seed,"
        "'input_length':96,'horizon':336,'input_scaling':'minmax',"
        "'experiment':'scaling','owner':'A'})",
        "第132行：缩放消融（Min-Max @L96/H336）改用 2021,2026,2027",
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not HELPER.is_file():
        print(f"FAIL: 找不到 {HELPER}")
        return 1

    if not PRISTINE.is_file():
        print(f"FAIL: 找不到原始交接包里的基线文件 {PRISTINE}")
        return 1

    original = PRISTINE.read_text(encoding="utf-8")
    text = original

    print(f"基线（B 原始）: {PRISTINE}")
    print(f"目标（A 副本）: {HELPER}")
    print(f"基线 SHA256   : {sha256_from_text(original)}\n")

    for old, new, desc in EDITS:
        n = text.count(old)
        if n != 1:
            print(f"FAIL: {desc}\n      在基线文件中匹配到 {n} 次（应为 1 次），已中止。")
            return 1
        text = text.replace(old, new, 1)
        print(f"  OK  {desc}")

    if text == original:
        print("FAIL: 基线文件与目标内容相同，补丁没有产生任何变化。")
        return 1

    already = HELPER.is_file() and HELPER.read_text(encoding="utf-8") == text
    HELPER.parent.mkdir(parents=True, exist_ok=True)
    HELPER.write_text(text, encoding="utf-8")
    new_sha = sha256(HELPER)
    print(f"\n新 SHA256: {new_sha}" + ("（文件本来就是这个内容）" if already else ""))

    # 给成员 B 的补丁：基线 = B 的原始文件，恒非空
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile="a/scripts/member_b_next_common.py",
            tofile="b/scripts/member_b_next_common.py",
        )
    )
    OUT_DIFF.write_text(diff, encoding="utf-8")
    OUT_SHA.write_text(
        f"file: scripts/member_b_next_common.py\n"
        f"sha256_before: {sha256_from_text(original)}\n"
        f"sha256_after: {new_sha}\n"
        f"change: TimesNet seeds [2026,2027,2028] -> [2021,2026,2027] "
        f"(XGBoost/Informer unchanged)\n",
        encoding="utf-8",
    )

    # 用改后的代码重新生成 run_plan.csv
    sys.path.insert(0, str(WORK / "scripts"))
    for mod in ("member_b_next_common", "eda_common"):
        sys.modules.pop(mod, None)
    import member_b_next_common as mb  # noqa: E402

    plan = mb.experiment_plan()
    # 用 B 自己的 write_csv，保证 float_format 与换行符（\n）与他的产物字节一致
    mb.write_csv(PLAN_CSV, plan)
    print(f"\n已重新生成: {PLAN_CSV}（{len(plan)} 行）")

    # ---------------- 校验 ----------------
    print("\n=== 校验 ===")
    a = plan[plan.owner == "A"]
    c = plan[plan.owner == "C"]
    print(f"计划总数           : {len(plan)}（应为 49）")
    print(f"A 的运行数         : {len(a)}（应为 21）")
    print(f"A 的种子分布       : {dict(a.seed.value_counts().sort_index())}")
    # C 的 seed 列混有字符串 'deterministic'（ARIMA），不能直接 sort_index
    csort = c.seed.astype(str).value_counts().sort_index()
    print(f"C 的种子分布       : {dict(csort)}（应仍含 2028）")

    got_2021 = sorted(a[a.seed == 2021].run_id)
    print(f"\nA 里 seed=2021 的 {len(got_2021)} 个 run_id：")
    for r in got_2021:
        print("   ", r)

    ok = (
        len(plan) == 49
        and len(a) == 21
        and set(a.seed.astype(str)) == {"2021", "2026", "2027"}
        and "2028" in set(c.seed.astype(str))
        and len(got_2021) == 7
    )
    print()
    print("=" * 64)
    print("结论：" + ("计划已正确改为 TimesNet {2021,2026,2027}，C 不受影响 ✅"
                    if ok else "校验未通过 ❌"))
    print("=" * 64)
    print(f"\n给成员 B 的补丁   : {OUT_DIFF.relative_to(ROOT)}")
    print(f"双方核对的 SHA256 : {OUT_SHA.relative_to(ROOT)}")
    return 0 if ok else 1


def sha256_from_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
