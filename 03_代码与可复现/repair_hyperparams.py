"""修复已完成运行里 `hyperparameters.json` 的"种子污染"，并清掉下游产物以便重做。

问题
----
早期版本把与随机种子相关的字段（`fix_seed`、`model_id`、`run_id`）写进了
`hyperparameters.json`。成员 B 的 `summarize` 在同一
(model, input_length, horizon, input_scaling) 分组内比对
`hyperparameters_sha256`，于是三个种子被判为"配置不一致"：

    同一条件的随机种子运行混入不同配置／代码：(...); hyperparameters_sha256

修复做法
--------
1. 用 `run_protocol.hyperparameters_payload()` 重新生成**种子无关**的超参数文件；
2. 顺带用 `fill_metadata()` 刷新 `run_metadata.json`（补充 `checkpoint_model_id`）；
3. 删除一旦内容变化就失效的下游产物，好让整条链路按正确顺序重做：
       validation_score.json → locked_selection.json → pred.npy → true.npy
       → test.log → reports/08_reproduction/runs/<run_id>/
   **不删除**：train.log、checkpoint.pth、validation_pred/true/origin（这些没变）

之后用 run_batch 重做评分链路（不会重新训练）：

    python tools/repair_hyperparams.py --all-done
    python tools/run_batch.py --only TimesNet_L96_H96_zscore_s2021
    python tools/run_batch.py --only TimesNet_L96_H96_zscore_s2026

用法
----
    python tools/repair_hyperparams.py --all-done
    python tools/repair_hyperparams.py --run-id TimesNet_L96_H96_zscore_s2021
    python tools/repair_hyperparams.py --all-done --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_protocol as rp  # noqa: E402

# 评测输出目录（run_batch 里叫 RUNS_DIR，run_protocol 没有这个常量）
RUNS_DIR = rp.WORK / "reports/08_reproduction/runs"

# 内容一变就必须重做的下游产物
STALE_FILES = [
    "validation_score.json", "locked_selection.json",
    "pred.npy", "true.npy", "test.log",
]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def repair(run_id: str, dry: bool) -> dict:
    plan = rp.load_plan()
    row = rp.plan_row(run_id, plan)
    params = rp.build_params(row, num_workers=4, epochs=None)
    base = rp.PRED_DIR / run_id

    hp_path = base / "hyperparameters.json"
    old = hp_path.read_bytes() if hp_path.is_file() else b""
    new_payload = rp.hyperparameters_payload(params)
    new = (json.dumps(new_payload, ensure_ascii=False, indent=2, allow_nan=False)
           + "\n").encode("utf-8")

    changed = old != new
    removed = []
    if not dry:
        hp_path.write_text(new.decode("utf-8"), encoding="utf-8")
        # 刷新 metadata（补 checkpoint_model_id；种子等字段由 B 的 new-run 写入）
        setting = rp.find_setting_dir(params["model_id"])
        rp.fill_metadata(base, row, params, setting=setting)
        for name in STALE_FILES:
            p = base / name
            if p.is_file():
                p.unlink()
                removed.append(name)
        run_out = RUNS_DIR / run_id
        if run_out.is_dir():
            shutil.rmtree(run_out)
            removed.append(f"reports/08_reproduction/runs/{run_id}/")

    return {
        "run_id": run_id,
        "sha_before": sha_bytes(old)[:16] if old else "(无)",
        "sha_after": sha_bytes(new)[:16],
        "changed": changed,
        "removed": removed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", action="append", default=None)
    ap.add_argument("--all-done", action="store_true",
                    help="处理所有训练已完成的 A 运行")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.run_id and not args.all_done:
        raise SystemExit("需指定 --run-id 或 --all-done")

    plan = rp.load_plan()
    mine = plan.loc[plan.owner == "A"]

    if args.run_id:
        targets = args.run_id
    else:
        targets = []
        for rid in mine.run_id:
            base = rp.PRED_DIR / rid
            if (base / "train.log").is_file() and (base / "checkpoint.pth").is_file():
                targets.append(rid)

    print(f"处理 {len(targets)} 个运行" + ("（--dry-run，不改动）" if args.dry_run else ""))
    print()
    results = [repair(rid, args.dry_run) for rid in targets]

    print(f"{'run_id':<40}{'原哈希':>18}{'新哈希':>18}  变化")
    print("-" * 86)
    for r in results:
        print(f"{r['run_id']:<40}{r['sha_before']:>18}{r['sha_after']:>18}"
              f"  {'改' if r['changed'] else '一致'}")
    print()

    # 关键校验：同一 (L,H,scaling) 组内，新哈希必须完全一致
    print("=== 同组种子哈希一致性校验 ===")
    groups: dict[tuple, set] = {}
    for r in results:
        row = rp.plan_row(r["run_id"], plan)
        key = (str(row.model), int(row.input_length), int(row.horizon), str(row.input_scaling))
        groups.setdefault(key, set()).add(r["sha_after"])
    ok = True
    for key, shas in groups.items():
        good = len(shas) == 1
        ok = ok and good
        print(f"  {key}  ->  {len(shas)} 种哈希  {'OK' if good else 'FAIL'}")
    if len(results) > 1 and all(len(s) == 1 for s in groups.values()) and len(
            {next(iter(s)) for s in groups.values()}) == 1:
        print("  （所有运行共用一个哈希，说明超参数文件已完全与种子无关）")

    print()
    for r in results:
        if r["removed"]:
            print(f"{r['run_id']} 已清理下游产物：")
            for x in r["removed"]:
                print(f"    - {x}")

    print()
    if args.dry_run:
        print("（--dry-run，未做任何改动）")
        return 0 if ok else 1

    print("下一步重做评分链路（不会重新训练）：")
    for r in results:
        print(f"  python tools/run_batch.py --only {r['run_id']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
