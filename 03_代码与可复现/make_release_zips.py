"""把全量证据（预测张量 + 权重）分卷打包成 GitHub Release 附件。

为什么需要分卷
--------------
GitHub 对**普通仓库提交**有单文件 100 MB 硬上限（本项目 60 个 .npy 超限，
H=720 的单个张量 208 MB），所以张量不能进仓库；
但 **Release 附件**的上限是单个 2 GB，因此把 8.27 GB 拆成若干 ≤1.8 GB 的 zip，
作为 Release 资产上传即可，仓库本身仍然只有一百多 MB。

分卷方式：按运行目录整目录打包（每个 zip 内部保留 `predictions/<run_id>/` 结构），
用贪心装箱保证每个包不超过 `CAP_MB`。这样任何一个包下载完都能独立解压使用。

产出
----
- `A_交付包_release/memberA_evidence_part{N}.zip`
- `A_交付包_release/RELEASE_README.txt`（含每个包的内容清单与 SHA-256）

用法：
    python tools/make_release_zips.py
    python tools/make_release_zips.py --cap-mb 1700 --out 某个目录
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT.parent / "A_交付包" / "01_预测与指标" / "predictions"
DEFAULT_OUT = ROOT.parent / "A_交付包_release"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-mb", type=int, default=1800,
                    help="每个分卷的大小上限（MB），须 < 2048")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-compress", action="store_true",
                    help="不压缩（更快，体积大一些）")
    args = ap.parse_args()

    if args.cap_mb >= 2048:
        raise SystemExit("cap 必须小于 2048 MB（GitHub Release 单文件上限）")
    if not SRC.is_dir():
        raise SystemExit(f"找不到源目录：{SRC}")

    runs = sorted([d for d in SRC.iterdir() if d.is_dir()],
                  key=lambda d: dir_size(d), reverse=True)
    total = sum(dir_size(d) for d in runs)
    print(f"源: {SRC}")
    print(f"运行目录 {len(runs)} 个，合计 {total / 1024**3:.2f} GB")
    print(f"分卷上限 {args.cap_mb} MB\n")

    # 贪心装箱（大件优先）
    cap = args.cap_mb * 1024 * 1024
    bins: list[list[Path]] = []
    loads: list[int] = []
    for d in runs:
        s = dir_size(d)
        for i, load in enumerate(loads):
            if load + s <= cap:
                bins[i].append(d)
                loads[i] += s
                break
        else:
            bins.append([d])
            loads.append(s)

    print(f"分卷结果：{len(bins)} 个包")
    for i, (b, load) in enumerate(zip(bins, loads), 1):
        print(f"  part{i}: {load / 1024**2:7.0f} MB  {len(b)} 个运行")
        for d in b:
            print(f"      - {d.name}  ({dir_size(d) / 1024**2:.0f} MB)")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    mode = zipfile.ZIP_STORED if args.no_compress else zipfile.ZIP_DEFLATED
    lvl = None if args.no_compress else 1
    manifest: list[tuple[str, int, str, list[str]]] = []

    for i, b in enumerate(bins, 1):
        name = f"memberA_evidence_part{i}.zip"
        out = args.out / name
        t0 = time.time()
        print(f"\n打包 {name} …")
        with zipfile.ZipFile(out, "w", compression=mode, compresslevel=lvl,
                             allowZip64=True) as zf:
            for d in b:
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        arc = Path("predictions") / d.name / f.relative_to(d)
                        zf.write(f, arc.as_posix())
        size = out.stat().st_size
        dt = time.time() - t0
        print(f"  {size / 1024**2:.0f} MB  用时 {dt:.0f}s")
        if size >= 2 * 1024**3:
            raise SystemExit(f"FAIL: {name} 超过 2 GB，请调小 --cap-mb")
        # 完整性自检
        with zipfile.ZipFile(out) as zf:
            bad = zf.testzip()
        if bad:
            raise SystemExit(f"FAIL: {name} 内含损坏文件 {bad}")
        print(f"  完整性自检通过")
        manifest.append((name, size, sha256_file(out), [d.name for d in b]))

    # 说明文件
    lines = [
        "成员 A 全量证据包（GitHub Release 附件）",
        "=" * 60,
        "",
        "内容：21 次 TimesNet/ETTm2 运行的完整证据目录，",
        "      含预测张量 (.npy) 与模型权重 (.pth)。",
        "      仓库本身不含这些文件（GitHub 单文件上限 100 MB）。",
        "",
        f"分卷数：{len(manifest)}    合计：{sum(m[1] for m in manifest) / 1024**3:.2f} GB",
        "",
        "使用方法：",
        "  1. 下载全部分卷，放在同一目录；",
        "  2. 逐个解压（每个 zip 独立，内部结构为 predictions/<run_id>/）；",
        "  3. 把 predictions/ 整体放入你的 work/experiments/ 下，即可用",
        "     work/scripts/08_reproduction_comparison.py 重新执行",
        "     score-validation / evaluate 进行独立复算。",
        "",
        "下载后建议校验 SHA-256：",
        "",
        f"{'文件名':<34}{'大小(MB)':>10}  SHA-256",
        "-" * 110,
    ]
    for name, size, h, runs_in in manifest:
        lines.append(f"{name:<34}{size / 1024**2:>10.0f}  {h}")
        lines.append(f"    包含: {', '.join(runs_in)}")
    manifest_path = args.out / "RELEASE_README.txt"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 70)
    print(f"完成：{len(manifest)} 个分卷，"
          f"合计 {sum(m[1] for m in manifest) / 1024**3:.2f} GB")
    print(f"输出目录：{args.out}")
    print(f"说明文件：{manifest_path.name}")
    print("=" * 70)
    for name, size, h, _ in manifest:
        print(f"  {name:<30}{size / 1024**2:>8.0f} MB   {h[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
