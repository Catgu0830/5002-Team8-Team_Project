# 完整数据（压缩后 4.29 GB）的获取方式

## 为什么不在仓库里

GitHub 对**普通仓库提交**有单文件 **100 MB** 硬上限。本项目有 60 个文件超限：

| 文件 | 单个大小 |
|---|---|
| `validation_pred.npy` / `validation_true.npy` / `pred.npy` / `true.npy`（H=720 三次运行） | **208 MB** 各 |
| 同四类文件（H=336 / L=192 / L=384 / minmax 各组） | 约 **100 MB** 各 |

Git LFS 免费额度只有 1 GB 存储 + 1 GB/月流量，也装不下。
因此张量与权重**不进仓库**，改放 **GitHub Release 附件**（单文件上限 2 GB），
仓库本身仍然约 130 MB。

## 怎么拿

打开本仓库的 **Releases** 页面，下载这 5 个分卷：

| 分卷 | 大小 | 包含的运行 |
|---|---:|---|
| `memberA_evidence_part1.zip` | 977 MB | H720 s2021, H720 s2026, H96 s2021 |
| `memberA_evidence_part2.zip` | 940 MB | H720 s2027, L384 s2021, L384 s2027, H96 s2027 |
| `memberA_evidence_part3.zip` | 899 MB | L192 s2021/s2026/s2027, L384 s2026, H96 s2026 |
| `memberA_evidence_part4.zip` | 819 MB | minmax s2021, minmax s2026, H336_zscore s2021/s2026 |
| `memberA_evidence_part5.zip` | 759 MB | minmax s2027, H336_zscore s2027, H192 s2021/s2026/s2027 |

（`s` 后缀为随机种子；H 为预测长度、L 为输入长度。）

各分卷的 **SHA-256 校验值**记录在 Release 说明与本仓库
`03_代码与可复现/` 对应的打包脚本输出中；如需重新生成，运行：

```bash
python 03_代码与可复现/make_release_zips.py --cap-mb 1800
```

解压后结构（每个 zip 独立，可单独解压使用）：

```
predictions/
├── TimesNet_L96_H96_zscore_s2021/
│   ├── run_metadata.json      hyperparameters.json    environment.txt
│   ├── command.txt            train.log              test.log
│   ├── patches.diff           checkpoint.pth
│   ├── validation_pred.npy    validation_true.npy    validation_origin_rows.npy
│   ├── pred.npy               true.npy               origin_rows.npy
│   ├── validation_score.json  locked_selection.json
└── …（全部 21 个运行目录）
```

## 拿到之后怎么用（独立复算）

把解压出的 `predictions/` 整个放进成员 B 的 `work/experiments/` 下（覆盖同名目录），
即可用 B 的统一评分工具重新执行：

```bash
# 重新验证验证集分数（只读验证集）
python work/scripts/08_reproduction_comparison.py score-validation \
    --run-id TimesNet_L96_H336_zscore_s2021 --project-root work

# 重新评分（需要 locked_selection.json，已随包提供）
python work/scripts/08_reproduction_comparison.py evaluate \
    --run-id TimesNet_L96_H336_zscore_s2021 --project-root work
```

仓库 `03_代码与可复现/verify_rescorable.py` 另提供一个**不依赖 B 的工具**、
只用张量重算 MSE/MAE 的参考实现。该脚本已在本机验证：
**21 次运行的最大相对偏差 2.38e-08**（float32 精度极限），
即「仅凭本包张量就能完整复算出报告中的指标」。

## 只想要小体积版本？

仓库里的 `01_预测与指标/` 已含全部**指标表、元数据、超参数、训练日志、封存记录
与误差分解 CSV**，只是不含张量与权重。
**如果只需要核对指标和图表，不需要下载 Release。**
