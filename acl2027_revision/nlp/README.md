# 在 NLP 面板上补跑四项修订实验

本目录在 6 个 NLP 面板（RewardBench 2、WMT23 13/12 流、WMT24 en-de/en-es/ja-zh）上运行审稿回复所需的四项实验：

| 命令 | 回应 | 内容 | 写入附录 |
|---|---|---|---|
| `maxcap` | Q4 | 平均目标 vs. 逐替换上限：全信息、零查询、α/H 归约，以及 reference priority 主停止点处平均证书违反单替换上限的比例 | R.2 |
| `weights` | Q1 | 均匀 / 按参考相关性加权（部署可能性先验）/ Dirichlet(1) 权重下的联合、并集、相加最小证书（均值池化，精确与主停止） | R.1 |
| `screen` | Q7 | 960 个均值实例上插件预筛查 V̂*+ε<\|S0\| 对联合增益的召回率/精确率 | U |
| `solvers` | W3 | 精确停止联合规划：HiGHS（presolve 开/关、两种种子）vs. CBC | W |

## 第一步：准备面板（两种方式任选其一）

**方式 A（推荐，可精确复现原文）：** 用你原来管线缓存的面板，导出为 `panels/<name>.npz`，名字为
`rewardbench2, wmt23_13, wmt23_12, wmt24_ende, wmt24_enes, wmt24_jazh`。每个文件包含：

- `raw`：(输出数, 评估器数) 原始分数，列顺序 = 冻结评估器顺序（前 7 列为基础池，其余为候选）；
- `group`：每个输出所属的 prompt / 源段落 / 文档编号（整数）；
- `group_order`：分组编号按原协议的排列顺序（RewardBench：prompt ID 数值排序；WMT23：段落顺序；WMT24：文档首次出现顺序）。

**方式 B：** 从公开数据重建（需要网络能访问 huggingface.co 和 data.statmt.org）：

```bash
python3 build_panels.py rewardbench --out panels
wget https://data.statmt.org/wmt26/mt-metrics-eval-v2.tgz && tar xzf mt-metrics-eval-v2.tgz
python3 build_panels.py wmt --mtme mt-metrics-eval-v2 --out panels
```

脚本会打印每个模型匹配到的结果文件、prompt/段落/系统/指标数量，请对照原文（1,723 prompts；557 段落 × 13 流 × 24 指标；WMT24 171/171/197 文档）。WMT24 的评估器顺序默认按名称字母序；若原文使用 `supplement/wmt24_evaluator_manifest.csv` 中的顺序，请用该顺序重排 `raw` 的列。

## 第二步：复现校验（必须先跑）

```bash
python3 run_nlp.py validate --panels panels
python3 run_nlp.py report
```

报告会把 q=.05、α=.10 下的零查询覆盖、全信息覆盖、reference priority 检查比例和 Q_exact 与原文表 2、表 8、附录 M.2 的数值并列显示。一致后再跑实验。实现细节：ECDF 和共识比较全部使用整数分子（精确算术），并列顺序使用原文的 `default_rng(split_seed+700001)` 排列；WMT24 原文的 10⁻¹² 定点优先级未实现，查询顺序在并列时可能略有差异。

## 第三步：四项实验（可按面板并行）

```bash
for p in rewardbench2 wmt23_13 wmt23_12 wmt24_ende wmt24_enes wmt24_jazh; do
  python3 run_nlp.py maxcap  --panels panels --only $p &           # 1,920 配置
done; wait
python3 run_nlp.py screen  --panels panels                         # 960 均值实例
python3 run_nlp.py weights --panels panels --splits 0-7            # 6×4×8×3 方案，每目标 60 s 限时
python3 run_nlp.py solvers --panels panels --splits 0-7            # 192 个精确目标
```

结果按行写入 `../results/nlp/*.jsonl`，可中断续跑。`screen` 的结果可以直接与原文附录 M.9 对照：57 个非全保留实例、57 个精确目标和 32 个主停止目标有严格增益。`maxcap` 中 reference priority 主停止的违反比例可以与原文附录 I 对照：中位数 308/960，均值 541/960。
