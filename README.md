# Qwen3.8-27B on BW100 — SGLang IFB TP2 性能优化

本仓库记录 Qwen3.8-27B 在海光 BW100 上的阶段性性能实验，分别保留官方 BF16 权重与官方 `Qwen3.8-27B-Channel-INT8-w8a8` 权重的结果。两条路线均为**单服务 IFB、TP2、两张 BW100**，没有采用 PD 分离，也没有自行量化。当前是可复核的**单轮筛选数据**，不是已经通过全部精度与正式评审的最终成绩。

## 优化配置

| 项目 | 当前候选 |
|---|---|
| 推理框架 | SGLang 0.5.12，DTK 26.04 环境 |
| 并行与设备 | IFB、TP2；实测使用物理 DCU2/3，启动前只检查所选两卡占用 |
| 注意力与 KV | FA3、HCU KV layout、`USE_CONTIGUOUS_FA=1`、BF16 KV cache |
| 推测解码 | EAGLE，`num_steps=3`、`topk=1`、`draft_tokens=4` |
| 其它 | 禁用 custom all-reduce；原版 GEMM；不启用 PDF 参考方案中的 NEXTN / FP8 KV / PD |
| 槽位 | BF16 统一候选 slots32；INT8 统一候选 slots64 |

FA3、HCU KV layout、连续 FA、EAGLE 和槽位设置是**组合优化**，下文的组合收益不能归因于其中某一项。曾试过的 custom all-reduce 因输出异常排除；NN GEMM 补丁因 64K 回退且尚未做最终精度验证，没有纳入当前候选。

## 固定长度：4K 输入 / 1K 输出

吞吐是实际输出 token/s；双卡合计除以 2 得单卡值。当前使用主办方澄清后的**均值**工作 SLA：TTFT `<5s` 且 TPOT `<50ms`。每档 2 轮预热、1 轮正式；正式请求数 `max(8, 2×并发)`，要求请求全部成功且输入/输出 token 数精确。按 C1/2/4/8/16/32 递增观察，C32 不满足后不再向上测试。本表只展示这六档；内部细扫记录仍在实验档案，不与官方离散档混排。

| 权重/配置 | 并发 | 双卡输出 tok/s | 单卡输出 tok/s | mean TTFT | mean TPOT | 工作 SLA | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| BF16 slots32 | C1 | 73.95 | 36.97 | 0.771s | 12.78ms | 通过，8/8 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c1/summary.json) |
| BF16 slots32 | C2 | 131.10 | 65.55 | 0.948s | 13.98ms | 通过，8/8 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c2/summary.json) |
| BF16 slots32 | C4 | 213.64 | 106.82 | 1.732s | 15.15ms | 通过，8/8 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c4/summary.json) |
| BF16 slots32 | C8 | 314.91 | 157.45 | 2.740s | 20.26ms | 通过，16/16 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c8/summary.json) |
| BF16 slots32 | C16 | 459.35 | **229.67** | 4.346s | 28.40ms | 通过，32/32 精确；六档内最高有效吞吐 | [`F026`](results/bf16/F026/fixed-4096-1024-c16/summary.json) |
| BF16 slots32 | C32 | 574.25 | 287.12 | 7.641s | 43.74ms | **未通过 TTFT**，64/64 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c32/summary.json) |
| INT8 slots64 | C1 | 98.93 | 49.46 | 0.593s | 9.54ms | 通过，8/8 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c1/summary.json) |
| INT8 slots64 | C2 | 179.98 | 89.99 | 0.728s | 10.33ms | 通过，8/8 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c2/summary.json) |
| INT8 slots64 | C4 | 302.35 | 151.18 | 1.446s | 11.30ms | 通过，8/8 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c4/summary.json) |
| INT8 slots64 | C8 | 367.61 | 183.81 | 2.080s | 18.18ms | 通过，16/16 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c8/summary.json) |
| INT8 slots64 | C16 | 555.91 | **277.96** | 3.184s | 23.65ms | 通过，32/32 精确；六档内最高有效吞吐 | [`I007`](results/int8/I007/fixed-4096-1024-c16/summary.json) |
| INT8 slots64 | C32 | 750.81 | 375.40 | 5.630s | 34.01ms | **未通过 TTFT**，64/64 精确 | [`I009`](results/int8/I009/fixed-4096-1024-c32/summary.json) |

两种权重在本表的官方离散档位均为 C1–C16 通过、C32 因 TTFT 未通过，故**六档内**的最高有效吞吐分别为 BF16 **229.67**、INT8 **277.96 tok/s/卡**。这不否定实验档案中对 C16–C32 之间的细扫，但本公开表按本次要求只放六档。BF16 官方同档双卡基线 C1/C8/C16/C32 分别为 21.195/182.734/304.034/433.847 tok/s；本方案 BF16 同档算术提升分别为 **+248.89%/+72.33%/+51.08%/+32.36%**，但 C32 未过 SLA，不能当作有效吞吐榜成绩。这不是主办方确认的最终得分。官方 INT8 权重的正式性能分母尚待明确；**不能把 INT8 与 BF16 的不同权重成绩当作同精度调参收益**。官方基线与评测规则见[基线帖](https://forum.sourcefind.cn/thread/d006a0e2-b624-11f1-93f4-0242ac150003)。

## 同 INT8 权重的优化前后

以下 A/B 使用**同一份官方 INT8 权重**、同卡、同槽位、同一轮正式请求。参考组已使用 FA3、BF16 KV 和禁用 custom all-reduce，并非未经设置的“开箱默认”或官方 INT8 基线。

| 场景 | 参考组双卡输出 | 优化组合双卡输出 | 组合提升 | 证据 |
|---|---:|---:|---:|---|
| 固定 4K/1K C16，slots32 | 367.79 tok/s | 555.62 tok/s | **+51.07%** | [`I005`](results/int8/I005/fixed-4096-1024-c16/summary.json) → [`I001`](results/int8/I001/fixed-4096-1024-c16/summary.json)，[审计](results/aggregate/int8-opt-vs-ref-fixed-20261015.json) |
| 公开 ShareGPT C64，slots64 | 538.10 tok/s | 797.94 tok/s | **+48.29%** | [审计](results/aggregate/int8-opt-vs-ref-sharegpt-20261015.json) |

INT8 slots64 在固定 4K C16 为 555.91 tok/s，较 slots32 的 555.62 基本持平；公开 ShareGPT C64 则从 slots32 的 603.49 提高到 slots64 的 797.94 tok/s（+32.22%）。[固定档审计](results/aggregate/int8-slots64-fixed-20261015.json) · [ShareGPT 审计](results/aggregate/int8-slots64-sharegpt-20261015.json)。公开 ShareGPT 文件身份尚未获主办方确认，**不能直接称为正式 ShareGPT 赛分**。

## 长输入：64K 输入 / 1K 输出

当前均值工作 SLA 为 TTFT `<20s` 且 TPOT `<60ms`。下表是**完整 65536/1024 token 请求**，不是短输出的容量 smoke。INT8 从已有 C1 继续测到 C2 通过、C4 首次未通过后停止；BF16 也在 C4 首次未通过后停止，均未继续 C8。

| 权重/配置 | 并发 | 双卡输出 tok/s | 单卡输出 tok/s | mean TTFT | mean TPOT | 工作 SLA | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| BF16 slots32 | C1 | 37.55 | 18.78 | 14.081s | 12.89ms | 通过，8/8 精确 | [`F023`](results/bf16/F023/fixed-65536-1024-c1/summary.json) |
| BF16 slots32 | C2 | 48.14 | 24.07 | 16.023s | 25.86ms | 通过，8/8 精确 | [`F029`](results/bf16/F029/fixed-65536-1024-c2/summary.json) |
| BF16 slots32 | C4 | 55.28 | 27.64 | 25.150s | 46.37ms | **未通过 TTFT**，8/8 精确 | [`F029`](results/bf16/F029/fixed-65536-1024-c4/summary.json) |
| INT8 slots64 | C1 | 44.68 | 22.34 | 11.268s | 11.39ms | 通过，8/8 精确 | [`I008`](results/int8/I008/fixed-65536-1024-c1/summary.json) |
| INT8 slots64 | C2 | 57.22 | 28.61 | 12.815s | 21.84ms | 通过，8/8 精确 | [`I013`](results/int8/I013/fixed-65536-1024-c2/summary.json) |
| INT8 slots64 | C4 | 66.37 | 33.19 | 20.111s | 38.76ms | **未通过 TTFT**，8/8 精确 | [`I014`](results/int8/I014/fixed-65536-1024-c4/summary.json) |

两种权重在本次单轮阶梯中均为 C1/C2 通过、C4 未通过，因此**最高已测通过档是 C2**，不等于多轮稳定上限。INT8 C4 的 TTFT 仅超门槛约 111ms，尤其需要正式复测后再定论。BF16 C1 对官方同档双卡基线 13.616 tok/s 的算术提升约 175.78%，但正式评分仍需对齐轮数与数据口径。INT8 C1 不拿不同权重的官方 BF16 基线直接当作已确认的 INT8 提升率。

## 满足 SLA 的最高单卡吞吐：相对官方基线

这里的“最高吞吐”定义为：**在已完成、满足当前均值 SLA 的并发档中，取单卡输出 tok/s 最大者**。赛题规范的两张基线表存在冲突：BW100/SGLang 4K/1K 的“官方最大并发 Baseline”填 **C1**，但固定档表中 C8 的 mean TTFT **3.816s**、mean TPOT **40.076ms**，均满足文档的 `<5s` / `<50ms`；C16 的 mean TTFT **6.801s** 则未通过。因此，若**按固定档均值重算**，4K 基线最大通过档应为 **C8**，双卡 **182.734 tok/s**，即 **91.367 tok/s/卡**。64K 固定档 C1 的 mean TTFT **19.266s**、mean TPOT **54.682ms** 满足 `<20s` / `<60ms`，C8 未通过，与公布的 C1 最大档一致；其双卡吞吐为 **13.616 tok/s**，即 **6.808 tok/s/卡**。下表优先展示按固定档均值重算的对比；它是**工作口径**，不能代替主办方公布的最大并发测试结论。公式为 `(候选单卡吞吐 ÷ 对应场景的基线单卡吞吐 − 1) × 100%`。双方都是 TP2，因此用双卡吞吐直接相除也得到同一比例。

| 场景与权重 | 官方 BF16 固定档基线：按均值筛选的最大通过档 / 单卡 tok/s | 本方案：已测 SLA 内最高吞吐档 / 单卡 tok/s | 相对该基线算术提升 | 证据 |
|---|---:|---:|---:|---|
| 4K/1K · BF16 slots32 | C8 / 91.3670 | C16 / **229.6728** | **+151.37%** | [`F026`](results/bf16/F026/fixed-4096-1024-c16/summary.json) |
| 4K/1K · INT8 slots64，仅公开六档 | C8 / 91.3670 | C16 / **277.9563** | **+204.22%（条件性）** | [`I007`](results/int8/I007/fixed-4096-1024-c16/summary.json) |
| 4K/1K · INT8 slots64，含额外细扫 | C8 / 91.3670 | C28 / **357.1228** | **+290.87%（条件性）** | [`I010`](results/peak-evidence/I010/fixed-4096-1024-c28/summary.json) |
| 64K/1K · BF16 slots32 | C1 / 6.8080 | C2 / **24.0678** | **+253.52%** | [`F029`](results/bf16/F029/fixed-65536-1024-c2/summary.json) |
| 64K/1K · INT8 slots64 | C1 / 6.8080 | C2 / **28.6110** | **+320.26%（条件性）** | [`I013`](results/int8/I013/fixed-65536-1024-c2/summary.json) |

这是**跨并发档的有效峰值对比**，不同于上文“同一并发档位”的性能提升。4K 的 BF16 细扫 C18 也曾单轮通过，但吞吐低于 C16，故未选为“最高吞吐”；INT8 C28 的 mean TTFT 为 4.966s，离 5s 门槛仅约 34ms，不能宣称稳定成绩。INT8 行使用**官方 BF16 固定档基线作条件性分母**，不是同权重调参收益；主办方尚未明确 INT8 正式性能基线，不能把条件性比例当作官方最终得分。若评审坚持直接采用赛题规范另表**公布**的 4K SLA 最大并发 C1，则对应分母是 10.5975 tok/s/卡，而非本表的 91.3670；两者不可混称。基线帖另有 p95 SLA 口径，与主办方后来澄清的赛题规范**均值**口径不同。正式评审如何解决两张表的矛盾，以及选档、轮数和提升率分母，仍待主办方确认。

## ShareGPT 标准数据集

赛题要求测 C1/C4/C16/C32/C64，记录吞吐、TTFT、TPOT 与成功率；**没有为 ShareGPT 单独给出数值 SLA**，不能直接套用随机 4K 的 5s/50ms。下面使用公开 ShareGPT 文件，正式请求每档 1000 条、预热 `2×并发`，目前每档只做 1 轮；官方基线教程写 3 轮，因此下表仍是筛选结果。公开文件是否与主办方最终指定文件完全一致，尚未确认。

| 权重/配置 | 并发 | 双卡输出 tok/s | 单卡输出 tok/s | mean TTFT | mean TPOT | 完成 | 原始运行 |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 slots32 | C1 | 61.09 | 30.54 | 240.78ms | 17.04ms | 1000/1000 | SG014 |
| BF16 slots32 | C4 | 200.27 | 100.14 | 243.18ms | 21.11ms | 1000/1000 | SG013 |
| BF16 slots32 | C16 | 420.52 | 210.26 | 285.19ms | 40.05ms | 1000/1000 | SG013 |
| BF16 slots32 | C32 | 552.63 | 276.31 | 461.22ms | 61.42ms | 1000/1000 | SG008 |
| BF16 slots32 | C64 | 552.00 | 276.00 | 11232.45ms | 61.00ms | 1000/1000 | SG011 |
| INT8 slots64 | C64 | **797.94** | **398.97** | 557.75ms | 90.59ms | 1000/1000 | I004 |

按此前约定，INT8 的 ShareGPT 只保留 C64；若要与 BF16 一样形成完整的五档矩阵，尚缺 INT8 C1/C4/C16/C32 的 1000 请求测试。BF16 C64 的吞吐与 C32 接近，但 TTFT 明显恶化；不可只看吞吐而忽略时延。INT8 C64 的 [同权重 A/B](results/aggregate/int8-opt-vs-ref-sharegpt-20261015.json) 和 [slots32/64 对照](results/aggregate/int8-slots64-sharegpt-20261015.json) 已在仓库中；BF16 C32/C64 的 [审计汇总](results/aggregate/sharegpt-slots32-20261014.json)与[槽位对照](results/aggregate/sharegpt-c64-slots-20261015.json)另附。

官方 BW100/SGLang BF16 ShareGPT 双卡基线 C1/C4/C16/C32/C64 分别为 21.454/106.207/307.450/435.996/528.158 tok/s。本表 BF16 同档算术提升依次为 **+184.72%/+88.57%/+36.78%/+26.75%/+4.51%**。现有 INT8 C64 的双卡 **797.9416 tok/s**，对同档官方 **BF16** 基线 **528.158 tok/s** 的算术提升为 **+51.08%**（单卡 398.9708 对 264.0790 tok/s，比例相同）；这是**跨权重、条件性的官方基线对照**，不同于上文相对同 INT8 权重参考组的 **+48.29%**。由于公开数据文件身份和轮数尚未与官方完全对齐，且官方 INT8 性能分母未明确，这些只是参考对照，不是正式榜单提升率。

## 尚未完成的验证

- BF16 的完整精度准入尚未在最终 slots32 配置上完成；既有 GSM8K 结果属于另一配置且严格协议存在截断，**不可移植为本配置的精度证明**。INT8 尚未做完整精度评测。最终性能方案确定后再分别验证两种权重。
- 固定长度和 ShareGPT 当前均为 1 轮正式筛选；提交前若要求 3 轮，须在最终配置上重跑，不把单轮统计冒充正式三轮成绩。
- 本仓库先公开脚本、逐档摘要和审计汇总。完整逐请求 JSONL、服务器日志、硬件快照仍保存在实验工作区，尚未纳入本次公开提交；摘要不能替代正式要求的原始日志。脚本输出 `summary.json` 内静态 `method` 文案仍写“三轮”，应以实际 `measure_rounds: 1`、逐轮记录和 runner 参数为准。

## 复现入口

本仓库的 `scripts/` 对应实验机 `/data/zk/qwen38/scripts/`；须先准备好预建容器、模型只读挂载、数据集与 SGLang/DTK 环境。以下命令仅供同类环境复现单轮筛选，**不要在未确认卡位、挂载和容器名时直接运行**。脚本默认卡位是历史值 `0,1`，因此务必显式指定空闲的 `DCU_DEVICES`；runner 启动前还会核对所选卡的显存与 HCU 占用。

```bash
# BF16：物理 DCU2/3、slots32
DCU_DEVICES=2,3 CONTAINER=zk-qwen38 \
  FIXED_CONCURRENCIES='1 2 4 8 16 32' FIXED_MEASURE_ROUNDS=1 \
  bash /data/zk/qwen38/scripts/run_fixed_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots32_no_custom_ar

# 官方 INT8 权重：物理 DCU2/3、slots64
DCU_DEVICES=2,3 CONTAINER=zk-qwen38-int8 \
  MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8 \
  SERVED_MODEL_NAME=qwen38-int8 FIXED_CONCURRENCIES='1 2 4 8 16 32' \
  FIXED_MEASURE_ROUNDS=1 \
  bash /data/zk/qwen38/scripts/run_fixed_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots64_no_custom_ar

# 长输入：复现已测阶梯；两种权重须分开运行，C4 未过后停止
DCU_DEVICES=2,3 CONTAINER=zk-qwen38 \
  FIXED_INPUT_LEN=65536 FIXED_SMOKE_64K=1 FIXED_CONCURRENCIES='1 2 4' \
  FIXED_MEASURE_ROUNDS=1 \
  bash /data/zk/qwen38/scripts/run_fixed_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots32_no_custom_ar
DCU_DEVICES=2,3 CONTAINER=zk-qwen38-int8 \
  MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8 \
  SERVED_MODEL_NAME=qwen38-int8 FIXED_INPUT_LEN=65536 FIXED_SMOKE_64K=1 \
  FIXED_CONCURRENCIES='1 2 4' FIXED_MEASURE_ROUNDS=1 \
  bash /data/zk/qwen38/scripts/run_fixed_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots64_no_custom_ar

# ShareGPT：须先核对公开文件哈希，用户当前仅要求 INT8 C64
DCU_DEVICES=2,3 CONTAINER=zk-qwen38 SHAREGPT_ROUNDS=1 \
  SHAREGPT_CONCURRENCIES='1 4 16 32 64' \
  bash /data/zk/qwen38/scripts/run_sharegpt_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots32_no_custom_ar
DCU_DEVICES=2,3 CONTAINER=zk-qwen38-int8 \
  MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8 \
  SERVED_MODEL_NAME=qwen38-int8 SHAREGPT_ROUNDS=1 SHAREGPT_CONCURRENCIES='64' \
  bash /data/zk/qwen38/scripts/run_sharegpt_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots64_no_custom_ar
```

每次换全新 run ID；runner 会在退出时停止本次启动的服务。`scripts/run_server.sh` 保留了历史实验变体，**当前有效配置只选上面列出的 no-custom-all-reduce 变体**。完整提交前还需确认官方数据版本、按最终要求补测轮数、精度与原始日志格式。

## 目录

- [`scripts/`](scripts/)：服务启动/停止、DCU 占用检查、固定长度与 ShareGPT benchmark、结果汇总脚本。
- [`results/bf16/`](results/bf16/) 和 [`results/int8/`](results/int8/)：本次挑选的逐档 `summary.json`。
- [`results/aggregate/`](results/aggregate/)：同权重 A/B、slots 比较和 ShareGPT 审计汇总。
- [`results/peak-evidence/`](results/peak-evidence/)：不混入官方六档主表的额外细扫峰值证据。

_更新：2026-10-16。后续逐步补齐正式多轮、精度、原始日志与提交材料。_
