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

吞吐是实际输出 token/s；双卡合计除以 2 得单卡值。当前使用主办方澄清后的**均值**工作 SLA：TTFT `<5s` 且 TPOT `<50ms`。每档 2 轮预热、1 轮正式；正式请求数 `max(8, 2×并发)`，要求请求全部成功且输入/输出 token 数精确。下表中的“最高”只指**已测单轮档位**，不代表长期稳定性或未测档位的全局最优。

| 权重/配置 | 并发 | 双卡输出 tok/s | 单卡输出 tok/s | mean TTFT | mean TPOT | 工作 SLA | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| BF16 slots32 | C16 | 459.35 | **229.67** | 4.346s | 28.40ms | 通过，32/32 精确；最高已测有效吞吐 | [`F026`](results/bf16/F026/fixed-4096-1024-c16/summary.json) |
| BF16 slots32 | C18 | 390.26 | 195.13 | 4.858s | 38.15ms | 通过，36/36 精确；最高已测通过并发 | [`F027`](results/bf16/F027/fixed-4096-1024-c18/summary.json) |
| BF16 slots32 | C19 | 470.71 | 235.36 | 5.021s | 32.61ms | 未通过 TTFT | [`F027`](results/bf16/F027/fixed-4096-1024-c19/summary.json) |
| INT8 slots64 | C27 | 699.21 | 349.60 | 4.823s | 31.58ms | 通过，54/54 精确；备选 | [`I011`](results/int8/I011/fixed-4096-1024-c27/summary.json) |
| INT8 slots64 | C28 | 714.25 | **357.12** | 4.966s | 31.91ms | 通过，56/56 精确；最高已测通过档，TTFT 仅余 34ms | [`I010`](results/int8/I010/fixed-4096-1024-c28/summary.json) |
| INT8 slots64 | C29 | 723.54 | 361.77 | 5.116s | 32.84ms | 未通过 TTFT | [`I011`](results/int8/I011/fixed-4096-1024-c29/summary.json) |

官方 BW100/SGLang BF16 基线的 4K/1K C16 双卡输出吞吐为 304.034 tok/s；本方案 BF16 同档 459.35 tok/s，算术提升 **51.08%**。这是相同并发档的对照，不是主办方确认的最终得分。官方 INT8 权重的正式性能分母尚待明确；**不能把 INT8 与 BF16 的不同权重成绩当作同精度调参收益**。官方基线与评测规则见[基线帖](https://forum.sourcefind.cn/thread/d006a0e2-b624-11f1-93f4-0242ac150003)。

## 同 INT8 权重的优化前后

以下 A/B 使用**同一份官方 INT8 权重**、同卡、同槽位、同一轮正式请求。参考组已使用 FA3、BF16 KV 和禁用 custom all-reduce，并非未经设置的“开箱默认”或官方 INT8 基线。

| 场景 | 参考组双卡输出 | 优化组合双卡输出 | 组合提升 | 证据 |
|---|---:|---:|---:|---|
| 固定 4K/1K C16，slots32 | 367.79 tok/s | 555.62 tok/s | **+51.07%** | [`I005`](results/int8/I005/fixed-4096-1024-c16/summary.json) → [`I001`](results/int8/I001/fixed-4096-1024-c16/summary.json)，[审计](results/aggregate/int8-opt-vs-ref-fixed-20261015.json) |
| 公开 ShareGPT C64，slots64 | 538.10 tok/s | 797.94 tok/s | **+48.29%** | [审计](results/aggregate/int8-opt-vs-ref-sharegpt-20261015.json) |

INT8 slots64 在固定 4K C16 为 555.91 tok/s，较 slots32 的 555.62 基本持平；公开 ShareGPT C64 则从 slots32 的 603.49 提高到 slots64 的 797.94 tok/s（+32.22%）。[固定档审计](results/aggregate/int8-slots64-fixed-20261015.json) · [ShareGPT 审计](results/aggregate/int8-slots64-sharegpt-20261015.json)。公开 ShareGPT 文件身份尚未获主办方确认，**不能直接称为正式 ShareGPT 赛分**。

## 长输入与测试边界

- BF16 slots32：完整 64K/1K C1 为 37.55 tok/s、C2 为 48.14 tok/s，均通过当前均值 SLA（TTFT `<20s`、TPOT `<60ms`）；C4 的 TTFT 25.15s 未通过。[C1](results/bf16/F023/fixed-65536-1024-c1/summary.json) · [C2](results/bf16/F029/fixed-65536-1024-c2/summary.json) · [C4](results/bf16/F029/fixed-65536-1024-c4/summary.json)。按实验计划，不再向上补测 64K。
- INT8 slots64：完整 64K/1K C1 为 44.68 tok/s，8/8 精确，通过当前均值 SLA。[I008](results/int8/I008/fixed-65536-1024-c1/summary.json)。更高并发未测。
- BF16 的完整精度准入尚未在最终 slots32 配置上完成；既有 GSM8K 结果属于另一配置且严格协议存在截断，**不可移植为本配置的精度证明**。INT8 尚未做完整精度评测。最终性能方案确定后再分别验证两种权重。
- 本仓库先公开脚本、逐档摘要和审计汇总。完整逐请求 JSONL、服务器日志、硬件快照仍保存在实验工作区，尚未纳入本次公开提交；摘要不能替代正式要求的原始日志。脚本输出 `summary.json` 内静态 `method` 文案仍写“三轮”，应以实际 `measure_rounds: 1`、逐轮记录和 runner 参数为准。

## 复现入口

本仓库的 `scripts/` 对应实验机 `/data/zk/qwen38/scripts/`；须先准备好预建容器、模型只读挂载、数据集与 SGLang/DTK 环境。以下命令仅供同类环境复现单轮筛选，**不要在未确认卡位、挂载和容器名时直接运行**。脚本默认卡位是历史值 `0,1`，因此务必显式指定空闲的 `DCU_DEVICES`；runner 启动前还会核对所选卡的显存与 HCU 占用。

```bash
# BF16：物理 DCU2/3、slots32
DCU_DEVICES=2,3 CONTAINER=zk-qwen38 \
  FIXED_CONCURRENCIES='1 8 16 32' FIXED_MEASURE_ROUNDS=1 \
  bash /data/zk/qwen38/scripts/run_fixed_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots32_no_custom_ar

# 官方 INT8 权重：物理 DCU2/3、slots64
DCU_DEVICES=2,3 CONTAINER=zk-qwen38-int8 \
  MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8 \
  SERVED_MODEL_NAME=qwen38-int8 FIXED_CONCURRENCIES='16 27 28 29' \
  FIXED_MEASURE_ROUNDS=1 \
  bash /data/zk/qwen38/scripts/run_fixed_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots64_no_custom_ar
```

每次换全新 run ID；runner 会在退出时停止本次启动的服务。`scripts/run_server.sh` 保留了历史实验变体，**当前有效配置只选上面列出的 no-custom-all-reduce 变体**。完整提交前还需确认官方数据版本、按最终要求补测轮数、精度与原始日志格式。

## 目录

- [`scripts/`](scripts/)：服务启动/停止、DCU 占用检查、固定长度与 ShareGPT benchmark、结果汇总脚本。
- [`results/bf16/`](results/bf16/) 和 [`results/int8/`](results/int8/)：本次挑选的逐档 `summary.json`。
- [`results/aggregate/`](results/aggregate/)：同权重 A/B、slots 比较和 ShareGPT 审计汇总。

_更新：2026-10-16。后续逐步补齐正式多轮、精度、原始日志与提交材料。_
