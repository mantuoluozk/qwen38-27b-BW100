# Qwen3.8-27B on BW100 — SGLang IFB TP2 性能优化

本仓库记录 Qwen3.8-27B 在海光 BW100 上的性能实验，分别保留官方 BF16 权重与官方 `Qwen3.8-27B-Channel-INT8-w8a8` 权重的结果。两条路线均为**单服务 IFB、TP2、两张 BW100**，没有采用 PD 分离，也没有自行量化。数据包含**固定档单轮与关键边界三轮复测**，分别标注；全量 GSM8K 已完成。2026-10-21 官方补充通知已明确只看 GSM8K，不再要求其他三个精度集；最终准入与得分仍由评审确认。

**最新：[2026-10-21 补充材料与完整报告索引](submissions/20261021/README.md)**。按官方补充要求，已补齐两种精度固定4K C64/C128、64K C8/C16/C32，并补4个边界档；新增1262个正式请求、1078个预热请求全部成功且长度精确。两种精度各有固定4K六个官方档、64K四档、ShareGPT五档、GSM8K及SLA边界数据；[BF16报告](docs/supplement-20261021/bf16.md)与[INT8报告](docs/supplement-20261021/int8.md)逐项列出TPS、TTFT、TPOT官方均值及对比比例。

**[原BF16 / INT8独立提交包](submissions/20261016/README.md)**保持不变，包含旧固定档、ShareGPT、GSM8K原始证据和复现依赖。新增测量明确发生于10月21日，不冒充截止前结果；补充原件另包保存，不覆盖原日志和哈希。

## 2026-10-21 SLA边界摘要

| 方案/场景 | 合并均值通过的最高三轮已测并发 | 单卡输出tok/s | mean TTFT / TPOT | 逐轮通过 | 三轮全过最高已测并发 |
|---|---:|---:|---|---|---:|
| BF16 4K/1K | C19 | 238.4864 | 4.9863s / 32.0413ms | **1/3** | C18 |
| BF16 64K/1K | C2 | 23.5553 | 16.0311s / 26.3923ms | 3/3 | C2 |
| INT8 4K/1K | C28 | 355.0269 | 4.9825s / 31.8910ms | **2/3** | C27 |
| INT8 64K/1K | C3 | 30.7849 | 15.7707s / 31.6928ms | **3/3** | C3 |

合并均值通过不等于每轮通过。BF16 C18三轮全过但仅199.1197 tok/s/卡，低于原C16的227.4383；C19均值边界吞吐较高但波动明显，原稳妥C16参考保留。INT8 4K稳妥参考仍为C27（348.8855 tok/s/卡）。INT8 64K新C3相对官方SLA内C1单卡6.808的算术提升为**+352.19%**。BF16 64K C3合并TTFT20.9278s未过，完整保留失败轮次。最大并发与最大有效吞吐分开报告；仅限已测集合，不宣称全局数学单调性。

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

吞吐是实际输出 token/s；双卡合计除以 2 得单卡值。当前使用主办方澄清后的**均值**工作 SLA：TTFT `<5s` 且 TPOT `<50ms`。每档 2 轮预热、1 轮正式；正式请求数 `max(8, 2×并发)`，要求请求全部成功且输入/输出 token 数精确。10月16日前按C1/2/4/8/16/32筛选；10月21日按官方补充要求继续完成C64/C128，不因SLA失败停止。内部细扫及三轮边界另表列出，不混改为本单轮曲线。

| 权重/配置 | 并发 | 双卡输出 tok/s | 单卡输出 tok/s | mean TTFT | mean TPOT | 工作 SLA | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| BF16 slots32 | C1 | 73.95 | 36.97 | 0.771s | 12.78ms | 通过，8/8 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c1/summary.json) |
| BF16 slots32 | C2 | 131.10 | 65.55 | 0.948s | 13.98ms | 通过，8/8 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c2/summary.json) |
| BF16 slots32 | C4 | 213.64 | 106.82 | 1.732s | 15.15ms | 通过，8/8 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c4/summary.json) |
| BF16 slots32 | C8 | 314.91 | 157.45 | 2.740s | 20.25ms | 通过，16/16 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c8/summary.json) |
| BF16 slots32 | C16 | 459.35 | **229.67** | 4.346s | 28.40ms | 通过，32/32 精确；六档内最高有效吞吐 | [`F026`](results/bf16/F026/fixed-4096-1024-c16/summary.json) |
| BF16 slots32 | C32 | 574.25 | 287.12 | 7.641s | 43.74ms | **未通过 TTFT**，64/64 精确 | [`F026`](results/bf16/F026/fixed-4096-1024-c32/summary.json) |
| BF16 slots32 | C64 | 588.84 | 294.42 | 46.711s | 46.96ms | **未通过 TTFT**，128/128精确 | [10-21补充](submissions/20261021/result_bf16.json) |
| BF16 slots32 | C128 | 606.21 | 303.11 | 125.533s | 48.57ms | **未通过 TTFT**，256/256精确 | [10-21补充](submissions/20261021/result_bf16.json) |
| INT8 slots64 | C1 | 98.93 | 49.46 | 0.593s | 9.54ms | 通过，8/8 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c1/summary.json) |
| INT8 slots64 | C2 | 179.98 | 89.99 | 0.728s | 10.33ms | 通过，8/8 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c2/summary.json) |
| INT8 slots64 | C4 | 302.35 | 151.18 | 1.446s | 11.30ms | 通过，8/8 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c4/summary.json) |
| INT8 slots64 | C8 | 367.61 | 183.81 | 2.080s | 18.18ms | 通过，16/16 精确 | [`I012`](results/int8/I012/fixed-4096-1024-c8/summary.json) |
| INT8 slots64 | C16 | 555.91 | **277.96** | 3.184s | 23.65ms | 通过，32/32 精确；六档内最高有效吞吐 | [`I007`](results/int8/I007/fixed-4096-1024-c16/summary.json) |
| INT8 slots64 | C32 | 750.81 | 375.40 | 5.630s | 34.01ms | **未通过 TTFT**，64/64 精确 | [`I009`](results/int8/I009/fixed-4096-1024-c32/summary.json) |
| INT8 slots64 | C64 | 889.89 | 444.95 | 10.355s | 57.24ms | **未通过两项**，128/128精确 | [10-21补充](submissions/20261021/result_int8.json) |
| INT8 slots64 | C128 | 899.22 | 449.61 | 62.322s | 62.16ms | **未通过两项**，256/256精确 | [10-21补充](submissions/20261021/result_int8.json) |

两种权重在本表均为 C1–C16通过、C32及以上未通过；在官方离散档中，最高有效单轮吞吐仍为BF16 **229.67**、INT8 **277.96 tok/s/卡**。C16–C32之间的细扫/三轮边界另表列出。BF16官方同档双卡基线C1/C8/C16/C32为21.195/182.734/304.034/433.847，C64/C128为545.911/577.456；新增两档吞吐算术提升+7.86%/+4.98%，但SLA未过，不作有效峰值。INT8新增C64/C128相对同一官方BF16基线+63.01%/+55.72%，不是同精度调参收益。TPS、TTFT、TPOT全部逐档对比见两份[最新报告索引](submissions/20261021/README.md)。官方基线见[基线帖](https://forum.sourcefind.cn/thread/d006a0e2-b624-11f1-93f4-0242ac150003)，最终得分由评审确认。

## 同 INT8 权重的优化前后

以下 A/B 使用**同一份官方 INT8 权重**、同卡、同槽位、同一轮正式请求。参考组已使用 FA3、BF16 KV 和禁用 custom all-reduce，并非未经设置的“开箱默认”或官方 INT8 基线。

| 场景 | 参考组双卡输出 | 优化组合双卡输出 | 组合提升 | 证据 |
|---|---:|---:|---:|---|
| 固定 4K/1K C16，slots32 | 367.79 tok/s | 555.62 tok/s | **+51.07%** | [`I005`](results/int8/I005/fixed-4096-1024-c16/summary.json) → [`I001`](results/int8/I001/fixed-4096-1024-c16/summary.json)，[审计](results/aggregate/int8-opt-vs-ref-fixed-20261015.json) |
| 公开 ShareGPT C64，slots64 | 538.10 tok/s | 797.94 tok/s | **+48.29%** | [审计](results/aggregate/int8-opt-vs-ref-sharegpt-20261015.json) |

INT8 slots64 在固定 4K C16 为 555.91 tok/s，较 slots32 的 555.62 基本持平；公开 ShareGPT C64 则从 slots32 的 603.49 提高到 slots64 的 797.94 tok/s（+32.22%）。[固定档审计](results/aggregate/int8-slots64-fixed-20261015.json) · [ShareGPT 审计](results/aggregate/int8-slots64-sharegpt-20261015.json)。公开 ShareGPT 文件身份尚未获主办方确认，**不能直接称为正式 ShareGPT 赛分**。

## 长输入：64K 输入 / 1K 输出

当前均值工作 SLA 为 TTFT `<20s` 且 TPOT `<60ms`。下表是**完整65536/1024 token请求**，不是短输出smoke。10月21日已补齐C8/C16/C32（各一轮）及C3（三轮），无论SLA是否通过都保留完整结果。

| 权重/配置 | 并发 | 双卡输出 tok/s | 单卡输出 tok/s | mean TTFT | mean TPOT | 工作 SLA | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| BF16 slots32 | C1 | 37.55 | 18.78 | 14.081s | 12.89ms | 通过，8/8 精确 | [`F023`](results/bf16/F023/fixed-65536-1024-c1/summary.json) |
| BF16 slots32 | C2 | 48.14 | 24.07 | 16.023s | 25.86ms | 通过，8/8 精确 | [`F029`](results/bf16/F029/fixed-65536-1024-c2/summary.json) |
| BF16 slots32 | C3（三轮） | 51.52 | 25.76 | 20.928s | 36.31ms | **合并均值未通过**，24/24精确、逐轮1/3通过 | [10-21补充](submissions/20261021/result_bf16.json) |
| BF16 slots32 | C4 | 55.28 | 27.64 | 25.150s | 46.36ms | **未通过 TTFT**，8/8 精确 | [`F029`](results/bf16/F029/fixed-65536-1024-c4/summary.json) |
| BF16 slots32 | C8 | 58.31 | 29.16 | 50.579s | 82.73ms | **未通过两项**，16/16精确 | [10-21补充](submissions/20261021/result_bf16.json) |
| BF16 slots32 | C16 | 59.15 | 29.57 | 153.144s | 91.25ms | **未通过两项**，32/32精确 | [10-21补充](submissions/20261021/result_bf16.json) |
| BF16 slots32 | C32 | 59.45 | 29.72 | 356.685s | 97.56ms | **未通过两项**，64/64精确 | [10-21补充](submissions/20261021/result_bf16.json) |
| INT8 slots64 | C1 | 44.68 | 22.34 | 11.268s | 11.39ms | 通过，8/8 精确 | [`I008`](results/int8/I008/fixed-65536-1024-c1/summary.json) |
| INT8 slots64 | C2 | 57.22 | 28.61 | 12.815s | 21.84ms | 通过，8/8 精确 | [`I013`](results/int8/I013/fixed-65536-1024-c2/summary.json) |
| INT8 slots64 | C3（三轮） | 61.57 | 30.78 | 15.771s | 31.69ms | **三轮均通过**，24/24精确 | [10-21补充](submissions/20261021/result_int8.json) |
| INT8 slots64 | C4 | 66.37 | 33.19 | 20.111s | 38.76ms | **未通过 TTFT**，8/8 精确 | [`I014`](results/int8/I014/fixed-65536-1024-c4/summary.json) |
| INT8 slots64 | C8 | 69.65 | 34.82 | 40.387s | 70.20ms | **未通过两项**，16/16精确 | [10-21补充](submissions/20261021/result_int8.json) |
| INT8 slots64 | C16 | 71.43 | 35.71 | 125.065s | 77.63ms | **未通过两项**，32/32精确 | [10-21补充](submissions/20261021/result_int8.json) |
| INT8 slots64 | C32 | 71.38 | 35.69 | 296.633s | 80.77ms | **未通过两项**，64/64精确 | [10-21补充](submissions/20261021/result_int8.json) |

新增C3后，BF16三轮合并均值仍以C2为最高已测通过档；INT8更新为C3（3/3通过），C4历史单轮未过。INT8 C4仅超TTFT约111ms，本次没有重测，不假称稳定失败。两种精度高并发固定档均已完成，未过SLA的吞吐不作有效峰值。基线对比、逐轮结果和限制详见[补充报告](submissions/20261021/README.md)。

## 单轮历史：满足 SLA 的最高单卡吞吐对比

本节保留单轮选优历史；当前申报建议以后一节的关键档位三轮均值为主，不把单轮最好值替代三轮结果。

这里的“最高吞吐”定义为：**在已完成、满足当前均值 SLA 的并发档中，取单卡输出 tok/s 最大者**。赛题规范的两张基线表存在冲突：BW100/SGLang 4K/1K 的“官方最大并发 Baseline”填 **C1**，但固定档表中 C8 的 mean TTFT **3.816s**、mean TPOT **40.076ms**，均满足文档的 `<5s` / `<50ms`；C16 的 mean TTFT **6.801s** 则未通过。因此，若**按固定档均值重算**，4K 基线最大通过档应为 **C8**，双卡 **182.734 tok/s**，即 **91.367 tok/s/卡**。64K 固定档 C1 的 mean TTFT **19.266s**、mean TPOT **54.682ms** 满足 `<20s` / `<60ms`，C8 未通过，与公布的 C1 最大档一致；其双卡吞吐为 **13.616 tok/s**，即 **6.808 tok/s/卡**。下表优先展示按固定档均值重算的对比；它是**工作口径**，不能代替主办方公布的最大并发测试结论。公式为 `(候选单卡吞吐 ÷ 对应场景的基线单卡吞吐 − 1) × 100%`。双方都是 TP2，因此用双卡吞吐直接相除也得到同一比例。

| 场景与权重 | 官方 BF16 固定档基线：按均值筛选的最大通过档 / 单卡 tok/s | 本方案：已测 SLA 内最高吞吐档 / 单卡 tok/s | 相对该基线算术提升 | 证据 |
|---|---:|---:|---:|---|
| 4K/1K · BF16 slots32 | C8 / 91.3670 | C16 / **229.6728** | **+151.37%** | [`F026`](results/bf16/F026/fixed-4096-1024-c16/summary.json) |
| 4K/1K · INT8 slots64，仅公开六档 | C8 / 91.3670 | C16 / **277.9563** | **+204.22%（条件性）** | [`I007`](results/int8/I007/fixed-4096-1024-c16/summary.json) |
| 4K/1K · INT8 slots64，含额外细扫 | C8 / 91.3670 | C28 / **357.1228** | **+290.87%（条件性）** | [`I010`](results/peak-evidence/I010/fixed-4096-1024-c28/summary.json) |
| 64K/1K · BF16 slots32 | C1 / 6.8080 | C2 / **24.0678** | **+253.52%** | [`F029`](results/bf16/F029/fixed-65536-1024-c2/summary.json) |
| 64K/1K · INT8 slots64 | C1 / 6.8080 | C2 / **28.6110** | **+320.26%（条件性）** | [`I013`](results/int8/I013/fixed-65536-1024-c2/summary.json) |

这是**跨并发档的有效峰值对比**，不同于上文“同一并发档位”的性能提升。4K 的 BF16 细扫 C18 也曾单轮通过，但吞吐低于 C16，故未选为“最高吞吐”；INT8 C28 的 mean TTFT 为 4.966s，离 5s 门槛仅约 34ms，不能宣称稳定成绩。INT8 行使用**官方 BF16 固定档基线作条件性分母**，不是同权重调参收益；主办方尚未明确 INT8 正式性能基线，不能把条件性比例当作官方最终得分。若评审坚持直接采用赛题规范另表**公布**的 4K SLA 最大并发 C1，则对应分母是 10.5975 tok/s/卡，而非本表的 91.3670；两者不可混称。基线帖另有 p95 SLA 口径，与主办方后来澄清的赛题规范**均值**口径不同。正式评审如何解决两张表的矛盾，以及选档、轮数和提升率分母，仍待主办方确认。

## 最终候选关键档位：三轮复测

以下为 2026-10-16 冻结代码后的复测，沿用最终 BF16 slots32 / INT8 slots64 配置、原版 GEMM、禁用 custom all-reduce，在同一对物理 DCU2/3 顺序运行。每档两轮预热（每轮 C 请求），三轮正式（每轮 `max(8, 2*C)` 请求）；表中吞吐为三轮算术平均，TTFT/TPOT 为全部正式请求的均值。原始单轮六档表仍保留为筛选历史，不改写成三轮结果。

| 权重 | 场景 | 并发 | 单卡输出 tok/s（三轮均值） | mean TTFT | mean TPOT | 逐轮均值 SLA 通过 | 合并均值 SLA | 相对官方 BF16 SLA 内基线 |
|---|---|---:|---:|---:|---:|---|---|---:|
| BF16 slots32 | 4K/1K | C16 | **227.44** | 4.3053s | 28.0975ms | **3/3** | 通过 | **+148.93%** |
| INT8 slots64 | 4K/1K | C16 | **273.54** | 3.1837s | 23.6197ms | **3/3** | 通过 | **+199.38%** |
| INT8 slots64 | 4K/1K | C28 | **355.03** | 4.9825s | 31.8910ms | **2/3** | 通过，余量约 17ms | **+288.57%** |
| INT8 slots64 | 4K/1K | C27 | **348.89** | 4.8115s | 31.3876ms | **3/3** | 通过 | **+281.85%** |
| BF16 slots32 | 64K/1K | C2 | **23.56** | 16.0311s | 26.3923ms | **3/3** | 通过 | **+246.00%** |
| INT8 slots64 | 64K/1K | C2 | **28.38** | 12.8162s | 21.9530ms | **3/3** | 通过 | **+316.91%** |

C28 三轮平均 TTFT 分别为 **4.9693 / 4.9086 / 5.0697s**，第 3 轮超限；完整保留该轮。合并均值通过与“三轮全过”是两个判断，后者是本项目额外的稳定性要求，不冒称官方明文要求。补测 C27 三轮全部通过，TTFT 分别为 **4.8249 / 4.7251 / 4.8845s**，最差轮仍有约 116ms 余量，平均吞吐比 C28 低约 1.73%，因此优先作为稳妥提交候选；若评审接受合并均值，C28 的更高吞吐参考同时保留。两种精度的 64K C2 三轮复测均已完成。

三轮结果按与上一节相同的官方 BF16 参考分母计算（4K C8 91.367、64K C1 6.808 tok/s/卡）；不同权重的比例是跨精度参考对照，不是同 INT8 权重调参收益，也不是主办方确认的最终得分。

[三轮逐请求审计汇总](results/verification-20261016/summary.json)包含全部轮次及原始请求文件 SHA256；[原始日志包](results/verification-20261016/VERIFY_FINAL_20261016_01_raw.tar.gz)与[校验值](results/verification-20261016/VERIFY_FINAL_20261016_01_raw.tar.gz.sha256)包含本轮五个服务运行的完整服务日志、逐请求记录、各轮汇总、硬件/驱动声明、时间戳与冻结代码快照。批次曾在服务退出后显存尚未释放时被空闲检查拒绝，原批次 status=1 如实保留；确认空闲后以独立恢复记录完成剩余阶段，不覆写失败日志。此包只覆盖本节复测，不代表旧单轮全档已经补齐三轮。

## ShareGPT 标准数据集

赛题要求测 C1/C4/C16/C32/C64，记录吞吐、TTFT、TPOT 与成功率；**没有为 ShareGPT 单独给出数值 SLA**，不能直接套用随机 4K 的 5s/50ms。下面使用公开 ShareGPT 文件，正式请求每档 1000 条、预热 `2×并发`，每档 1 轮；本次按用户决定保留现有单轮数据，官方教程三轮的差异在提交报告中披露。INT8 C1/C4/C16/C32 来自同一次 I015 运行，C64 来自此前 I004；五档的输入长度指纹和输入/输出 token 总数一致。公开文件是否与主办方最终指定文件完全一致，尚未确认。

| 权重/配置 | 并发 | 双卡输出 tok/s | 单卡输出 tok/s | mean TTFT | mean TPOT | 完成 | 相对官方 BF16 同档基线吞吐 | 原始运行 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BF16 slots32 | C1 | 61.08 | 30.54 | 240.78ms | 17.04ms | 1000/1000 | **+184.72%** | SG014 |
| BF16 slots32 | C4 | 200.27 | 100.13 | 243.18ms | 21.11ms | 1000/1000 | **+88.57%** | SG013 |
| BF16 slots32 | C16 | 420.52 | 210.26 | 285.19ms | 40.05ms | 1000/1000 | **+36.78%** | SG013 |
| BF16 slots32 | C32 | 552.63 | 276.31 | 461.22ms | 61.42ms | 1000/1000 | **+26.75%** | SG008 |
| BF16 slots32 | C64 | 552.00 | 276.00 | 11232.45ms | 61.00ms | 1000/1000 | **+4.51%** | SG011 |
| INT8 slots64 | C1 | 75.21 | 37.61 | 255.32ms | 13.59ms | 1000/1000 | **+250.57%** | I015 |
| INT8 slots64 | C4 | 223.38 | 111.69 | 262.45ms | 19.14ms | 1000/1000 | **+110.33%** | I015 |
| INT8 slots64 | C16 | 432.93 | 216.47 | 295.97ms | 40.06ms | 1000/1000 | **+40.81%** | I015 |
| INT8 slots64 | C32 | 591.76 | 295.88 | 344.34ms | 58.61ms | 1000/1000 | **+35.73%** | I015 |
| INT8 slots64 | C64 | **797.94** | **398.97** | 557.75ms | 90.59ms | 1000/1000 | **+51.08%** | I004 |

INT8 的五档 ShareGPT 单轮结果现已齐全。[I015/I004 逐档审计汇总](results/aggregate/int8-sharegpt-five-tiers-20261016.json)记录精确吞吐、完成数、时延、样本指纹与原始文件哈希；完整 JSONL 和服务器日志已附独立提交包。I004 与 I015 的启动脚本 SHA 不同，包内按 SHA 提供对应历史源码，不声称同一代码快照或同一次 uptime；按用户决定不重跑。BF16 C64 的吞吐与 C32 接近，但 TTFT 明显恶化。INT8 C64 的 [同权重 A/B](results/aggregate/int8-opt-vs-ref-sharegpt-20261015.json) 和 [slots32/64 对照](results/aggregate/int8-slots64-sharegpt-20261015.json)已附；BF16 C32/C64 的 [审计汇总](results/aggregate/sharegpt-slots32-20261014.json)与[槽位对照](results/aggregate/sharegpt-c64-slots-20261015.json)另附。

官方 BW100/SGLang BF16 ShareGPT 双卡基线 C1/C4/C16/C32/C64 分别为 21.454/106.207/307.450/435.996/528.158 tok/s。表格的提升比例均按**同一并发档**计算，双卡与单卡比例相同；因此 INT8 五行明确表示**相对官方 BF16 同档基线的算术提升**，不是“同 INT8 权重优化前后”的收益。例如 C64 的 **+51.08%** 不同于上文相对同 INT8 权重参考组的 **+48.29%**。由于公开数据文件身份和轮数尚未与官方完全对齐，且官方 INT8 性能分母未明确，这些只是参考对照，不是正式榜单提升率。

## 最终候选的 GSM8K 精度

2026-10-16 已分别完成 BF16 slots32 与官方 INT8 slots64 的完整 GSM8K 测试，均使用 IFB TP2、禁用 custom all-reduce。评测采用固定 OpenCompass 客户端与本地 v3 协议：完整 test 1319 题、4-shot CoT、temperature 0、seed 0、客户端并发 4、生成上限 16384。两次使用相同数据、提示、答案后处理与审计脚本，提示消息指纹一致。

| 最终候选 | 正确 / 总题数 | 准确率 | 相对 BF16 的准确率变化 | 正常结束 / 截断 | 严格协议审计 | 证据 |
|---|---:|---:|---:|---:|---|---|
| BF16 slots32 | 1291 / 1319 | **97.8772%** | — | 1319 / 0 | 通过 | [ACC_BF16_FINAL01](results/accuracy/ACC_BF16_FINAL01/accuracy-full-v3-audit-summary.json) |
| INT8 slots64 | 1287 / 1319 | **97.5739%** | **−0.3033 个百分点** | 1319 / 0 | 通过 | [ACC_INT8_FINAL01](results/accuracy/ACC_INT8_FINAL01/accuracy-full-v3-audit-summary.json) |

两套结果均无空响应，`protocol_valid=true`、`problems=[]`，runner均以状态0结束，均高于GSM8K 95.0%门槛。10月21日官方补充通知明确只评GSM8K，不再要求其他三个精度集；最终准入仍由评审核验本地协议与原件。旧E013存在截断，本表以最终候选结果为准；完整逐请求响应与服务日志见原独立提交包。

## 提交口径与保留限制

- BF16与INT8全量GSM8K已完成并通过本地协议审计；官方最新补充通知不再要求另外三个精度集，不将其列为必补项。
- 按10月21日补充要求完成两精度全部固定档和新增边界。固定档及ShareGPT仍为单轮、边界三轮；与教程全档三轮的差异如实披露，不把单轮冒充三轮。
- 两套独立提交包已包含引用的全档、ShareGPT、GSM8K 完整逐请求记录、服务日志、硬件/驱动、结束时间和对应源码。`summary.json` 中静态 `method` 文案虽写“三轮”，应以真实 `measure_rounds`、逐轮记录和 runner 参数为准。
- Demo启动/访问脚本已提供，未录制视频。论坛ID为zack233；原包内“待填写”发帖模板保留作历史文件，最新[补充回复文案](submissions/20261021/FORUM_SUPPLEMENT.md)已填好，不代发。截图未伪造，保留带时间戳命令行原文。

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

# ShareGPT：须先核对公开文件哈希；两种权重的五档现均有单轮数据
DCU_DEVICES=2,3 CONTAINER=zk-qwen38 SHAREGPT_ROUNDS=1 \
  SHAREGPT_CONCURRENCIES='1 4 16 32 64' \
  bash /data/zk/qwen38/scripts/run_sharegpt_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots32_no_custom_ar
DCU_DEVICES=2,3 CONTAINER=zk-qwen38-int8 \
  MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8 \
  SERVED_MODEL_NAME=qwen38-int8 SHAREGPT_ROUNDS=1 SHAREGPT_CONCURRENCIES='1 4 16 32 64' \
  bash /data/zk/qwen38/scripts/run_sharegpt_round.sh <new-run-id> \
  fa3_hcu_contigfa_eagle314_slots64_no_custom_ar
```

每次换全新 run ID；runner 会在退出时停止本次启动的服务。`scripts/run_server.sh` 保留历史实验变体，**当前有效配置只选上面列出的 no-custom-all-reduce 变体**。这些是供评审复现的入口，本次整理没有执行新测试；完整的安装、启动、Demo、Benchmark 说明见独立提交包 README。

## 目录

- [`scripts/`](scripts/)：服务启动/停止、DCU 占用检查、固定长度与 ShareGPT benchmark、结果汇总脚本。
- [`results/bf16/`](results/bf16/) 和 [`results/int8/`](results/int8/)：本次挑选的逐档 `summary.json`。
- [`results/aggregate/`](results/aggregate/)：同权重 A/B、slots 比较和 ShareGPT 审计汇总。
- [`results/peak-evidence/`](results/peak-evidence/)：不混入官方六档主表的额外细扫峰值证据。
- [`results/accuracy/`](results/accuracy/)：BF16 与 INT8 最终候选的全量 GSM8K 审计摘要与 runner 记录。
- [`results/verification-20261016/`](results/verification-20261016/)：关键档位三轮复测、边界超限轮次与完整原始日志包。
- [`docs/submission-final-20261016/`](docs/submission-final-20261016/)：分别整理的 BF16 / INT8 结果卡草稿与环境声明。

_更新：2026-10-21。按官方要求补齐固定缺档、SLA边界与三指标基线对比；原提交包不改，新补充单独归档。_
