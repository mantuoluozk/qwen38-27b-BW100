# 2026-10-16 提交材料

测试数据已封存：全档固定长度与 ShareGPT 每档单轮，关键 SLA 边界另有三轮；本次只整理材料，没有追加推理或精度测试。两精度独立打包，实际服务均为 SGLang 0.5.12 / BW100 两卡 / IFB TP2。

| 方案 | 完整提交包 | 五段式报告 | 提交帖文案 | 文件数 |
|---|---|---|---|---:|
| BF16 slots32 | [下载 BF16（约 7.9 MiB）](qwen38-bw100-sglang-bf16-20261016.tar.gz) | [BF16 报告](../../docs/submission-final-20261016/bf16.md) | [FORUM_BF16.md](FORUM_BF16.md) | 333 |
| 官方 INT8 slots64 | [下载 INT8（约 9.9 MiB）](qwen38-bw100-sglang-int8-20261016.tar.gz) | [INT8 报告](../../docs/submission-final-20261016/int8.md) | [FORUM_INT8.md](FORUM_INT8.md) | 476 |

每包包含 `README.md`、`env_declare.yaml`、`config/`、`scripts/run.sh`、`patch/`、`requirements.txt`、`PERFORMANCE_REPORT.md`、`FORUM_POST.md`、`logs/`、`result/`、固定 OpenCompass 源码归档、`FILES.sha256`。模型和数据集大文件不附，提供路径、来源与哈希；推理依赖由指定 DTK 镜像提供。

## 申报数据摘要

| 方案 | 4K 稳妥档：单卡 tok/s / 相对官方基线 | 64K 已测档：单卡 tok/s / 相对官方基线 | GSM8K |
|---|---|---|---|
| BF16 | C16：227.44 / +148.93% | C2：23.56 / +246.00% | 1291/1319，97.8772% |
| INT8 | C27：348.89 / +281.85% | C2：28.38 / +316.91% | 1287/1319，97.5739% |

以上吞吐均取边界三轮平均且每轮满足均值 SLA；提升使用报告声明的官方 BF16 参考分母（4K C8 单卡 91.367，64K C1 单卡 6.808 tok/s），不是主办方已确认的最终得分。INT8 C28 三轮平均 355.03 tok/s/卡、合并均值通过，但第 3 轮 TTFT 超限，原件完整保留，优先申报余量更大的 C27。

## 已核验与保留限制

[verification.json](verification.json)记录压缩包外层哈希、包内全部文件哈希、9 个固定长度档、5 个 ShareGPT 档、GSM8K 完整审计、历史启动源码哈希和 Python 语法校验；Bash 语法另外在原服务器检查通过。没有执行新 Demo，也没有将静态检查称为模型实跑验证。

论坛 ID 尚未提供，两个提交帖中保留“待填写”；演示视频未录制，已提供启动和访问脚本。LiveCodeBench v6 / RealWorldQA / MathVision 按用户要求未测；不声明四项完整准入通过。官方教程全档三轮与本次单轮提交的差异、ShareGPT 文件身份、基线表冲突和跨轮 SLA 聚合口径均在报告中披露，不自动改写为已经获准。

校验下载文件：

```bash
sha256sum -c SHA256SUMS
tar -xzf qwen38-bw100-sglang-bf16-20261016.tar.gz
cd bf16
sha256sum -c FILES.sha256
```

INT8 对应解压目录为 `int8`。数据索引另公开为 [result_bf16.json](result_bf16.json) / [result_int8.json](result_int8.json)，环境声明为 [env_bf16.yaml](env_bf16.yaml) / [env_int8.yaml](env_int8.yaml)。
