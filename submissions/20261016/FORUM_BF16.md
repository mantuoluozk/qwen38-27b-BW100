【论坛 ID】待填写

【参赛组别】先锋组 BW100；子赛道 B（SGLang）；IFB TP2，2 卡；BF16，slots32

【参选榜单】吞吐榜 + 时延榜（最终准入由评审确认）

【吞吐结果】4K-in/1K-out C16，三轮平均 **227.44 tokens/s/卡**，mean TTFT 4305.29ms、TPOT 28.10ms，三轮均值 SLA 均通过。按官方 BF16 固定档均值重算 C8 的 91.367 tok/s/卡分母，算术提升 **148.93%**。64K/1K C2 三轮平均 **23.56 tok/s/卡**，相对 C1 基线 6.808 的算术提升 **246.00%**。

【低并发时延】4K/1K C1 单轮：TTFT mean 771.28ms / p95 776.13ms，TPOT mean 12.78ms / p95 14.80ms，输出 36.97 tok/s/卡。

【精度】GSM8K 全量 1291/1319 = 97.8772%；本地固定 OpenCompass 4-shot CoT 协议，1319 条正常结束、无截断。其他三个精度数据集未测，不声明完整四项准入通过。

【测试轮数】固定全档和 ShareGPT 五档均单轮；关键边界三轮。轮数差异如实声明，不把单轮改写为三轮。ShareGPT 使用公开文件并附哈希，文件身份待主办方确认。

【代码仓库】https://github.com/mantuoluozk/qwen38-27b-BW100

【实测报告】https://github.com/mantuoluozk/qwen38-27b-BW100/blob/main/docs/submission-final-20261016/bf16.md

【独立提交包】https://github.com/mantuoluozk/qwen38-27b-BW100/raw/refs/heads/main/submissions/20261016/qwen38-bw100-sglang-bf16-20261016.tar.gz

【报告与原始日志】随附本精度独立提交包，PERFORMANCE_REPORT.md 为五段式报告，logs/ 含硬件/驱动、完整启动、benchmark 请求与结束统计/时间戳；FILES.sha256 校验全部附件。Demo 启动/访问脚本已附，视频未录制。

【声明】结果来自所声明环境的真实日志，未伪造或拼接。性能比例采用正文说明的参考分母，非已确认官方排名。BF16 4K C16、64K C2 三轮均通过当前均值 SLA。未把有限已测档位称为全局最大；内部细扫不混入全档主表。
