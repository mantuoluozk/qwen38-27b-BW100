【论坛 ID】待填写

【参赛组别】先锋组 BW100；子赛道 B（SGLang）；IFB TP2，2 卡；官方 INT8 W8A8，slots64

【参选榜单】吞吐榜 + 时延榜（最终准入由评审确认）

【吞吐结果】4K-in/1K-out C27，三轮平均 **348.89 tokens/s/卡**，mean TTFT 4811.48ms、TPOT 31.39ms，三轮均值 SLA 均通过。按官方 BF16 固定档均值重算 C8 的 91.367 tok/s/卡分母，算术提升 **281.85%**。64K/1K C2 三轮平均 **28.38 tok/s/卡**，相对 C1 基线 6.808 的算术提升 **316.91%**。

【低并发时延】4K/1K C1 单轮：TTFT mean 592.81ms / p95 597.03ms，TPOT mean 9.54ms / p95 11.30ms，输出 49.46 tok/s/卡。

【精度】GSM8K 全量 1287/1319 = 97.5739%；本地固定 OpenCompass 4-shot CoT 协议，1319 条正常结束、无截断。其他三个精度数据集未测，不声明完整四项准入通过。

【测试轮数】固定全档和 ShareGPT 五档均单轮；关键边界三轮。轮数差异如实声明，不把单轮改写为三轮。ShareGPT 使用公开文件并附哈希，文件身份待主办方确认。

【代码仓库】https://github.com/mantuoluozk/qwen38-27b-BW100

【实测报告】https://github.com/mantuoluozk/qwen38-27b-BW100/blob/main/docs/submission-final-20261016/int8.md

【独立提交包】https://github.com/mantuoluozk/qwen38-27b-BW100/raw/refs/heads/main/submissions/20261016/qwen38-bw100-sglang-int8-20261016.tar.gz

【报告与原始日志】随附本精度独立提交包，PERFORMANCE_REPORT.md 为五段式报告，logs/ 含硬件/驱动、完整启动、benchmark 请求与结束统计/时间戳；FILES.sha256 校验全部附件。Demo 启动/访问脚本已附，视频未录制。

【声明】结果来自所声明环境的真实日志，未伪造或拼接。性能比例采用正文说明的参考分母，非已确认官方排名。INT8 C28 的三轮 TTFT 4.9693/4.9086/5.0697s，第 3 轮超限；合并均值 4.9825s 仍通过，但只有 2/3 轮通过。C27 三轮均过，作为稳妥申报档位；保留 C28 全部原件，不删除超限轮次。
