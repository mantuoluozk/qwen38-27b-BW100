【论坛 ID】zack233

【数据补充说明】按官方补充通知，补齐BF16、官方INT8两种精度的固定4K/1K C64/C128及64K/1K C8/C16/C32，共10个缺档；另补4个SLA边界档。新增测试时间为2026-10-21 10:21～13:31（UTC+8），原提交结果和原始日志不覆盖。

【完整报告链接】
BF16：https://github.com/mantuoluozk/qwen38-27b-BW100/blob/main/docs/supplement-20261021/bf16.md
INT8：https://github.com/mantuoluozk/qwen38-27b-BW100/blob/main/docs/supplement-20261021/int8.md

两份报告现均列明GSM8K、ShareGPT C1/C4/C16/C32/C64、固定4K C1/C8/C16/C32/C64/C128、固定64K C1/C8/C16/C32及SLA边界；TPS、TTFT、TPOT采用均值并逐项与原官方BF16基线对比。物理卡型hy-smi记录为BW100，与申报一致。

【SLA边界】BF16 4K C19合并均值通过但仅1/3轮通过，三轮均通过的最高已测并发为C18；64K C2三轮均通过，C3合并均值未通过。INT8 4K C28合并均值通过但仅2/3轮通过，稳妥C27三轮均过；64K补测C3三轮均过，输出30.7849 tokens/s/卡、TTFT15.7707秒、TPOT31.6928毫秒。最大并发和最大有效吞吐分开说明，不删除超限轮次。

【原始数据】新增1262个正式请求全部成功且输入/输出长度精确，1078个预热请求通过。完整启动、占用、逐请求、统计及结束时间已独立打包，附件索引：
https://github.com/mantuoluozk/qwen38-27b-BW100/tree/main/submissions/20261021

【边界声明】固定档/ShareGPT保持单轮，关键边界三轮；精度沿用原全量GSM8K（BF16 97.8772%，INT8 97.5739%），不重测、不冒称其他精度数据已完成。原10-16独立包及SHA256不变。新增补充是否纳入最终评比，以组委会核验为准。
