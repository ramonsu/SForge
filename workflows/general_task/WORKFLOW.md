# General Task

这是一个通用任务认知环境。Workflow 是声明式的有向有环状态空间：状态节点同时声明
认知说明、目标、评估标准、Memory scope 与 Capability scope，边声明允许的迁移条件。

Agent 决定是否申请本 Workflow，以及何时提出状态迁移。Harness 只校验请求是否命中已声明
的节点和边，并原子装载相应资源；它不判断任务是否已经“完成审计”，也不替 Agent 规划。

`active -> audit -> active` 是合法循环，不是由 Harness 自动推进的固定流水线。
