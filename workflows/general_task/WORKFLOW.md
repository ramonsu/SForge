# General Task

这是一个薄的通用任务环境。Workflow 只声明初始状态可见的 Capability、Memory scope
与环境说明；如何解决任务完全由同一种 AgentProcess 决定，所有副作用都必须经过 Harness。

V1 不执行 Workflow DAG，也不把阶段、角色或任务语义写进 Harness。
