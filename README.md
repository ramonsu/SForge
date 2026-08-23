# SForge V1

SForge 是一个最小、可测试的 Agent 运行时。它把“思考”和“机制”严格分开：

- Agent 只读取上下文，并返回 `FinalAnswer` 或 `ActionRequest`。
- Harness 只管理生命周期、上下文装配、准入和 Capability 执行，不理解任务语义。
- Workflow 只定义 Agent 当前可见的状态空间，不是角色系统或 DAG 执行器。
- Memory 是可替换的数据来源；进入状态空间后，Harness 才按 scope 挂载更多记忆。
- Persona 只参与最终表达，不参与决策和运行控制。

## 架构

```text
User
  |
  v
Harness
  |-- AgentManager -------- AgentProcess 生命周期与独立进程
  |-- ContextManager ------ 分别构造 Operational / Presentation Context
  |-- WorkflowRegistry ---- 加载薄 Workflow 定义
  |-- MemoryProvider ------ InMemory / SQLite
  |-- CapabilityRegistry -- 注册、Schema 校验
  `-- AdmissionPolicy ---- 根据 RuntimeState 做准入
         |
         v
      Agent
      |-- immutable Persona reference
      `-- ReasoningProcess
```

Agent 本身不持有 MemoryProvider、CapabilityRegistry、WorkflowRegistry 或权限判断器。它不能创建
Agent，也不能直接执行外部操作。

## 一次运行

```text
create AgentProcess + RuntimeState
              |
              v
      OperationalContext
              |
              v
 Agent -> FinalAnswer | ActionRequest
              |               |
              |               v
              |       Harness admission + execution
              |               |
              |<------ structured ActionResult
              v
   PresentationContext
   (Persona + communication memory + draft)
              |
              v
         User response
```

`AgentProcess` 是可销毁的运行身份，`RuntimeState` 是独立的可变状态。Persona 是 Agent
生命周期内不可变的静态引用，不属于 RuntimeState。

Persona 和沟通偏好从源头就不会进入 OperationalContext；最终表达的输出也不会重新进入
Action loop。因此 Persona 不能改变 Workflow、权限、Capability、Memory scope 或生命周期。

## 目录

```text
agent/                    Agent 协议、LLM 客户端与独立 Worker
harness/
  core.py                 小型 Harness API 与有界运行循环
  models.py               AgentProcess、RuntimeState、Decision、Result
  capability.py           统一 Capability 与 AdmissionPolicy
  context_manager.py      两类 Context 的独立装配
  memory_manager.py       MemoryProvider 与两种实现
  workflow_*.py           薄 Workflow 加载/注册
  persona.py              不可变 Persona 配置
capabilities.py           echo/read_text/write_text 内置能力
config/persona.json       默认 Persona
memory/core_memory.json   Core Memory 引导数据
workflows/                Workflow JSON + Markdown
tests/                    单元、边界和确定性 E2E 测试
runtime.py                依赖组装和 Python 入口
main.py                   命令行入口
SForge_V1_PLAN.md         V1 设计基线
```

`harness/tool_manager.py` 和 `harness/shared_memory.py` 只保留轻量导入兼容层；运行核心不再
存在 Skill/Tool 双体系或第二套 Memory 实现。

## 安装与 API 配置

需要 Python 3.10 或更新版本。在 PowerShell 中：

```powershell
cd "D:\Desktop\创作\SForge"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

专门的模型 API 配置文件是根目录的 `.env`。模板为：

```env
DEEPSEEK_API_KEY=replace-with-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

底层使用 OpenAI-compatible 客户端，因此兼容服务可以通过更换这三个值接入。不要提交
`.env`；仓库只提交 `.env.example`。

Persona 在 `config/persona.json` 中配置：

```json
{
  "id": "persona_ada",
  "version": "1.0",
  "name": "Ada",
  "description": "A curious and careful long-term personal assistant",
  "traits": ["analytical", "patient"],
  "communication_style": "concise and structured"
}
```

可以增加 `favorite_topics`、`writing_style` 等表达元数据，但不能声明 permissions、tools、
capabilities、workflow 或 memory_scope 等运行控制字段。

## 启动

```powershell
python main.py
```

输入任务后，SForge 会为这次请求创建独立 AgentProcess，运行有界循环，返回结果后销毁它。
输入 `exit`、`quit` 或 `退出` 结束程序。

也可以作为 Python API 使用：

```python
from runtime import AgentApplication

app = AgentApplication()
try:
    answer = app.handle("把 README.md 的第一段概括成一句话")
    novel_answer = app.handle("给我一个科幻短篇梗概", workflow_id="novel_writing")
finally:
    app.close()
```

底层 Harness API 保持小而显式：

```python
from harness.models import ActionRequest, TaskSpec
from runtime import create_runtime

runtime = create_runtime()
process = runtime.create_agent(TaskSpec("echo test"))
result = runtime.execute_action(
    process.id,
    ActionRequest("echo", {"text": "hello"}),
)
runtime.terminate_agent(process.id)
runtime.close()
```

## Workflow

每个 Workflow 是一个 JSON 定义和一个 Markdown 环境说明：

```text
workflows/<workflow-id>/
  workflow.json
  workflow.md
```

最小定义：

```json
{
  "id": "novel_writing",
  "initial_state": "writing",
  "states": {
    "writing": {
      "allowed_capabilities": ["echo", "read_text", "write_text"],
      "memory_scope": "workflow",
      "context_sources": []
    }
  }
}
```

V1 只装载初始状态，不在 Harness 中实现 DAG、角色切换或业务阶段调度。未来增加阶段时，也应由
Agent 提出状态变化，通过 Harness 接口验证，而不是让 Harness 决定小说应该怎样写。

## Memory 与 Capability

MemoryProvider 的最小契约是 `write/retrieve/get/close`。默认运行使用
`SQLiteMemoryProvider`，测试使用 `InMemoryMemoryProvider`。持久数据写入
`memory/runtime.sqlite3`。

V1 只有一种 Capability 抽象。内置能力：

- `echo`：返回文本。
- `read_text`：读取工作区内 UTF-8 文本。
- `write_text`：写入工作区内 UTF-8 文本。

文件路径会被限制在工作区内。未知、参数无效或未授权的请求返回 `rejected`；执行异常返回
`failed`；成功返回 `success`。这些结果都作为结构化 Observation 回到同一 Agent。

## 测试

完整确定性测试不访问网络，也不依赖真实模型：

```powershell
python -B -W error::ResourceWarning -m unittest discover -s tests -v
```

覆盖内容包括：

- 生命周期、进程隔离与非法状态转换；
- RuntimeState 与 AgentProcess 分离；
- Capability 注册、Schema、准入、失败映射和路径逃逸；
- MemoryProvider 契约与 scope；
- Direct 模式写入、读取、最终回答的端到端闭环；
- 小说 Workflow 使用相同 Agent 抽象；
- Persona/沟通记忆与 OperationalContext 的单向隔离；
- 有界循环及清理行为。

真实模型验证通过 `python main.py` 单独执行，以免常规测试依赖 API 或网络。

## V1 明确不做

- 多 Agent、父子 Agent、角色 Agent 或 Agent 类型树；
- Workflow DAG 执行器和内置业务规划；
- Skill/Tool 两套抽象；
- Scheduler、事件总线、分布式执行、Checkpoint/Restore；
- Persona Registry 或完整 Presentation Engine；
- Persona 驱动任何运行权限与控制。
