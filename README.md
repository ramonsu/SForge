# SForge V1.6

SForge 是一个最小、可测试的 Agent 运行时。它把“思考”和“机制”严格分开：

- Agent 只读取上下文，并返回 `FinalAnswer`、`ActionRequest`、`ResourceBindingRequest`、
  `WorkflowRequest` 或 `WorkAssignmentRequest`。
- Harness 只是稳定接口并转发调用；确定性的生命周期、上下文装配、准入和 Capability
  执行位于 RuntimeEngine，二者都不理解任务语义。
- Workflow 是厚的声明式有向有环状态空间，不是角色系统或自动图执行器。
- Identity 跨临时 AgentProcess 保持连续；CognitivePolicy 与 Profession 可独立按需绑定。
- CognitivePolicy 只改变合法候选信息的排序偏置；Profession 只提供专业资源和 Skill 引用。
- `author/editor/reader/developer/reviewer` 是 Assignment 内的 WorkRole，不是独立 runtime state。
- WorkAssignment 显式绑定 Process、Workspace、Role、Task、可选 Workflow 与 Grants；
  Assignment 结束时立即撤销临时权限。
- Workspace 是可发现的项目实体；只有活动 WorkAssignment 才装载其内部上下文、Archive scope
  和 local skills。
- Memory 是可替换的数据来源；进入状态空间后，RuntimeEngine 才按 scope 挂载更多记忆。
- Persona 只参与最终表达，不参与决策和运行控制。
- RuntimeEvent 只记录机制边界，EventLogger 与 Inspector 都没有控制权限。

## 架构

```text
Desktop UI / future client
  |
  v
RunService
  |
  v
Harness
  |  thin public API
  v
RuntimeEngine
  |-- AgentManager -------- AgentProcess 生命周期与独立进程
  |-- ContextManager ------ 构造内部 Operational truth、四区模型投影与响应渲染上下文
  |-- WorkflowRegistry ---- 加载有环 Workflow 状态空间
  |-- CognitivePolicy ----- 合法候选信息的认知排序偏置
  |-- Profession ---------- 专业知识、专业检索、方法与 Skill 引用
  |-- SkillRegistry ------- 声明式方法资源，不能执行外部动作
  |-- Workspace ----------- 项目档案、项目检索与 local skills
  |-- WorkAssignment ------ Workspace + Role + Task + Workflow? + Grants
  |-- MemoryProvider ------ InMemory / SQLite
  |-- CapabilityRegistry -- 注册、Schema 校验
  |-- AdmissionPolicy ---- 根据 RuntimeState 做准入
  |-- EventLogger -------- 有界历史与只读实时订阅
  `-- RuntimeInspector --- 只读运行快照
         |
         v
      Agent
      |-- immutable Identity reference
      |-- immutable Persona reference
      `-- ReasoningProcess
```

Harness 不持有任务语义，只转发稳定 API。RuntimeEngine 只做确定性协调；Agent 本身不持有
MemoryProvider、CapabilityRegistry、WorkflowRegistry 或权限判断器，也不能直接执行外部操作。
RunService 位于应用层，负责异步会话、公开进度和 Token 汇总；它不进入 Harness，也没有运行
权限。桌面窗口只依赖 RunService 的可序列化契约。

内部 `OperationalContext` 继续使用细粒度的类型化状态作为唯一运行事实；模型侧投影稳定收敛为
`Runtime Envelope / Life / Profession / Work` 四个区域。Persona 不进入这四个区域，只在响应渲染
阶段生效。实验 treatment 通过可注入的投影扩展完成，核心 ContextManager 不识别 condition 名称。

## 一次运行

```text
create AgentProcess + RuntimeState
              |
              v
       Identity + Core Memory + Base Grants
              |
              v
 Agent -> ResourceBindingRequest | WorkAssignmentRequest | ActionRequest
              |                          |                       |
              v                          v                       v
 Policy/Profession binding    Assignment admission      Grant check + Capability
              |                          |                       |
              `--------------- structured Observation ---------->
              v
   User Response Rendering Context
   (Persona + communication memory + draft)
              |
              v
         User response
```

`AgentProcess` 是可销毁的推理进程，不是 Identity。`RuntimeState` 保存活动 Policy、Profession
和当前 Assignment 引用，不单独保存 WorkRole 或 Workspace binding。WorkRole 始终从活动
Assignment 解析。Effective Grants 每次由 `Base Grants ∪ active Assignment Grants` 派生；
Assignment 结束后，Workspace context、Archive scope、local skills 与临时 Grants 一并移除。
reasoning、respond 和资源申请属于 Agent 协议本身，不是外部副作用 Capability；默认 Base
Capability 集只允许无副作用能力，当前为 `echo`。

Persona 和沟通偏好从源头就不会进入 OperationalContext；最终表达的输出也不会重新进入
Action loop。因此 Persona 不能改变 Workflow、权限、Capability、Memory scope 或生命周期。

## 可观察运行轨迹

运行时通过不可变 RuntimeEvent 记录既有机制边界：

```text
agent_created
  -> context_built
  -> reasoning_started
  -> reasoning_completed      # 携带模型 API 返回的 Token usage
  -> resource_binding_requested
  -> resource_binding_completed
  -> workflow_requested
  -> workflow_admission_completed
  -> work_assignment_requested
  -> work_assignment_admission_completed
  -> capability_requested
  -> capability_completed     # 只有真正调用了 Capability 才出现
  -> action_completed
  -> work_assignment_ended      # 临时授权已撤销
  -> ...
```

未知、参数无效或未授权的动作仍会产生 `action_completed`，但不会伪造
`capability_completed`。Runtime 异常产生 `error`，同时保持原有 FAILED 状态和进程清理语义。

事件只包含关联 ID、状态、阶段和计数；不会复制任务文本、Persona、Memory 内容、Action 参数、
Capability 输出、异常消息或 API 配置。EventLogger 是诊断历史，不是权限依据或 Event Sourcing。
实时订阅者失败不会影响运行，订阅接口也不能修改 RuntimeState。

## 目录

```text
agent/                    Agent 协议、LLM 客户端与独立 Worker
ui/
  contracts.py            可序列化 RunSnapshot / ProgressEvent
  service.py              前端无关的异步 RunService
  desktop.py              本地桌面窗口适配器
harness/
  core.py                 纯 Harness API 门面
  runtime_engine.py       确定性运行协调与有界循环
  models.py               Process、RuntimeState、Assignment、Decision、Result
  capability.py           统一 Capability 与 AdmissionPolicy
  context_manager.py      Operational truth、四区模型投影与响应渲染上下文
  memory_manager.py       MemoryProvider 与两种实现
  workflow_*.py           有环 Workflow 加载/注册
  identity.py             持久 Identity 配置
  cognitive_policy.py     可组合的认知偏置预设
  profession.py           长期专业资源定义
  skill.py                声明式方法资源
  workspace.py            项目资源与检索配置
  work_role.py            Assignment 内的临时职责定义
  persona.py              不可变 Persona 配置
  events.py               RuntimeEvent 与有界 EventLogger
  inspector.py            只读 RuntimeSnapshot
capabilities.py           echo 与 filesystem.* 内置能力
config/persona.json       默认 Persona
config/identity.json      默认 Identity
config/cognitive_policies.json  16 个结构化 CognitivePolicy 预设
config/professions.json   专业知识、检索、方法与评价标准
config/skills.json        可复用声明式 Skill
config/workspace.json     当前项目资源与检索配置
config/work_roles.json    临时职责定义
memory/core_memory.json   Core Memory 引导数据
workflows/                Workflow JSON + Markdown
tests/
  unit/                   组件契约、Schema 与单对象不变量
  integration/            Cognitive Context / Work Runtime 子系统集成
  runtime/                完整 Harness/Runtime 架构故事
  regression/             历史回归与明确 deprecated/skip 规范
experiments/              V6 Policy / Profession 真实模型消融实验
runtime.py                依赖组装和 Python 入口
main.py                   桌面入口与可选 CLI 入口
sforge.pyw                Windows 无控制台桌面启动入口
SForge_V1_PLAN.md         V1 设计基线
SForge_V1_1_PLAN.md       V1.1 可观察运行时规范
SForge_V1_2_PLAN.md       V1.2 Action/Memory 经验闭环规范
SForge_V1_3_PLAN.md       V1.3 Agent 自主 Workflow 准入规范
v1_4_PLAN.md              V1.4 Cognition / Workspace 解耦规范
v1_5_PLAN.md              V1.5 Identity / WorkAssignment 规范
v1_6_PLAN.md              V1.6 CognitivePolicy / Profession / Workspace 规范
SForge_UI_PLAN.md         独立软件 UI 架构和接口规范
```

运行核心只有统一 Capability 和 MemoryProvider。Skill 是上下文中的方法知识，不是第二套
Tool，也没有执行器。

## 安装与 API 配置

需要 Python 3.10 或更新版本。在 PowerShell 中：

```powershell
cd "..\SForge"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

桌面窗口使用 Python 标准库 Tk。Windows 官方 Python 和常规 Conda Python 通常已包含；如果
当前发行版没有 Tk，仍可使用 `python main.py --cli`，或为该 Python 环境补装 Tk 支持。

如果使用已有 Conda 环境：

```powershell
conda activate sforge
python -m pip install -r requirements.txt
```

依赖必须安装到实际启动 `main.py` 的同一个环境；尤其需要 `python-dotenv` 才能自动读取
根目录 `.env`。

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

持久身份在 `config/identity.json` 中配置。Identity 不再内嵌 Personality、Profession、Role 或
Workspace：

```json
{
  "id": "ada",
  "display_name": "Ada",
  "owner_binding": "local_user",
  "created_at": "2026-01-01T00:00:00+00:00",
  "persona_reference": "persona_Ada@1.0",
  "default_cognitive_policy_id": null
}
```

V1.6 的资源配置分为四类：

- `config/cognitive_policies.json`：BasePolicy 与 I/E、S/N、T/F、J/P 参数增量，生成 16 个
  可替换预设。它们不是心理学判断，也不能改变合法 Memory scope。
- `config/professions.json`：专业知识引用、专业记忆标签、preferred skills、方法和评价标准；
  不含 Grants。
- `config/skills.json`：可复用的方法步骤。Skill 不执行动作，也不映射成 Capability。
- `config/workspace.json`：项目档案来源、检索偏好和 local skills。未建立 Assignment 时只公开
  Workspace id 与描述，不装载这些内部资源。

`config/work_roles.json` 只描述当前 Workspace 中的临时职责和完成标准。Role 没有独立绑定 API，
也不包含专业记忆、Skill 或 Capability；它只能作为 WorkAssignment 的组成部分出现。

## 启动

```powershell
python main.py
```

默认打开 SForge 桌面窗口。窗口提供对话、Workflow 建议、当前运行阶段、WorkRole、Workflow
状态、正在请求的 Capability、实时活动记录和累计 Token 用量。Token 数量来自模型 API 的 usage 字段，
不会用本地估算冒充真实数据。

界面中的 Workflow 选择只是给 Agent 的任务提示。新 Agent 仍从 Core Memory 和基础能力启动；
只有 Agent 自己返回 `WorkflowRequest` 并通过 RuntimeEngine 校验后，界面才显示真正激活的
Workflow 与状态。

Windows 下也可以使用当前环境的无控制台解释器启动，后续打包独立程序时复用同一入口：

```powershell
pythonw sforge.pyw
```

每次提交仍然创建独立 AgentProcess，运行结束后销毁推理进程。界面持有的是 RunSnapshot，
不是 Agent、Harness 或权限资源。

保留轻量终端适配器：

```powershell
python main.py --cli
```

在终端模式启用开发者 Runtime Inspector：

```powershell
python main.py --cli --inspect
```

`python main.py --inspect` 仍作为兼容快捷方式进入相同的终端调试模式。
它会在每次成功或失败的运行后输出 JSON 快照，包括 Agent、RuntimeState、Workflow、当前
Memory scope、可见 Capability 和最近事件。Inspector 不会增加权限、额外调用模型或延长
Agent 进程生命周期。Memory 内容只会出现在这个显式的本地调试视图中，不会进入事件。

面向其他前端的应用接口：

```python
from ui import RunService

service = RunService()
unsubscribe = service.subscribe(lambda event: print(event.as_dict()))
try:
    run_id = service.start("给我一个科幻短篇梗概", "novel_writing")
    result = service.wait(run_id)
    print(result.answer, result.token_usage.as_dict())
finally:
    unsubscribe()
    service.close()
```

`RunService`、`ProgressEvent` 和 `RunSnapshot` 不依赖 Tk，可以在未来直接包装为 WebSocket、
HTTP、WebView 或独立客户端进程接口。

也可以作为 Python API 使用：

```python
from runtime import AgentApplication

app = AgentApplication()
try:
    answer = app.handle("把 README.md 的第一段概括成一句话")
    novel_answer = app.handle("给我一个科幻短篇梗概", workflow_id="novel_writing")
    snapshot = app.inspect().as_dict()
finally:
    app.close()
```

底层 Harness API 保持小而显式：

```python
from harness.models import (
    ActionRequest,
    ResourceBindingRequest,
    TaskSpec,
    WorkAssignmentRequest,
    WorkflowRequest,
)
from runtime import create_runtime

runtime = create_runtime()
process = runtime.create_agent(TaskSpec("echo test"))
runtime.request_binding(
    process.id,
    ResourceBindingRequest(
        "profession", "activate", "software_engineering"
    ),
)
assignment = runtime.request_work_assignment(
    process.id,
    WorkAssignmentRequest(
        "developer",
        workflow_id="general_task",
        requested_capabilities=("filesystem.read",),
    ),
)
result = runtime.execute_action(
    process.id,
    ActionRequest("echo", {"text": "hello"}),
)
runtime.end_work_assignment(process.id)
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

一个 Workflow 由厚状态节点和声明式有向边组成。节点就是认知环境，不再平行维护另一套
`CognitiveMode`：

```json
{
  "id": "novel_writing",
  "description": "Long-form fiction creation workflow",
  "initial_state": "creation",
  "states": {
    "creation": {
      "context": "Explore worldbuilding, characters and narrative structure.",
      "goal": "Create a coherent draft.",
      "allowed_capabilities": [
        "filesystem.read",
        "filesystem.write",
        "filesystem.list"
      ],
      "memory_scope": ["workflow", "creative_memory"],
      "memory_write_scope": "creative_memory",
      "memory_hints": ["idea", "character", "theme"],
      "evaluation": {
        "criteria": ["originality", "coherence"]
      }
    },
    "revision": {
      "context": "Improve structure and consistency.",
      "goal": "Produce a stronger revision.",
      "allowed_capabilities": ["filesystem.read", "filesystem.write"],
      "memory_scope": ["workflow", "draft_memory"],
      "memory_write_scope": "draft_memory",
      "evaluation": {
        "criteria": ["plot_quality", "character_consistency"]
      }
    }
  },
  "transitions": {
    "creation": [
      {"condition": "draft_completed", "target": "revision"}
    ],
    "revision": [
      {"condition": "changes_required", "target": "creation"}
    ]
  }
}
```

Workflow 可以有环。Agent 判断何时申请初始准入，以及某个声明条件何时成立；Harness 只验证
当前节点是否存在精确的目标边，然后原子切换 Context、Memory scope 和 Capability scope。
Harness 不判断草稿是否完成，也不自动推进节点。状态中禁止固定 Agent 角色或 Persona。

首次准入只能进入 `initial_state`。同一 Run 内后续请求只能沿当前 Workflow 的声明边迁移；非法
节点、非法边或跨 Workflow 切换返回结构化 `WorkflowAdmission(status="rejected")`，运行状态不变。

## CognitivePolicy、Profession、Workspace 与 WorkAssignment

Policy 与 Profession 使用同一个通用资源绑定接口，且没有固定加载顺序：

```python
runtime.request_binding(
    process.id,
    ResourceBindingRequest("cognitive_policy", "activate", "INTJ"),
)
runtime.request_binding(
    process.id,
    ResourceBindingRequest("profession", "activate", "software_engineering"),
)
```

`activate/deactivate` 只更新资源引用和合法记忆读取视图，不更新 Capability。CognitivePolicy 与
Profession 可在 Assignment 前后独立绑定或解绑，Assignment 结束时也不会自动清除它们。

Agent 通过 `WorkAssignmentRequest` 提出完整工作关系：

```text
WorkAssignment = Workspace + Role + Task + Workflow? + temporary Grants
```

Runtime 校验当前 Workspace、Task、Role、Workflow 初始状态和 requested capabilities，然后原子
创建或替换当前 Process 唯一的 active Assignment。`WorkAssignment.grants` 只保存本次关系的临时
Grants；`RuntimeState.allowed_capabilities` 由 Base Grants 与当前 Assignment Grants 动态合并。
Workflow 状态迁移还要与 Assignment Grants 取交集，不能借迁移扩大授权。

`create_runtime(workspace_root=..., workspace_id=...)` 用 `workspace_root` 约束文件 Capability，
用 `workspace_id` 标识 Workspace。没有单独的 `WorkspaceBinding` 或 `active_workspace` 状态：未建立
Assignment 时只能发现 id/description；建立后才装载项目上下文、Archive scope 与 local skills。

运行前只有 Core Memory 和无外部副作用的基础 `echo`。Profession 激活后可增加匹配其检索配置的
`identity:<identity_id>` 专业经验；Assignment 激活后再增加：

- `workspace:<workspace_id>`：项目事实、Assignment 历史与动作证据；
- 当前 Workflow State 声明的 scopes。

上下文装配顺序固定为：先根据当前资源关系确定合法 scopes，再进行 Profession/Workspace/Task
相关性处理，最后由 CognitivePolicy 对仍然合法的候选记录排序。Policy 永远看不到未授权 scope。

`record_work_experience()` 要求至少提供 objective outcome 或 external feedback，避免把 Agent 的
自我评价直接固化为事实。Workspace Archive 记录“发生了什么”，Identity Memory 记录“学到了
什么”，二者仍使用同一个 MemoryProvider 和关系 metadata，不创建角色数据库或 Workspace
Memory 数据库。

## Memory 与 Capability

MemoryProvider 的最小契约是 `write/retrieve/get/close`。默认运行使用
`SQLiteMemoryProvider`，测试使用 `InMemoryMemoryProvider`。持久数据写入
`memory/runtime.sqlite3`。

SForge 只有一种 Capability 抽象。内置能力：

- `echo`：返回文本。
- `filesystem.read`：读取工作区内 UTF-8 文本。
- `filesystem.write`：写入工作区内 UTF-8 文本。
- `filesystem.list`：按稳定顺序列出目录，返回相对路径和条目类型。

从 V1.1 升级时，需要把 Workflow 或外部调用中的 `read_text`、`write_text` 分别改为
`filesystem.read`、`filesystem.write`。V1.2 不注册旧名称别名，以免同一种行为占用两套
Capability ID。

文件路径会被限制在工作区内。未知、参数无效或未授权的请求返回 `rejected`；执行异常返回
`failed`；成功返回 `success`。这些结果都作为结构化 Observation 回到同一 Agent。
显式的空 Capability 集合表示零权限，不会退化成“显示全部能力”。

Bootstrap 阶段只读取 Core Memory，运行时 Action/Final 记录不会反写并污染公共 Core。激活
Profession 后可检索与其 tags 匹配的 Identity 专业经验；成功建立 WorkAssignment 后再读取
Workspace Archive 与当前 Workflow State 声明的 scopes。写入范围由 Assignment/Workflow 的
`memory_write_scope` 决定，单独发现 Workspace 不会获得项目写入范围。
`task` 解析为 `task:<task_id>`，`workflow` 解析为稳定的 `workflow:<workflow_id>`，自定义范围会
命名到 `workflow:<workflow_id>:<scope>` 下。因此旧 Agent 结束并销毁推理进程后，新 Agent 可以
按相同 Workflow 范围获得连续记忆，而不会继承旧进程身份。记忆查询统一使用
`retrieve(scope=..., query=...)`，不另建同义的 Search 或向量记忆系统。

## V1.6 真实模型消融实验

`experiments/v6_cognitive_profession_ablation.py` 直接复用现有 Runtime、Worker、Memory、Skill、
Assignment 和 Capability/Admission 主干，对同一模型与任务运行 Base、Profession、两组 Policy
及组合条件。实验 Workspace 只授予 `filesystem.read`，不会
向真实项目写入文件。

直接复用项目已有 `sforge` 环境，不需要也不应为实验新建虚拟环境：

```powershell
conda activate sforge
python experiments/v6_cognitive_profession_ablation.py --dry-run
python experiments/v6_cognitive_profession_ablation.py --runs 3
python experiments/v6_cognitive_profession_ablation.py --dry-run --policy-transmission `
  --task mechanism_policy_cache --task mechanism_policy_dependency
python experiments/v6_cognitive_profession_ablation.py --dry-run --strength-sweep `
  --strengths 0,0.25,0.5,0.75,1 --task mechanism_policy_cache
python experiments/v6_cognitive_profession_ablation.py --dry-run `
  --causal-decomposition --task causal_cache_invalidation
python experiments/v6_cognitive_profession_ablation.py --smoke --json-mode `
  --output experiments/results/v6_policy_causal_round4_smoke.jsonl
```

`--dry-run` 不调用模型，直接打印最终 RuntimeState、活动资源、检索结果、完整 structured context
和模型配置。真实运行从现有 `.env` / 环境变量读取 DeepSeek 配置；每个 run 写一条 JSONL，并在
相邻位置生成 summary CSV。默认输出目录是 `experiments/results/`。本地上下文 token 数在没有
专用 tokenizer 时明确标记为估算值；模型输入/输出 token 始终使用 API usage 返回值。

完整参数、条件、Policy transmission、strength sweep、Round 4 Policy Causal
Decomposition 和输出字段见 `experiments/README.md`。Round 4 使用无 Profession 的
`generalist` 工作身份，并将 Runtime 控制元数据与模型可见的 Policy 投影严格分离。

## 测试

测试分为四层：组件契约、子系统集成、Runtime 架构场景、真实模型实验。
前三层全部 deterministic，不访问网络，也不依赖真实模型：

```powershell
python -B -m pytest tests
python -B -W error::ResourceWarning -m unittest discover -s tests -t . -v
```

`tests/unit/` 验证单个资源、解析器、Schema 和局部不变量；`tests/integration/`
验证 CognitivePolicy + Profession + Memory + Context，以及 Assignment + Workspace +
Admission + Capability 的状态传播；`tests/runtime/` 只通过公开 Harness/应用边界验证
完整架构故事；`tests/regression/deprecated/` 保留 V1.4/V1.5 的明确 skip 语义。
具体迁移对照见 `tests/TEST_MIGRATION.md`。

第四层位于 `experiments/`，不纳入 pytest 普通发现。普通测试只校验实验脚本的
dry-run、条件构造、输出 Schema 和 deterministic evaluator，不调用 DeepSeek。

覆盖内容包括：

- 生命周期、进程隔离与非法状态转换；
- RuntimeState 与 AgentProcess 分离；
- Capability 注册、Schema、准入、失败映射和路径逃逸；
- MemoryProvider 契约与 scope；
- SQLite MemoryProvider 关闭并重建后的持久性；
- Agent 请求通用 Workflow 后写入、读取、最终回答的端到端闭环；
- 文件系统 read/write/list 的稳定输出和路径逃逸保护；
- 小说 Workflow 使用相同 Agent 抽象；
- Workflow 初始准入、非法边原子拒绝与有环迁移；
- Identity 跨 Process 连续性与 Persona 单向隔离；
- CognitivePolicy/Profession 独立绑定、任意加载顺序和不赋权边界；
- Profession 专业记忆筛选、Workspace Assignment 检索和 Skill 多来源复用；
- 同一合法候选集的 Policy 排序差异，以及 Memory scope 高于 Policy 的负向验证；
- WorkRole 只存在于 Assignment，进入后授权、结束后撤权和越权原子拒绝；
- Workspace Archive 项目隔离与有证据的专业经验跨 Workspace 复用；
- 两个独立 Agent 通过 Workflow Memory 完成跨运行经验连续性；
- Persona/沟通记忆与 OperationalContext 的单向隔离；
- 有界循环及清理行为。
- RuntimeEvent 不可变性、顺序、容量、过滤和关联 ID；
- 模型 Token usage 从 API、Worker 到运行会话的完整传递；
- EventLogger 实时订阅、订阅者隔离和取消订阅；
- RunService 的异步状态、Capability 进度、成功与失败终态；
- 默认桌面入口与独立 CLI/Inspector 入口；
- 事件载荷的隐私边界、Capability 执行真实性和错误追踪；
- Inspector 对运行中及终态 Agent 的只读快照。
- V6 消融实验的四组条件、无网络 dry-run、输出 Schema 与确定性评分器。

真实模型验证通过 `python main.py` 单独执行，以免常规测试依赖 API 或网络。

## V1.6 明确不做

- 多 Agent、父子 Agent、角色 Agent 或 Agent 类型树；
- Workflow 自动执行器、Harness 语义判断和内置业务规划；
- 固定角色、角色 Agent 或嵌入 Workflow 的第二套 CognitiveModeDefinition；
- 独立 `WorkspaceBinding` / `active_workspace` 两阶段状态；
- WorkRole 作为独立 runtime state，或持有 Profession、权限和 Memory scope；
- CognitivePolicy/Profession/Workspace/Skill 参与 Capability 授权；
- Personality 固定绑定 Identity，或 Persona 参与运行控制；
- 完整 Memory Graph、自动经验晋升或自动人格演化；
- Memory Graph、MemoryView 或新的 Workspace 管理子系统；
- Skill 执行器、Skill 到 Capability 的隐式映射或 Skill/Tool 双执行体系；
- Scheduler、Event Sourcing、事件回放、分布式追踪、Checkpoint/Restore；
- Persona Registry 或完整 Presentation Engine；
- Persona 驱动任何运行权限与控制。
