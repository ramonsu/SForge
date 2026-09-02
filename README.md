# SForge

> A lightweight stateful runtime for long-running LLM agents.

SForge 是一个面向长期运行 LLM Agent 的轻量级 Runtime / Harness 原型。

它解决的核心问题是：

> **AgentProcess 可以随时结束或重建，但 Identity、Memory、Work 和 Authority 不应该随进程一起消失。**

SForge 将长期状态保存在 Runtime 中，将 AgentProcess 保持为短生命周期的推理执行器，并在每次运行时重新构造当前真正需要的模型上下文。

---

## What I Built

* **Persistent Identity + Disposable AgentProcess**
  AgentProcess 可以被销毁和重新创建，而长期 Identity 与工作状态继续存在。

* **Work Continuity**
  Profession、WorkAssignment、Workspace、Role、Task 和 Workflow 可以跨 AgentProcess 延续。

* **Bounded Context Construction**
  长期 Memory 可以持续积累，但模型侧 Context 经过检索、去重和预算控制，不会无限增长。

* **Declarative Resources**
  Workspace、Profession、Workflow、Skill、CognitivePolicy 等通过配置定义，Runtime 负责管理它们之间的动态关系。

* **Explicit Capability Admission**
  外部操作必须经过 Runtime 权限校验，资源本身不能直接获得或扩大执行权限。

---

## Results

### Context Economy

长期运行后，历史文件读取结果曾导致 Context 大量重复装载。

优化后：

| Scenario           |                 Before |  After | Reduction |
| ------------------ | ---------------------: | -----: | --------: |
| Synthetic 50k read | 28.6k estimated tokens |   3.8k | **86.7%** |
| Real model run     |      ~46k input tokens | ~10.4k |  **~77%** |

完整历史证据仍然保存在 MemoryProvider 中，但模型只获得当前任务真正需要的有界投影。

### Testing

```text
156 passed
12 skipped
```

核心测试全部 deterministic，可在不调用真实模型 API 的情况下运行。

---

## Core Idea

```text
              Persistent Runtime
        ┌─────────────────────────┐
        │ Identity                │
        │ Memory                  │
        │ Profession              │
        │ WorkAssignment          │
        │ Workflow State          │
        └────────────┬────────────┘
                     │
               Context Builder
                     │
                     ▼
               AgentProcess #1
                     │
                     ×
                     │
                     ▼
               AgentProcess #2
                     │
               continue work
```

SForge 的基本原则：

```text
Resources describe possibilities.
Runtime owns relationships.
AgentProcess consumes projections.
```

以及：

```text
Persist rich state.
Project small context.
```

---

## Architecture

SForge 将系统分成三个主要层次：

```text
Declarative Resources
Workspace / Profession / Workflow / Skill / Policy
                    │
                    ▼
              Runtime Kernel
       lifecycle / state / binding
       admission / context building
                    │
                    ▼
               AgentProcess
             Context → LLM
              → Decision
```

资源描述“可以存在什么”。

Runtime 管理“当前实际是什么”。

AgentProcess 只负责读取当前 Context，并返回结构化决策。

---

## Runtime Model

### AgentProcess

AgentProcess 是短生命周期的推理进程。

它不拥有长期 Memory、Workspace 或权限状态。

```text
Context
   ↓
LLM reasoning
   ↓
Structured Decision
```

---

### WorkAssignment

当前工作关系由 WorkAssignment 显式表示：

```text
WorkAssignment
=
Workspace
+ Role
+ Task
+ optional Workflow
+ temporary Grants
```

因此：

```text
Process lifetime < WorkAssignment lifetime
```

AgentProcess 结束并不天然意味着工作结束。

---

### Context Economy

模型侧 Context 固定为四个区域：

```text
Runtime Envelope
Life
Profession
Work
```

候选信息经过：

```text
legal scope
    ↓
relevance
    ↓
deduplication
    ↓
budget
    ↓
projection
```

完整事实可以长期保存，但不会默认全文回放给模型。

---

## Capability Boundary

SForge 区分：

```text
Skill       = method knowledge
Capability  = executable operation
Grant       = execution authority
```

所有外部动作统一经过：

```text
ActionRequest
      ↓
Runtime Admission
      ↓
Capability
      ↓
ActionResult
```

当前内置：

* `echo`
* `filesystem.read`
* `filesystem.write`
* `filesystem.list`

文件操作被限制在 Workspace 范围内。

---

## Quick Start

要求 Python 3.10+。

```powershell
git clone <repository-url>
cd SForge

conda activate sforge
python -m pip install -r requirements.txt
```

复制环境配置：

```powershell
Copy-Item .env.example .env
```

启动 CLI：

```powershell
python main.py --cli
```

启用 Runtime Inspector：

```powershell
python main.py --cli --inspect
```

运行测试：

```powershell
pytest
```

普通测试不会调用 DeepSeek API。

---

## Project Structure

```text
agent/        LLM interface and reasoning process
harness/      stateful runtime kernel
config/       declarative resources
workflows/    workflow definitions
tests/        deterministic architecture tests
experiments/  real-model evaluations
ui/           desktop / RunService interface
```

---

## Current Scope

SForge 当前重点验证：

* long-running single-identity Agent runtime
* disposable AgentProcess
* persistent work continuity
* bounded context construction
* explicit capability admission
* declarative resource composition

暂不重点实现：

* distributed execution
* full multi-agent orchestration
* full Runtime checkpoint / restore
* Event Sourcing

---

## Design Notes

更详细的设计说明可以拆分到：

```text
docs/
├── architecture.md
├── runtime-continuity.md
├── context-economy.md
├── resources.md
└── testing.md
```

---

## Motivation

SForge 想回答的不是“如何再封装一个 LLM API”，而是长期 Agent 系统中的几个 Runtime 问题：

```text
What survives when an AgentProcess dies?

Where does long-term state live?

How is work authority represented?

How can memory grow without context growing forever?
```

SForge 当前的答案是：

> **Persistence lives in Runtime.
> Cognition is instantiated through Context.
> Reasoning is delegated to the model.**
