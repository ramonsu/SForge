# SForge V6 CognitivePolicy / Profession Ablation

This is an opt-in real-model experiment around the existing SForge runtime. It
does not implement a second Agent loop. Every condition uses the same base
model, task, experiment Workspace, reviewer WorkRole, `general_task` Workflow,
and `filesystem.read` Grant. Only CognitivePolicy and Profession bindings vary.

Experiment-only causal treatments are implemented as an injected model-projection
override in this package. The core `ContextManager` always exposes the stable
`Runtime Envelope / Life / Profession / Work` projection and does not know any
experiment condition name. The override changes only model-facing evidence and
guidance; it cannot change RuntimeState, Assignment grants, Admission, or Capability
execution.

## Conditions

| id | CognitivePolicy | Profession |
| --- | --- | --- |
| `base` | none | none |
| `profession_only` | none | `software_engineering` |
| `policy_a_only` | `INTJ` | none |
| `profession_and_policy_a` | `INTJ` | `software_engineering` |
| `policy_b_only` | `ENFP` | none |

Profession can change legal professional-memory retrieval and Skill sources.
CognitivePolicy can change the order of already legal memory candidates. Neither
changes Capability grants.

Policy-mechanism tasks add a stricter relevance boundary: legal scope is checked
first, then explicit `retrieval_task_id(s)` / `fixture_scope`, then
Profession/Workspace relevance,
then CognitivePolicy ranking, and finally the context budget. Policy never ranks
an unrelated fixture into the task.

## Environment

Use the existing project environment; do not create another environment:

```powershell
conda activate sforge
cd D:\Desktop\创作\SForge
```

The current requirements are sufficient (`openai` and `python-dotenv`). The
experiment deliberately does not require `tiktoken`: it labels local context
token counts as estimates and records provider-returned prompt/completion usage
as the exact model usage.

Normal runs read the existing `.env` or process environment:

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

The CLI sets `DEEPSEEK_TEMPERATURE` and `DEEPSEEK_MAX_TOKENS` for the existing
Worker. `DEEPSEEK_SEED` is optional because not every OpenAI-compatible endpoint
supports it; use `--seed 42` only when the configured endpoint does.

## Run

Network-free inspection of all five conditions and all built-in tasks:

```powershell
python experiments/v6_cognitive_profession_ablation.py --dry-run
```

Limit dry-run output while inspecting one task:

```powershell
python experiments/v6_cognitive_profession_ablation.py --dry-run --task mutable_default_argument
```

Inspect the controlled Policy A/B transmission pair. Profession, Workspace,
Assignment, task and memory candidates stay fixed:

```powershell
python experiments/v6_cognitive_profession_ablation.py --dry-run `
  --policy-transmission `
  --task mechanism_policy_cache `
  --task mechanism_policy_dependency
```

Inspect the continuous bias-strength sweep:

```powershell
python experiments/v6_cognitive_profession_ablation.py --dry-run `
  --strength-sweep `
  --strengths 0,0.25,0.5,0.75,1 `
  --task mechanism_policy_cache `
  --task mechanism_policy_dependency
```

Real DeepSeek run, three replicates per task and condition:

```powershell
python experiments/v6_cognitive_profession_ablation.py --runs 3
```

Explicit controls and output path:

```powershell
python experiments/v6_cognitive_profession_ablation.py `
  --runs 3 `
  --model deepseek-chat `
  --policy INTJ `
  --policy-b ENFP `
  --profession software_engineering `
  --temperature 0.1 `
  --max-tokens 2048 `
  --output experiments/results/v6_ablation.jsonl
```

There are 18 deterministic-friendly tasks: 14 ordinary capability benchmarks
and 4 mechanism-validation tasks. The latter independently trace Profession or
CognitivePolicy through retrieval order, constructed context, cited reasoning
evidence, and answer focus. These suites are never merged into one score.
With the default 18 tasks and five conditions, `--runs 3` writes 270 records.

Each record separates `runtime_completion_success` from `protocol_success`, and
stores strict/fallback Decision parsing, raw Decision, Decision semantics,
response-preservation metrics, actual policy parameters, ranked resources,
token usage, latency, Capability/Admission observations, and failures. A sibling
`*.summary.csv` has separate architecture, mechanism, protocol, capability, and
response-rendering columns. Policy tasks additionally store legal/task-relevant/
ranked memory ids, score components, structured `primary_evidence_id`,
`secondary_evidence_ids`, `final_choice`, and separate ranking/primary/final
transmission rates. API keys are never written to either file.

If the configured DeepSeek-compatible endpoint supports JSON-object mode, it
can be enabled for Decision calls only:

```powershell
python experiments/v6_cognitive_profession_ablation.py --runs 3 --json-mode
```

Without this option SForge still performs strict parsing plus at most one bounded,
observable repair for a literal control character or one missing final object
brace. Recovery never counts as protocol success; unrepaired output remains a
hard failure.

The experiment Workspace is `experiments/workspace/`. Only `filesystem.read` is
granted; all actions still flow through the existing
`ActionRequest -> Admission -> Capability -> ActionResult` path.

## Round 4: Policy Causal Decomposition

Round 4 separates Runtime control metadata from the context visible to the
model. Runtime traces still retain the active policy id, raw/effective
parameters and strength, while the model receives only the treatment declared
by the selected condition. All conditions use the neutral `generalist` role,
no Profession, the same read-only Assignment, and exactly two task-scoped
evidence records.

| condition | evidence order | explicit priority | reasoning guidance |
| --- | --- | --- | --- |
| `causal_neutral` | counterbalanced fixture order | no | no |
| `causal_order_only` | Runtime policy order | no | no |
| `causal_explicit_rank` | Runtime policy order | rank 1/2 and high/low | no |
| `causal_reasoning_only` | counterbalanced fixture order | no | neutral operational guidance |
| `causal_full` | Runtime policy order | no | neutral operational guidance |

No model-facing condition contains `INTJ`, `ENFP`, `CognitivePolicy`, raw
weights, or policy strength. Explicit priority is described as attention order,
not confidence, correctness, or truth. The continuous strength sweep is paused
for this round because an unchanged two-item order would not be a distinct
model-visible treatment.

Inspect one paired fixture across all five conditions without an API call:

```powershell
python experiments/v6_cognitive_profession_ablation.py --dry-run `
  --causal-decomposition `
  --task causal_cache_invalidation
```

Before the full run, use the two bounded JSON-mode smoke checks. The first runs
one task under `causal_neutral`; the second runs that same task once under all
five conditions. A smoke command exits non-zero unless the provider accepts the
request, a structured Decision and `primary_evidence_id` are received,
Presentation completes, and provider token usage is non-zero.

```powershell
python experiments/v6_cognitive_profession_ablation.py `
  --smoke `
  --json-mode `
  --output experiments/results/v6_policy_causal_round4_smoke.jsonl

python experiments/v6_cognitive_profession_ablation.py `
  --smoke-all-conditions `
  --json-mode `
  --output experiments/results/v6_policy_causal_round4_smoke_all.jsonl
```

Run the full eight-pair Round 4 experiment locally, with three repeats per
condition (120 records):

```powershell
python experiments/v6_cognitive_profession_ablation.py `
  --causal-decomposition `
  --runs 3 `
  --model deepseek-chat `
  --temperature 0.1 `
  --max-tokens 2048 `
  --json-mode `
  --output experiments/results/v6_policy_causal_round4.jsonl
```

The sibling summary CSV reports risk/preferred-evidence selection rates and
deltas from the neutral baseline per evidence pair and across all pairs. The
primary observation is the formal `primary_evidence_id`; prose mentions and
secondary evidence do not change it. API request success, Decision receipt,
protocol compliance among received Decisions, Runtime completion, and
Presentation completion are reported with separate denominators. When no
Decision is received, causal metrics remain blank rather than being counted as
zero effect.

## CognitivePolicy V6 Final Validation

The final micro-mechanism round replaces the saturated Round 4 fixtures with
16 balanced boundary decisions under `experiments/tasks/policy_final/`. Each
pair contains one risk/precedent/verification option and one
exploration/novelty/flexibility option of similar specificity and length. Risk
versus exploration target direction, A/B label, neutral evidence order and
preferred-first position are each balanced 8:8.

The default final treatments are only `final_neutral` and
`final_reasoning_only`. Both receive the same counterbalanced evidence order.
Reasoning-only additionally receives operational guidance compiled by the
active `CognitivePolicy`; model-facing context never contains the preset id,
raw parameters, policy strength or evidence ids in the guidance. Add
`--include-explicit-rank` only when the weak Round 4 reference is useful.

Inspect the bounded final setup locally without calling a provider:

```powershell
python experiments/v6_cognitive_profession_ablation.py --dry-run --final-validation --smoke --json-mode
```

Run one real-model smoke after local checks:

```powershell
python experiments/v6_cognitive_profession_ablation.py --final-validation --smoke --json-mode --output experiments/results/v6_policy_final_smoke.jsonl
```

Run all 16 pairs, both conditions and three repeats (96 records):

```powershell
python experiments/v6_cognitive_profession_ablation.py --final-validation --runs 3 --model deepseek-chat --temperature 0.1 --max-tokens 2048 --json-mode --output experiments/results/v6_policy_final_validation.jsonl
```

The summary CSV reports the neutral baseline preference, policy-preferred rate,
delta from Neutral, paired switches toward and away from the target direction,
task- and direction-level effects, and token overhead. Freeze the reasoning
projection only when the shift is consistently aligned in both risk and
exploration directions, distributed across fixtures, and operationally cheap.
Otherwise freeze V6 as retrieval/context bias only; do not infer reliable
control over the final reasoning direction.
