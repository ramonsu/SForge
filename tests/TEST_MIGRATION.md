# V1.6 test migration map

No passing behavior was silently discarded. Files were classified by the
largest boundary they exercise; shared Runtime composition moved out of test
modules so tests no longer import other tests as fixtures.

| Previous file | New layer / file | Responsibility |
| --- | --- | --- |
| `test_models.py` | `unit/test_models.py` | typed model contracts |
| `test_events.py` | `unit/test_events.py` | immutable event/logger contracts |
| `test_llm_client.py` | `unit/test_llm_client.py` | provider adapter configuration/usage |
| `test_memory_and_tools.py` | `unit/test_memory_and_capability_contracts.py` | provider and registry contracts |
| `test_workflow_manager.py` | `unit/test_workflow_contracts.py` | declarative Workflow parsing/registry |
| `test_harness.py` | `integration/test_harness_boundaries.py` | lifecycle and action boundary composition |
| `test_persona.py` | `integration/test_persona_context.py` | operational/presentation separation |
| `test_workflow_admission.py` | `integration/test_workflow_admission.py` | Workflow state propagation/admission |
| `test_worker_protocol.py` | `integration/test_worker_protocol.py` | Worker/supervisor protocol |
| `test_ui_service.py` | `integration/test_ui_service.py` | frontend-neutral service propagation |
| `test_v1_6_architecture.py` | `integration/test_v6_context_and_work_runtime.py` | V6 resource/work composition baseline |
| `test_v1_6_experiment.py` | `integration/test_experiment_boundary.py` | network-free experiment boundary only |
| `test_novel_workflow.py` | `runtime/test_agent_runtime_scenarios.py` | full Agent loop scenarios |
| `test_runtime_trace.py` | `regression/test_runtime_trace_regressions.py` | confirmed trace/inspector regressions |
| `test_cli.py` | `runtime/test_application_entrypoints.py` | application entrypoint story |
| `test_v1_5_architecture.py` | `regression/deprecated/test_v1_5_architecture.py` | explicitly skipped V1.5 contract |

`tests/fakes.py` and the composition helpers formerly embedded in
`test_harness.py` were merged into `tests/support/`. This is the only removed
duplicate implementation; it was not a test case.

The already-skipped V1.4 `CognitiveProfile` compatibility test and its isolated
loader/config were removed during the Thin Core convergence. They were referenced
only by each other and had been superseded by the V6 `CognitivePolicy` contract;
no passing behavior or active runtime path was removed.

New focused coverage:

- `unit/test_v6_resource_contracts.py`: Policy, Profession, Skill,
  WorkAssignment, Grant/Admission, and Workspace contracts.
- `integration/test_cognitive_context_module.py`: candidate visibility,
  ranking, and context refresh.
- `integration/test_work_runtime_module.py`: Assignment-driven Workspace,
  skills, grants, and Admission.
- `runtime/test_v6_invariant_scenarios.py`: four public-boundary V6 stories.
- `integration/test_thin_core_context_projection.py`: Agent ownership, four-region
  model projection, Assignment cleanup, and experiment/core source boundaries.
