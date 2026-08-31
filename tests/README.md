# SForge deterministic test layers

The ordinary suite never calls DeepSeek or the network.

- `unit/`: one component, parser, schema, contract, or local invariant.
- `integration/`: deterministic composition of the Cognitive Context and Work
  Runtime subsystems, plus adapters around those subsystems.
- `runtime/`: complete SForge stories through the public Harness/Application
  boundary. These tests assert architecture-level state consistency and do not
  call private helpers.
- `regression/`: confirmed historical behavior. The superseded V1.5
  specification remains under `regression/deprecated/` with an explicit skip
  reason. The isolated V1.4 CognitiveProfile compatibility shim and its matching
  skipped test were removed after confirming that no V6 runtime path used them.
- `support/`: small fake reasoning process and a real Runtime composition
  factory. It does not mock RuntimeEngine, ContextManager, Admission,
  Capability, Memory, or Assignment.

The real-model effect evaluation remains under `experiments/` and is not part
of ordinary discovery. Tests may validate the experiment CLI's network-free
contract and deterministic evaluator, but they never run a real model.

`integration/test_thin_core_context_projection.py` protects the convergence
boundary: Agent ownership stays minimal, model-facing operational context keeps
exactly four top-level regions, experiment treatment names stay outside the core,
and Assignment cleanup does not detach long-lived Policy or Profession resources.

Run the environment-independent suite with the project’s existing test runner:

```powershell
python -B -W error::ResourceWarning -m unittest discover -s tests -t . -v
```

If `pytest` is installed in the active environment, the same package tree is
also collectible with `python -m pytest tests`.
