# V6 experiment task fixtures

The JSON files in this directory are deterministic fixtures, not prompts for an
external judge model.

- `suite: capability` measures ordinary software-engineering task correctness.
- `suite: mechanism` measures a causal trace: mounted resource, retrieval rank,
  constructed context, cited reasoning evidence, and answer focus.
- `required_findings` and `forbidden_claims` are normalized and evaluated with
  explainable evidence. Selected findings also use regex, Python AST, or
  structure-specific checks.
- `expected_action_type` defaults to `final`. These fixtures do not authorize a
  tool call; any real action still uses the existing SForge Admission boundary.

The four mechanism tasks intentionally separate two questions:

1. Profession fixtures (`SE-17`, `SE-23`) are inaccessible without the
   `software_engineering` Profession and test professional evidence use.
2. Policy fixtures pair risk/precedent evidence with novel/exploratory evidence.
   INTJ and ENFP are expected to rank the legal pairs in opposite directions;
   neither direction is labelled universally better.
