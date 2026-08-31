# Novel Writing Workflow


## Purpose

Support long-form fiction creation.

This workflow does not create multiple agents.

A single AgentProcess requests cognitive-environment changes through workflow states.


## State Space


### Creation State

Focus:

- imagination
- world building
- character creation
- narrative planning


Memory:

- creative ideas
- previous concepts
- unfinished drafts



### Revision State

Focus:

- identify weaknesses
- improve structure
- maintain consistency


Memory:

- draft versions
- previous revisions
- constraints



### Review State

Focus:

- external perspective
- emotional response
- accessibility


Memory:

- feedback
- preferences
- audience expectation



## Principles

Agent is responsible for reasoning.

Workflow declares a directed cyclic graph. In particular, review can return to revision;
this is intentional and must not be flattened into a DAG or automatic pipeline.

Harness only validates and mounts:

- state transition
- capability access
- memory scope
- execution permission


The workflow does not contain fixed Agent roles or an automatic writing pipeline.

The Agent decides when to request admission and when a declared transition condition is
appropriate. Harness verifies the requested node and edge, but never evaluates the story.
