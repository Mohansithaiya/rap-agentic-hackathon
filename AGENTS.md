# RAP Hackathon Project

## Goal

Build a reliable Agentic AI system for the RAP AI/ML Hackathon.

## Core Architecture

USER
  ?
AGENT
  ?
HARNESS
  ?
TOOLS
  ?
RESULT
  ?
VERIFY
  ?
FINAL RESPONSE

## Engineering Rules

1. Keep the implementation simple.
2. Do not add unnecessary frameworks.
3. Use typed structured schemas.
4. Every tool must have a clear input/output contract.
5. Tools must return structured results.
6. Agent actions must pass through the harness.
7. Enforce a maximum number of tool calls.
8. Handle tool failures gracefully.
9. Validate important outputs.
10. Log important agent and tool events.
11. Write tests for critical behavior.
12. Never expose API keys or secrets.
13. Do not modify unrelated files.
14. Prefer deterministic logic for business rules.
15. Before adding a dependency, check whether it is actually necessary.

## Development Workflow

Plan ? Implement ? Test ? Fix ? Verify.

Before major changes:

- Inspect existing files.
- Explain the implementation plan.
- Make the smallest useful change.

After changes:

- Run tests.
- Report failures clearly.
- Fix failures.
- Report what changed.

## Hackathon Principle

Prefer:

Small product + strong reasoning + reliable execution + observable behavior

over:

Large product + unnecessary complexity.
