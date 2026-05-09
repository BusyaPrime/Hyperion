# Contribution workflow

## Branching

Create focused branches from `main`:

- `docs/<topic>` for documentation;
- `test/<area>` for coverage;
- `fix/<bug>` for behavior fixes;
- `refactor/<area>` for internal structure;
- `ci/<check>` for automation changes.

## Commit messages

Use Conventional Commits:

- `test: cover diagnostics report warnings`
- `refactor: extract trace replay helper`
- `fix: handle scalar observed values`
- `docs: add backend debugging guide`

## Pull request notes

Every pull request should explain:

- what changed and why;
- which module boundary it touches;
- which tests/checks were run;
- whether examples, docs, or benchmarks need updates.
