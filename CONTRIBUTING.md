# Contributing

Thanks for your interest in improving `mlcs`.

## Development setup

1. Install Python 3.10+
2. Install Poetry
3. Install dependencies

```bash
poetry install
poetry run pre-commit install
```

## Quality checks

Run all checks before opening a PR:

```bash
make check
```

Or individually:

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run pytest
```

## Commit & pull request guidelines

- Keep PRs focused and small.
- Use clear, imperative commit messages.
- Include a short test plan in every PR.
- Update docs for any user-visible change.

## Scope

This repository focuses on clean, reproducible implementations of completed
coursework. New contributions should improve:

- code quality and maintainability,
- reproducibility,
- documentation,
- developer tooling.
