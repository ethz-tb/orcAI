# Project conventions

- This is a python 3.11 project
- it uses type hints and docstring throughout.
- Avoid flowery language and keep descriptions short and to the point.
- Files created or edited by Claude AI are always annotated with a line in the docstring stating model identifier used in this session and the date of of creation. If no docstring exists yet, add one with a title and a short description.
- It uses uv as a package and project manager
- It uses uv as the build backend
- It uses ruff as a formatter and linter.
- It uses ty for type checking
- It uses pytest for the testing framework
- it uses pre-commit to manage git hooks

## Package Management Commands

- All Python dependencies **must be installed and synchronized** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`

## Build commands

- Build package: `uv build`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools with `uv run <tool>` (e.g. `uv run pytest`, `uv run ruff`, `uv run pre-commit`)
- Launch a Python REPL with `uv run python`

## Testing

- pytest settings are defined in pyproject.toml
- tests are in the tests module
- Tests use fixtures defined in conftest.py. If necessary, define new fixtures in conftest.py.
- Run tests: `uv run pytest`
- Type checking: `uv run ty check`

## Linting and formatting

- Run linter and formatter after after done with edits.
- Lint: `uv run ruff check <files> --fix`
- Format: `uv run ruff format <files>`
