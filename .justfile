ruffle files=".":
    uv run ruff check {{ files }} --select I --fix
    uv run ruff format {{ files }}
    uv run ty check {{ files }}

build:
    uv sync
    uv build

test:
    uv run pytest

dev-setup:
    uv sync --dev
    uv run pre-commit install
