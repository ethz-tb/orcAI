lint files=".":
    uv run ruff check {{ files }} --fix
    uv run ruff format {{ files }}
    uv run ty check {{ files }}

build:
    uv sync
    uv build

test:
    uv run pytest
