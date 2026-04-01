lint files=".":
    uv run ruff check {{ files }} --fix
    uv run ruff format {{ files }}
    uv run ty check {{ files }}

build:
    uv sync
    uv build

test:
    uv run pytest

test-integration:
    uv run pytest -m "integration" --wav-file "local_test_data/oo23_181b004_extract_40m55m.wav"

test-all:
    uv run pytest -m "" --wav-file "local_test_data/oo23_181b004_extract_40m55m.wav"

test-coverage:
    uv run pytest --cov=orcai --cov-report=term-missing 
