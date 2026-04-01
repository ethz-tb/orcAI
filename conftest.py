"""Root-level pytest configuration.

Defines CLI options that must be registered before argument parsing.

Created using: claude-sonnet-4-6 on 2026-03-31
"""


def pytest_addoption(parser):
    parser.addoption(
        "--wav-file", action="store", help="Path to a .wav file for integration tests"
    )
    parser.addoption(
        "--channel",
        default=1,
        type=int,
        action="store",
        help="Audio channel to use (default: 1)",
    )
