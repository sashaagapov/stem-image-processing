"""Regression tests for user-facing command-line errors."""
import sys

import pytest

from common.cli import main


@pytest.mark.parametrize(
    ("member", "pipeline"),
    [("memberX", "pipeline1"), ("member1", "pipelineX")],
)
def test_cli_invalid_pipeline_shows_readable_error(
    monkeypatch: pytest.MonkeyPatch, member: str, pipeline: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli", "--member", member, "--pipeline", pipeline, "--image", "missing.jpg"],
    )

    with pytest.raises(SystemExit, match=r"Помилка: Пайплайн .* не знайдено"):
        main()
