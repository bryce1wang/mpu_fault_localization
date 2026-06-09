from __future__ import annotations

from pathlib import Path

from dpu_fault_agent.cli import main


def test_cli_run_status_resume_report(tmp_path: Path, capsys) -> None:
    db = tmp_path / "checkpoints.sqlite"
    log = tmp_path / "boot.log"
    log.write_text(
        "[dpu_drv] init failed with DPU_ERR_TIMEOUT status=0xdead\n",
        encoding="utf-8",
    )
    source = tmp_path / "src"
    source.mkdir()
    (source / "driver.c").write_text(
        "int dpu_init(void) {\n    return DPU_ERR_TIMEOUT;\n}\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--checkpoint-db",
                str(db),
                "run",
                "--thread-id",
                "cli-case",
                "--problem",
                "DPU init timeout",
                "--log",
                str(log),
                "--source",
                str(source),
            ]
        )
        == 0
    )
    assert "human_gate: interrupted" in capsys.readouterr().out

    assert main(["--checkpoint-db", str(db), "status", "--thread-id", "cli-case"]) == 0
    assert "approval: pending" in capsys.readouterr().out

    assert (
        main(
            [
                "--checkpoint-db",
                str(db),
                "resume",
                "--thread-id",
                "cli-case",
                "--approve",
                "H1",
            ]
        )
        == 0
    )

    report = tmp_path / "report.md"
    assert (
        main(
            [
                "--checkpoint-db",
                str(db),
                "report",
                "--thread-id",
                "cli-case",
                "--output",
                str(report),
            ]
        )
        == 0
    )
    assert "DPU Fault Localization Report" in report.read_text(encoding="utf-8")


def test_cli_problem_only_then_resume_with_materials(tmp_path: Path, capsys) -> None:
    db = tmp_path / "checkpoints.sqlite"
    log = tmp_path / "boot.log"
    log.write_text(
        "[dpu_drv] init failed with DPU_ERR_TIMEOUT status=0xdead\n",
        encoding="utf-8",
    )
    source = tmp_path / "src"
    source.mkdir()
    (source / "driver.c").write_text(
        "int dpu_init(void) {\n    return DPU_ERR_TIMEOUT;\n}\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--checkpoint-db",
                str(db),
                "run",
                "--thread-id",
                "problem-only",
                "--problem",
                "VF init timeout",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "human_gate: interrupted" in out
    assert "diagnosis_plan:" in out

    assert (
        main(
            [
                "--checkpoint-db",
                str(db),
                "resume",
                "--thread-id",
                "problem-only",
                "--log",
                str(log),
                "--source",
                str(source),
                "--note",
                "failure happens after queue setup",
            ]
        )
        == 0
    )
    assert "final_report: available" in capsys.readouterr().out
