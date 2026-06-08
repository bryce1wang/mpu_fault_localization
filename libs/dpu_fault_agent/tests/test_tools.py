from __future__ import annotations

from pathlib import Path

from dpu_fault_agent.tools import derive_search_terms, search_source, triage_logs


def test_log_triage_extracts_error_codes_and_modules(tmp_path: Path) -> None:
    log = tmp_path / "boot.log"
    log.write_text(
        "[dpu_drv] init failed with DPU_ERR_TIMEOUT status=0xdead\n",
        encoding="utf-8",
    )

    observations = triage_logs([str(log)])

    summaries = "\n".join(obs["summary"] for obs in observations)
    assert "DPU_ERR_TIMEOUT" in summaries
    assert "dpu_drv" in summaries
    assert any(obs.get("line") == 1 for obs in observations)


def test_source_search_finds_symbols(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "driver.c").write_text(
        "int dpu_init(void) {\n    return DPU_ERR_TIMEOUT;\n}\n",
        encoding="utf-8",
    )

    hits = search_source(str(source), ["DPU_ERR_TIMEOUT"])

    assert hits
    assert hits[0]["path"].endswith("driver.c")
    assert hits[0]["line"] == 2


def test_derive_search_terms_uses_problem_and_observations() -> None:
    terms = derive_search_terms(
        "DPU init timeout",
        [{"summary": "DPU_ERR_TIMEOUT from dpu_drv", "symbol": "DPU_ERR_TIMEOUT"}],
    )

    assert "DPU_ERR_TIMEOUT" in terms
