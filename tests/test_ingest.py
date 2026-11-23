import json
from pathlib import Path

import pytest

from aegis.ingest import _part, _render, build_chunks, parse_reference, resolve_params

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
needs_chunks = pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="chunks.jsonl not built")


def test_parse_nist_reference():
    ref = parse_reference("NIST SP 800-53 IA-2(1): Multi-factor Authentication to Privileged Accounts")
    assert (ref.framework, ref.ref_id) == ("NIST SP 800-53", "IA-2(1)")


def test_parse_bare_nist_id():
    ref = parse_reference("AC-2: Account Management")
    assert (ref.framework, ref.ref_id) == ("NIST SP 800-53", "AC-2")


def test_csf_not_mistaken_for_800_53():
    ref = parse_reference("PR.AA-01: Identities and credentials for authorized users")
    assert (ref.framework, ref.ref_id) == ("NIST CSF 2.0", "PR.AA-01")


def test_parse_mvsp_reference():
    ref = parse_reference("MVSP 2.7: Logging")
    assert (ref.framework, ref.ref_id) == ("MVSP", "2.7")


def test_resolve_params():
    control = {"params": [{"id": "x_prm_1", "label": "time period (days)"}]}
    out = resolve_params("Review every {{ insert: param, x_prm_1 }} day(s).", control)
    assert out == "Review every [agency-defined: time period (days)] day(s)."


@pytest.mark.skipif(not Path("data/raw/tech-standards/catalogs/im8-reform.json").exists(),
                    reason="raw GovTech catalogue not cloned (data/ is gitignored)")
def test_one_chunk_per_control():
    chunks = build_chunks()
    assert len(chunks) == 137
    assert len({c.control_id for c in chunks}) == 137   # no duplicate IDs
    assert all("{{" not in c.text for c in chunks)      # every placeholder resolved


def test_nested_list_items_are_kept():
    control = {"parts": [{
        "name": "statement",
        "prose": "Certificates must:",
        "parts": [
            {"props": [{"name": "label", "value": "a."}], "prose": "be signed by a trusted CA"},
            {"props": [{"name": "label", "value": "b."}], "prose": "use at least 2048-bit keys"},
        ],
    }]}
    out = _part(control, "statement")
    assert "Certificates must:" in out
    assert "(a.) be signed by a trusted CA" in out
    assert "(b.) use at least 2048-bit keys" in out


@needs_chunks
def test_ns6_contains_all_four_certificate_rules():
    chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    ns6 = next(c for c in chunks if c["control_id"] == "NS-6")
    assert all(f"({label})" in ns6["text"] for label in ("1", "2", "3", "4"))


def test_oscal_link_uuid_stripped_label_kept():
    out = _render({"prose": "See [SP 800-63b](#e59c5a7c-8b1f-4a2d-9c3e-111111111111) for details."})
    assert out == "See SP 800-63b for details."


@needs_chunks
def test_no_chunk_contains_raw_oscal_link():
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        assert "](#" not in json.loads(line)["text"]


@needs_chunks
def test_committed_corpus_matches_freshly_built_chunks():
    corpus = Path("corpus/chunks.jsonl")
    if not corpus.exists():
        pytest.skip("no corpus snapshot")
    assert corpus.read_text(encoding="utf-8") == CHUNKS_PATH.read_text(encoding="utf-8")
