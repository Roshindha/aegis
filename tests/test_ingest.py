from aegis.ingest import build_chunks, parse_reference, resolve_params


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


def test_one_chunk_per_control():
    chunks = build_chunks()
    assert len(chunks) == 137
    assert len({c.control_id for c in chunks}) == 137   # no duplicate IDs
    assert all("{{" not in c.text for c in chunks)      # every placeholder resolved