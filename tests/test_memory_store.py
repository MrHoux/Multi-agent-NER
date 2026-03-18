from maner.memory.store import MemoryStore


def test_memory_write_retrieve_promote(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path)
    try:
        r1 = store.writeback(
            kind="term",
            key="Acme Corp",
            value={"ent_type": "ORG"},
            confidence=0.9,
            verifier_pass=True,
            promote_threshold=2,
        )
        assert r1["written"] is True
        assert r1["status"] == "candidate"

        hits1 = store.retrieve("Acme", top_k=5)
        assert hits1
        assert hits1[0]["status"] in {"candidate", "active"}

        r2 = store.writeback(
            kind="term",
            key="Acme Corp",
            value={"ent_type": "ORG"},
            confidence=0.95,
            verifier_pass=True,
            promote_threshold=2,
        )
        assert r2["status"] == "active"

        hits2 = store.retrieve("Acme", top_k=5)
        assert hits2[0]["status"] == "active"
    finally:
        store.close()


def test_memory_gate_by_verifier_and_confidence(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    try:
        r1 = store.writeback(
            kind="case",
            key="bad",
            value={"x": 1},
            confidence=0.9,
            verifier_pass=False,
        )
        r2 = store.writeback(
            kind="case",
            key="low",
            value={"x": 1},
            confidence=0.1,
            verifier_pass=True,
        )
        assert r1["written"] is False
        assert r2["written"] is False
        assert store.retrieve("bad", top_k=5) == []
    finally:
        store.close()
