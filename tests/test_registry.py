from concurrent.futures import ThreadPoolExecutor

from deval_ragflow.models import DocumentExtraction, Passage


def extraction(uid="doc", version="version"):
    passage = Passage(
        uid + "-p", uid, version, 1, 1, None, (0.0, 0.0, 10.0, 10.0), 0, 4, "text"
    )
    return DocumentExtraction(
        uid, version, "a" * 64, "file.pdf", "extracted", 1, 4, "text", [passage]
    )


def test_extraction_and_mapping_are_idempotent(registry):
    item = extraction()
    registry.save_extraction(item)
    registry.save_extraction(item)
    assert registry.counts()["documents"] == 1
    assert registry.counts()["document_versions"] == 1
    assert registry.counts()["passages"] == 1
    registry.upsert_dataset("scope", "remote-dataset", "name", config={"x": 1})
    first = registry.reserve_mapping("scope", item.version_uid, "deval-version.pdf")
    second = registry.reserve_mapping("scope", item.version_uid, "deval-version.pdf")
    assert first.mapping_uid == second.mapping_uid
    renamed = registry.reserve_mapping("scope", item.version_uid, "file.pdf")
    assert renamed.remote_name == "file.pdf"
    registry.set_mapping_remote("scope", item.version_uid, "remote-doc")
    assert (
        registry.get_mapping_by_remote_id("remote-doc").version_uid == item.version_uid
    )


def test_delete_mapping_removes_unshared_provenance(registry):
    item = extraction()
    registry.save_extraction(item)
    registry.upsert_dataset("scope", "remote-dataset", "name")
    registry.reserve_mapping("scope", item.version_uid, "same.pdf")
    registry.set_mapping_remote("scope", item.version_uid, "remote-doc")
    registry.upsert_index("scope", "parse", item.version_uid, state="DONE")

    deleted = registry.delete_mapping("scope", item.version_uid)

    assert deleted is not None
    assert registry.get_mapping("scope", item.version_uid) is None
    assert registry.counts()["ragflow_documents"] == 0
    assert registry.counts()["document_versions"] == 0
    assert registry.counts()["documents"] == 0
    assert registry.counts()["indexes"] == 0


def test_concurrent_reservations_share_one_mapping(registry):
    item = extraction()
    registry.save_extraction(item)
    registry.upsert_dataset("scope", "remote-dataset", "name")

    def reserve(_):
        return registry.reserve_mapping(
            "scope", item.version_uid, "same.pdf"
        ).mapping_uid

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(reserve, range(20)))
    assert len(set(ids)) == 1
    assert registry.counts()["ragflow_documents"] == 1


def test_unowned_dataset_cannot_be_claimed_by_a_later_upsert(registry):
    registry.upsert_dataset("scope", "remote-dataset", "name", owned=False)
    registry.upsert_dataset("scope", "remote-dataset", "name")
    assert registry.get_dataset("scope").owned is False


def test_indexes_and_reset_only_remove_local_rows_after_success(registry):
    item = extraction()
    registry.save_extraction(item)
    registry.upsert_dataset("scope", "remote-dataset", "name")
    registry.reserve_mapping("scope", item.version_uid, "same.pdf")
    registry.upsert_index("scope", "parse", item.version_uid, state="DONE", progress=1)
    assert registry.get_index("scope", "parse", item.version_uid)["state"] == "DONE"
    # A caller that has not completed remote deletion cannot use this method.
    registry.delete_dataset_rows("scope", "remote-dataset")
    assert registry.counts()["datasets"] == 0
    assert registry.counts()["ragflow_documents"] == 0
