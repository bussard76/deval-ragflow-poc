from deval_ragflow.citations import CitationResolver, normalize_text, token_dice
from deval_ragflow.models import DocumentExtraction, Passage


def setup_mapping(registry):
    passages = [
        Passage("p1", "doc", "ver", 1, 1, None, (0, 0, 1, 1), 0, 11, "Hello World"),
        Passage("p2", "doc", "ver", 1, 2, None, (0, 1, 1, 2), 12, 24, "A second passage"),
        Passage("p3", "doc", "ver", 2, 1, None, (0, 1, 1, 2), 25, 30, "Third"),
    ]
    registry.save_extraction(DocumentExtraction("doc", "ver", "a" * 64, "paper.pdf", "extracted", 2, 1, "Hello World\nA second passage\fThird", passages))
    registry.upsert_dataset("scope", "dataset", "dataset")
    registry.reserve_mapping("scope", "ver", "deval-ver.pdf")
    registry.set_mapping_remote("scope", "ver", "remote-doc")
    return CitationResolver(registry, threshold=0.60)


def test_normalization_and_verified_position(registry):
    assert normalize_text("  Ｈｅｌｌｏ\n WORLD ") == "hello world"
    resolver = setup_mapping(registry)
    result = resolver.resolve({"document_id": "remote-doc", "positions": [[1, 1, 0, 11]], "content": "irrelevant"})
    assert result.method == "position"
    assert result.passage_uid == "p1"
    assert result.confidence == 1.0
    official = resolver.resolve({"document_id": "remote-doc", "positions": [[1, 0, 1, 0, 1]], "content": "irrelevant"})
    assert official.method == "position"
    assert official.passage_uid == "p1"
    zero_based = resolver.resolve({"document_id": "remote-doc", "_pdf_positions": [[[0, 0], 0, 1, 0, 1]], "content": "irrelevant"})
    assert zero_based.method == "position"
    assert zero_based.passage_uid == "p1"
    referenced = resolver.resolve({"document_id": "remote-doc", "page_number": 2, "position_int": [[1, 0, 1, 1, 2]], "content": "irrelevant"})
    assert referenced.method == "position"
    assert referenced.passage_uid == "p3"


def test_exact_token_fallback_ties_and_unknowns(registry):
    resolver = setup_mapping(registry)
    exact = resolver.resolve({"document_id": "remote-doc", "content": " HELLO   world "})
    assert exact.method == "exact"
    overlap = resolver.resolve({"document_id": "remote-doc", "content": "hello world extra"})
    assert overlap.method == "token_dice"
    assert overlap.passage_uid == "p1"
    unknown = resolver.resolve({"document_id": "missing", "content": "Hello World"})
    assert unknown.method == "unresolved"
    assert "unknown" in unknown.reason
    low = resolver.resolve({"document_id": "remote-doc", "content": "unrelated words"})
    assert low.method == "unresolved"


def test_dice_is_multiset_and_below_threshold_is_zero():
    assert token_dice("a a b", "a b b") == 2.0 * 2 / 6
    assert token_dice("", "text") == 0
