"""Tests for detection/open_ai.py's chunking in classify_file(): each chunk
after the first repeats the tail of the previous one, so an obfuscation
pattern straddling a chunk boundary isn't split blind between two LLM calls.
No network calls - get_llm_prediction is monkeypatched to just record what
it was asked to classify.
"""
from detection import open_ai


def _classify_and_record_chunks(monkeypatch, tmp_path, text, chunk_size=3000, chunk_overlap=200):
    seen_chunks = []

    def fake_predict(script_text, api_key, model=None, max_tokens=None):
        seen_chunks.append(script_text)
        return "NonObfuscated"

    monkeypatch.setattr(open_ai, "get_llm_prediction", fake_predict)

    file_path = tmp_path / "sample.js"
    file_path.write_text(text)

    verdict = open_ai.classify_file(
        str(file_path), "fake-key", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return verdict, seen_chunks


def test_classify_file_single_chunk_when_text_fits(monkeypatch, tmp_path):
    text = "A" * 500
    verdict, chunks = _classify_and_record_chunks(monkeypatch, tmp_path, text)
    assert verdict == "NonObfuscated"
    assert chunks == [text]


def test_classify_file_chunks_overlap_and_cover_the_whole_text(monkeypatch, tmp_path):
    text = "A" * 3000 + "B" * 3000 + "C" * 500  # 6500 chars, 3 chunks expected
    verdict, chunks = _classify_and_record_chunks(monkeypatch, tmp_path, text)

    assert verdict == "NonObfuscated"
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [3000, 3000, 900]

    # The 200-char overlap means each chunk after the first starts with the
    # tail of the previous chunk.
    assert chunks[1][:200] == chunks[0][-200:]
    assert chunks[2][:200] == chunks[1][-200:]

    # No gaps: the marker straddling the A/B boundary at index 3000 appears
    # whole inside a single chunk rather than being split across two.
    boundary_marker = text[2900:3100]  # spans the A/B boundary
    assert any(boundary_marker in chunk for chunk in chunks)
