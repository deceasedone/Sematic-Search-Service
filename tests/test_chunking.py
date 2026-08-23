from app.search.chunking import chunk_text


def test_short_text_stays_one_chunk():
    chunks = chunk_text("hello world this is short", target_words=300, overlap_words=50)
    assert len(chunks) == 1
    assert chunks[0].word_count == 5


def test_long_text_splits_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, target_words=300, overlap_words=50)
    assert len(chunks) > 1
    first_tail = chunks[0].text.split()[-50:]
    second_head = chunks[1].text.split()[:50]
    assert first_tail == second_head
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(words)


def test_empty_text_returns_one_empty_chunk():
    chunks = chunk_text("")
    assert len(chunks) == 1
    assert chunks[0].word_count == 0
