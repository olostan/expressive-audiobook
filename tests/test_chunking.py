from expressive_audiobook.chunking import chunk_chapter
from expressive_audiobook.config import ChunkingConfig
from expressive_audiobook.models import Block, Chapter, LinkedElement
from expressive_audiobook.textutil import split_sentences


def _config() -> ChunkingConfig:
    return ChunkingConfig(
        wpm=60,
        target_min_seconds=10,
        target_max_seconds=20,
        hard_cap_seconds=30,
        figure_description_sentences=2,
        words_per_spoken_sentence=5,
        formula_description_words=8,
        footnote_lead_in_words=2,
    )


def _words(count: int, end: str = ".") -> str:
    tokens = [f"w{index}" for index in range(count - 1)]
    tokens.append("end" + end)
    return " ".join(tokens)


def _chapter(blocks, elements=None, kind="prose") -> Chapter:
    return Chapter(
        id="ch-001",
        index=1,
        title="A Short Chapter",
        kind=kind,
        page_start=1,
        page_end=1,
        detection="heuristic",
        blocks=blocks,
        elements=elements or [],
    )


def _block(index: int, text: str, kind="paragraph", speaker=None) -> Block:
    return Block(
        id=f"b-{index}",
        kind=kind,
        text=text,
        page_start=1,
        page_end=1,
        stream_index=index,
        speaker=speaker,
    )


def test_packs_inside_the_window_and_never_splits_a_sentence():
    config = _config()
    sentences = [_words(6) for _ in range(6)]
    text = " ".join(sentences)
    chunks = chunk_chapter(_chapter([_block(0, text)]), config)
    rebuilt = " ".join(chunk.text for chunk in chunks)
    assert rebuilt == text
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            assert sentence in text
        assert chunk.word_count <= config.hard_words
        assert chunk.estimated_duration_sec < config.hard_cap_seconds


def test_unsplittable_sentence_stays_whole_and_warns():
    config = _config()
    text = _words(40)
    chunks = chunk_chapter(_chapter([_block(0, text)]), config)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert "sentence_exceeds_hard_cap" in chunks[0].warnings


def test_footnote_attaches_inline_when_it_fits():
    config = _config()
    block = _block(0, "The rain " + "[[footnote:fn-1]] " + _words(8))
    note = LinkedElement(
        id="fn-1",
        kind="footnote",
        anchor_block_id=block.id,
        anchor_offset=0,
        stream_index=0,
        page=1,
        text=_words(4),
        marker="1",
    )
    chunks = chunk_chapter(_chapter([block], [note]), config)
    spoken = [item for chunk in chunks for item in chunk.associated if item.id == "fn-1" and item.speak]
    assert len(spoken) == 1
    assert spoken[0].placement == "inline"
    assert note.text in spoken[0].text


def test_oversized_footnote_becomes_the_next_chunk():
    config = _config()
    block = _block(0, _words(16))
    note = LinkedElement(
        id="fn-1",
        kind="footnote",
        anchor_block_id=block.id,
        anchor_offset=0,
        stream_index=0,
        page=1,
        text=_words(20),
        marker="1",
    )
    chunks = chunk_chapter(_chapter([block], [note]), config)
    assert len(chunks) == 2
    assert any(item.id == "fn-1" and not item.speak for item in chunks[0].associated)
    assert chunks[1].kind == "footnote"
    assert chunks[1].associated[0].speak
    assert note.text in chunks[1].text


def test_figure_budget_counts_toward_duration():
    config = _config()
    block = _block(0, _words(12))
    figure = LinkedElement(
        id="fig-1",
        kind="figure",
        anchor_block_id=block.id,
        anchor_offset=0,
        stream_index=0,
        page=1,
        text="Figure 1. A crossing.",
        caption="Figure 1. A crossing.",
        image_path="001/figures/fig-1.png",
        figure_source="vector",
    )
    chunks = chunk_chapter(_chapter([block], [figure]), config)
    assert len(chunks) == 1
    assert chunks[0].word_count == 12 + 10
    assert chunks[0].associated[0].estimated_spoken_words == 10
    assert chunks[0].associated[0].image_path.endswith("fig-1.png")


def test_dialogue_turns_stay_intact():
    config = _config()
    blocks = [
        _block(0, _words(8), kind="dialogue_turn", speaker="Guide"),
        _block(1, _words(8, end="?"), kind="dialogue_turn", speaker="Wanderer"),
        _block(2, _words(8), kind="dialogue_turn", speaker="Guide"),
    ]
    chunks = chunk_chapter(_chapter(blocks, kind="dialogue"), config)
    assert chunks[0].kind == "dialogue"
    assert chunks[0].text.count("end") == 2
    assert chunks[0].speakers == ["Guide", "Wanderer"]
    assert chunks[1].text == blocks[2].text
    assert "end?" in chunks[0].text
