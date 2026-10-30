from expressive_audiobook.textutil import (
    join_hyphenated_lines,
    max_words_under,
    normalize_text,
    split_sentences,
)


def test_rejoin_line_break_hyphen_and_keep_compounds():
    text = join_hyphenated_lines(["The travelers agreed to recon-", "sider the route."])
    assert text == "The travelers agreed to reconsider the route."
    compound = join_hyphenated_lines(["A well-known path", "continues."])
    assert compound == "A well-known path continues."


def test_ligatures_and_soft_hyphen():
    assert normalize_text("the ﬁnal soft\u00adhyphen") == "the final softhyphen"


def test_split_sentences_keeps_abbreviations():
    parts = split_sentences("Dr. Lane arrived. Then we left.")
    assert parts == ["Dr. Lane arrived.", "Then we left."]


def test_hard_cap_word_limit_is_strict():
    assert max_words_under(60, 150) == 149
    assert max_words_under(60, 150) / 150 * 60 < 60
