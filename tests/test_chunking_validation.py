import pytest

from utils.chunking import chunk_text


@pytest.mark.parametrize('max_tokens,overlap', [
    (0, 0), (-1, 0), (3, 3), (3, 4), (3, -1),
])
@pytest.mark.parametrize('text', ['one two three', ''])
def test_invalid_chunk_settings_raise_value_error(max_tokens, overlap, text):
    with pytest.raises(ValueError):
        chunk_text(text, max_tokens=max_tokens, overlap=overlap)


def test_maximum_valid_overlap_advances_one_word():
    assert chunk_text('a b c d', max_tokens=3, overlap=2) == [
        'a b c', 'b c d', 'c d', 'd',
    ]


def test_single_word_chunks_preserve_every_word():
    assert chunk_text('a b c', max_tokens=1, overlap=0) == ['a', 'b', 'c']


def test_empty_text_with_valid_settings_returns_no_chunks():
    assert chunk_text('', max_tokens=1, overlap=0) == []
