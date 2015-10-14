import logging

import unicodedata
import collections
from functools import partial
from mecab import tokenize_rant

KATAKANA = "KATAKANA"
HIRAGANA = "HIRAGANA"
KANJI = "CJK"
ALPHA = "LATIN"
DIGIT = "DIGIT"
SYMBOL = {'!', '?'}


def add_manual_features(x, rant):
    x.append(count_unicode_chars(rant, KATAKANA))
    x.append(count_unicode_chars(rant, HIRAGANA))
    x.append(count_unicode_chars(rant, KANJI))
    x.append(count_unicode_chars(rant, ALPHA))
    x.append(count_unicode_chars(rant, DIGIT))
    x.append(len(list(filter(is_symbol, rant))))
    counts_dict = token_counts(tokenize_rant(rant))
    total_tokens = sum(counts_dict.values())
    x.append(total_tokens)
    x.append(counts_dict[1])
    x.append(counts_dict[2])
    x.append(counts_dict[3])
    x.append(counts_dict[4])
    x.append(len(list(filter(lambda k: k >= 5, counts_dict.keys()))))
    if total_tokens > 0:
        x.append(sum(counts_dict.values()) / total_tokens)
    else:
        x.append(0)
    return x


def is_digit(c):
    return unicodedata.name(c)[:len(DIGIT)] == DIGIT


def is_symbol(c):
    return c in SYMBOL


def is_katakana(c):
    try:
        return unicodedata.name(c)[:len(KATAKANA)] == KATAKANA
    except ValueError as ve:
        logging.warning(ve)
    return False


def is_hiragana(c):
    try:
        return unicodedata.name(c)[:len(HIRAGANA)] == HIRAGANA
    except ValueError as ve:
        logging.warning(ve)
    return False


def is_kanji(c):
    return unicodedata.name(c)[:len(KANJI)] == KANJI


def is_alphabet(c):
    return unicodedata.name(c)[:len(ALPHA)] == ALPHA


def is_unicode_name(name, c):
    try:
        return unicodedata.name(c)[:len(name)] == name
    except ValueError as ve:
        logging.warning(ve)
    return False


def count_unicode_chars(rant, name):
    return len(list(filter(partial(is_unicode_name, name), rant)))


def token_counts(rant_tokens):
    counts_ = collections.defaultdict(int)
    for t in rant_tokens:
        counts_[len(t)] += 1
    if 0 in counts_:
        counts_.pop(0)
    return counts_
