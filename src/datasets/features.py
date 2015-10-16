import logging
import collections
from functools import partial

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse as sp

import unicodedata
from util.mecab import tokenize_rant

KATAKANA = "KATAKANA"
HIRAGANA = "HIRAGANA"
KANJI = "CJK"
ALPHA = "LATIN"
DIGIT = "DIGIT"
MARKS = {'!', '?', '！', '？'}
PUNCT = {'、', '。', '「', '」', '（', '）', '＆', 'ー', '-', '＃', '￥'}


def get_header():
    return "katacount,hiracount,kanjicount,alphacount,digitcount,markcount,punctcount,totaltokens,1chartokens," + \
           "2chartokens,3chartokens,4chartokens,5+chartokens,avgTokenLength"


def rant_text_features(x, rant):
    """
    Creates a number of new features based on the characteristics of the post words and characters.

    :param x: the raw row data for a post
    :param rant: the post text
    :return:
    """
    x.append(count_unicode_chars(rant, KATAKANA))  # katacount
    x.append(count_unicode_chars(rant, HIRAGANA))  # hiracount
    x.append(count_unicode_chars(rant, KANJI))  # kanjicount
    x.append(count_unicode_chars(rant, ALPHA))  # alphacount
    x.append(count_unicode_chars(rant, DIGIT))  # digitcount
    x.append(count_chars_in_set(rant, is_mark))  # markcount
    x.append(count_chars_in_set(rant, is_punct))  # punctcount
    counts_dict = token_counts(tokenize_rant(rant, min_length=1))
    total_tokens = sum(counts_dict.values())
    x.append(total_tokens)  # totaltokens (words)
    x.append(counts_dict[1])  # 1chartokens
    x.append(counts_dict[2])  # 2chartokens
    x.append(counts_dict[3])  # 3chartokens
    x.append(counts_dict[4])  # 4chartokens
    x.append(counts_dict[5])  # 5+chartokens
    if total_tokens > 0:
        x.append(sum([k * v for k, v in counts_dict.items()]) / total_tokens)  # avgTokenLength
    else:
        x.append(0)
    return x


def is_digit(c):
    return unicodedata.name(c)[:len(DIGIT)] == DIGIT


def is_mark(c):
    return c in MARKS


def is_punct(c):
    return c in PUNCT


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


def count_unicode_chars(rant, charset_name):
    return len(list(filter(partial(is_unicode_name, charset_name), rant)))


def count_chars_in_set(rant, predicate_func):
    return len(list(filter(lambda c: predicate_func(c), rant)))


def token_counts(rant_tokens):
    counts_ = collections.defaultdict(int)
    for t in rant_tokens:
        count = len(t)
        if count > 5:
            count = 5
        counts_[count] += 1
    if 0 in counts_:
        counts_.pop(0)
    return counts_


def tfidf_word(raw_documents, tokenizer, stop_words, min_df, max_features):
    if max_features is 0:
        return sp.csr_matrix([])
    wdvec = TfidfVectorizer(tokenizer=tokenizer, strip_accents='unicode', stop_words=stop_words, min_df=min_df,
                            max_features=max_features)
    word_vects = wdvec.fit_transform(raw_documents)
    logging.info("Rants vectorized: {}".format(word_vects.shape))
    return word_vects


def tfidf_pos(raw_documents, tokenizer, ngram_range, min_df, max_features):
    if max_features is 0:
        return sp.csr_matrix([])
    posvec = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range, strip_accents='unicode', min_df=min_df,
                             max_features=max_features)
    pos_vects = posvec.fit_transform(raw_documents)
    logging.info("POS vectorized: {}".format(pos_vects.shape))
    return pos_vects


def tf_pos(raw_documents, tokenizer, ngram_range, min_df, max_features):
    if max_features is 0:
        return sp.csr_matrix([])
    posvec = CountVectorizer(tokenizer=tokenizer, ngram_range=ngram_range, strip_accents='unicode', min_df=min_df,
                             max_features=max_features)
    pos_vects = posvec.fit_transform(raw_documents)
    logging.info("POS vectorized: {}".format(pos_vects.shape))
    return pos_vects


def encode_categoricals(X):
    """
    Transforms the categorical string values into int (necessary for some algorithms that can't process strings).
    :param X:
    :return:
    """
    import sklearn.preprocessing as pp
    age_enc = pp.LabelEncoder()
    encoded_age = age_enc.fit_transform([x[6] for x in X])
    state_enc = pp.LabelEncoder()
    encoded_state = state_enc.fit_transform([x[7] for x in X])
    job_enc = pp.LabelEncoder()
    encoded_job = job_enc.fit_transform([x[9] for x in X])
    for x, ea, es, ej in zip(X, encoded_age, encoded_state, encoded_job):
        x[6] = ea
        x[7] = es
        x[9] = ej


def categorical_to_binary(X):
    import sklearn.preprocessing as pp
    values = [len(set(x[6] for x in X)), len(set(x[6] for x in X)), len(set(x[6] for x in X))]
    ohe = pp.OneHotEncoder(n_values=values, categorical_features=[6, 7, 9], sparse=False)
    return ohe.fit_transform(X)
