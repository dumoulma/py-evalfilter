import logging
import collections
from functools import partial

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import unicodedata
from mecab import tokenize_rant

KATAKANA = "KATAKANA"
HIRAGANA = "HIRAGANA"
KANJI = "CJK"
ALPHA = "LATIN"
DIGIT = "DIGIT"
SYMBOL = {'!', '?'}


def get_header():
    return "katacount,hiracount,kanjicount,alphacount,digitcount,symbolcount,totaltokens,1chartokens," + \
           "2chartokens,3chartokens,4chartokens,5+chartokens,avgTokLength"


def add_manual_features(x, rant):
    x.append(count_unicode_chars(rant, KATAKANA))  # katacount
    x.append(count_unicode_chars(rant, HIRAGANA))  # hiracount
    x.append(count_unicode_chars(rant, KANJI))  # kanjicount
    x.append(count_unicode_chars(rant, ALPHA))  # alphacount
    x.append(count_unicode_chars(rant, DIGIT))  # digitcount
    x.append(len(list(filter(is_symbol, rant))))  # symbolcount
    counts_dict = token_counts(tokenize_rant(rant))
    total_tokens = sum(counts_dict.values())
    x.append(total_tokens)  # totaltokens (words)
    x.append(counts_dict[1])  # 1chartokens
    x.append(counts_dict[2])  # 2chartokens
    x.append(counts_dict[3])  # 3chartokens
    x.append(counts_dict[4])  # 4chartokens
    if total_tokens > 0:
        x.append(sum(counts_dict.values()) / total_tokens)  # 5+chartokens
    else:
        x.append(0)
    x.append(len(list(filter(lambda k: k >= 5, counts_dict.keys()))))  # avgTokLength
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
