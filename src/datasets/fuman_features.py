import logging
import collections
from functools import partial

import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

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


class RantStats(BaseEstimator, TransformerMixin):
    """Extract features from each rant for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, rants):
        def get_token_counts(_rants):
            for rant in _rants:
                yield token_counts(tokenize_rant(rant, min_length=1))

        def avg_token_length(counts_dict):
            total_tokens = sum(counts_dict.values())
            if not token_counts:
                return 0.0
            return sum([k * v for k, v in counts_dict.items()]) / total_tokens  # avgTokenLength

        return [{'kata': count_unicode_chars(rant, KATAKANA),
                 'hira': count_unicode_chars(rant, HIRAGANA),
                 'kanji': count_unicode_chars(rant, KANJI),
                 'alpha': count_unicode_chars(rant, ALPHA),
                 'digit': count_unicode_chars(rant, DIGIT),
                 'marks': count_chars_in_set(rant, is_mark),
                 'punct': count_chars_in_set(rant, is_punct),
                 'tokens': sum(counts_dict.values()),
                 '1char': counts_dict[1],
                 '2char': counts_dict[2],
                 '3char': counts_dict[3],
                 '4char': counts_dict[4],
                 '5+char': counts_dict[5],
                 'avgTokenLength': avg_token_length(counts_dict)
                 }
                for rant, counts_dict in zip(rants, get_token_counts(rants))
                ]


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


def vectorize_text(raw_documents, _, vectorizer, tokenizer, min_df, max_features, stop_words=None, ngram_range=(1, 1)):
    if max_features is 0:
        return sp.csr_matrix([]), []
    vec = vectorizer(tokenizer=tokenizer, ngram_range=ngram_range, stop_words=stop_words, strip_accents='unicode',
                     min_df=min_df, max_features=max_features)
    transformed = vec.fit_transform(raw_documents)
    logging.info("Vectorized: {} ({})".format(transformed.shape, tokenizer.__name__))
    return transformed, vec.get_feature_names()


def vectorise_text_fit(raw_documents, fit_documents, vectorizer, tokenizer, ngram_range, min_df, max_features):
    if max_features is 0:
        return sp.csr_matrix([])
    vec = vectorizer(tokenizer=tokenizer, ngram_range=ngram_range, strip_accents='unicode', min_df=min_df,
                     max_features=max_features)
    vec.fit(fit_documents)
    transformed = vec.transform(raw_documents)
    logging.info("Vectorized (fit): {}".format(transformed.shape))
    return transformed


def encode_categoricals(X):
    """
    Transforms the categorical string values into int (necessary for some algorithms that can't process strings).
    :param X:
    :return:
    """
    import sklearn.preprocessing as pp
    state_enc = pp.LabelEncoder()
    encoded_state = state_enc.fit_transform([x[7] for x in X])
    gender_enc = pp.LabelEncoder()
    encoded_gender = gender_enc.fit_transform([x[8] for x in X])
    job_enc = pp.LabelEncoder()
    encoded_job = job_enc.fit_transform([x[9] for x in X])
    for x, es, eg, ej in zip(X, encoded_state, encoded_gender, encoded_job):
        x[0] = 1 if x[0] is 'True' else 0
        x[1] = 1 if x[1] is 'True' else 0
        x[2] = 1 if x[2] is 'True' else 0
        x[3] = 1 if x[3] is 'True' else 0
        x[4] = 1 if x[4] is 'True' else 0
        x[7] = es
        x[8] = eg
        x[9] = ej
    return X


def categorical_to_binary(X):
    import sklearn.preprocessing as pp
    # b = {'True', 'False'}
    # values = [b, b, b, b, b, len(set(x[7] for x in X)), len(set(x[8] for x in X)), len(set(x[9] for x in X))]
    ohe = pp.OneHotEncoder(categorical_features=[0, 1, 2, 3, 4, 7, 8, 9], sparse=False)
    ohe.fit(X)
    return ohe.transform(X)