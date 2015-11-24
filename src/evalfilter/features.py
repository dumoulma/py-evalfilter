import logging
import collections
from functools import partial

import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

import unicodedata
from evalfilter.analysis import tokenize_rant

KATAKANA = "KATAKANA"
HIRAGANA = "HIRAGANA"
KANJI = "CJK"
ALPHA = "LATIN"
DIGIT = "DIGIT"
MARKS = {'!', '?', '！', '？'}
PUNCT = {'、', '。', '「', '」', '（', '）', '＆', 'ー', '-', '＃', '￥'}


class FieldSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class UserProfileStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, user_profiles):
        return [user_profile for user_profile in user_profiles]


class RantStats(BaseEstimator, TransformerMixin):
    """Extract features from each rant for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, rants):
        def get_counts(_rants):
            for rant in _rants:
                tokens = tokenize_rant(rant, min_length=1)
                tokens = [t for t in tokens if t != '']
                yield token_counts(tokens), token_type_counts(tokens)

        def avg_token_length(counts_dict):
            total_tokens = sum(counts_dict.values())
            if not token_counts:
                return 0.0
            return sum([k * v for k, v in counts_dict.items()]) / total_tokens  # avgTokenLength

        def token_ratio(type_count, total_tokens):
            if type_count is 0 or total_tokens is 0:
                return 0
            return type_count / total_tokens

        return [{'kata': count_unicode_chars(rant, KATAKANA),
                 'hira': count_unicode_chars(rant, HIRAGANA),
                 'kanji': count_unicode_chars(rant, KANJI),
                 'alpha': count_unicode_chars(rant, ALPHA),
                 'digit': count_unicode_chars(rant, DIGIT),
                 'marks': count_chars_in_set(rant, is_mark),
                 'punct': count_chars_in_set(rant, is_punct),
                 'kataTokensRatio': token_ratio(type_dict['kata'], sum(counts_dict.values())),
                 'hiraTokensRatio': token_ratio(type_dict['hira'], sum(counts_dict.values())),
                 'kanjiTokensRatio': token_ratio(type_dict['kanji'], sum(counts_dict.values())),
                 'alphaTokensRatio': token_ratio(type_dict['alpha'], sum(counts_dict.values())),
                 'digitTokensRatio': token_ratio(type_dict['digit'], sum(counts_dict.values())),
                 'tokens': sum(counts_dict.values()),
                 '1char': counts_dict[1],
                 '2char': counts_dict[2],
                 '3char': counts_dict[3],
                 '4char': counts_dict[4],
                 '5+char': counts_dict[5],
                 'avgTokenLength': avg_token_length(counts_dict)
                 }
                for rant, (counts_dict, type_dict) in zip(rants, get_counts(rants))
                ]


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
        logging.debug("WARNING: is_katakana: {}:'{}'".format(ve, c))
    return False


def is_hiragana(c):
    try:
        return unicodedata.name(c)[:len(HIRAGANA)] == HIRAGANA
    except ValueError as ve:
        logging.debug("WARNING: is_hiragana: {}:'{}'".format(ve, c))
    return False


def is_kanji(c):
    try:
        return unicodedata.name(c)[:3] == "CJK"
    except ValueError as ve:
        logging.debug("WARNING: is_kanji: {}:'{}'".format(ve, c))
    return False


def is_alphabet(c):
    try:
        return unicodedata.name(c)[:len(ALPHA)] == ALPHA
    except ValueError as ve:
        logging.debug("WARNING: is_alphabet: {}:'{}'".format(ve, c))
    return False


def is_unicode_name(name, c):
    try:
        return unicodedata.name(c)[:len(name)] == name
    except ValueError as ve:
        logging.debug("WARNING: is_unicode_name: {}:'{}'".format(ve, c))
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


def map_to_token_type(word):
    if all(is_katakana(c) for c in word):
        return "kata"
    if all(is_hiragana(c) for c in word):
        return "hira"
    if all(is_kanji(c) or is_hiragana(c) for c in word):  # a kanji word has at least one kanji
        return "kanji"
    if all(is_alphabet(c) for c in word):
        return "alpha"
    if word.isdigit():
        return "digit"
    if all(is_punct(c) for c in word):
        return "punct"
    if all(is_mark(c) for c in word):
        return "mark"
    return "other"


def token_type_counts(rant_tokens):
    type_counts_ = dict()
    type_counts_['kata'] = 0
    type_counts_['hira'] = 0
    type_counts_['kanji'] = 0
    type_counts_['alpha'] = 0
    type_counts_['digit'] = 0
    for t in rant_tokens:
        if all(is_katakana(c) for c in t):
            type_counts_['kata'] += 1
        if all(is_hiragana(c) for c in t):
            type_counts_['hira'] += 1
        if all(is_kanji(c) or is_hiragana(c) for c in t):  # a kanji word has at least one kanji
            type_counts_['kanji'] += 1
        if all(is_alphabet(c) for c in t):
            type_counts_['alpha'] += 1
        if t.isdigit():
            type_counts_['digit'] += 1
    return type_counts_


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
