import logging
import json
import os


def make_csv_row(x, i, pos_vects, rants_vects) -> str:
    features = ','.join(str(i) for i in x[:-1]) + ','
    if pos_vects.shape[1] is 0:
        pos = ''
    else:
        pos = ','.join(str(j) for j in pos_vects[i].todense().tolist()[0]) + ','
    if rants_vects.shape[1] is 0:
        wd = ''
    else:
        wd = ','.join(str(k) for k in rants_vects[i].todense().tolist()[0]) + ','
    row = features + pos + wd + str(x[-1]) + '\n'
    if i and i % 1000 == 0:
        logging.debug("{}:{}\t{}".format(i, row.split(',')[-1], row.split(',')[:25]))
        logging.debug("{}:{}".format(i, len(row.split(','))))
    return row


def make_svmlight_row(x, y, i, pos_vects, rants_vects) -> str:
    new_line = list()
    new_line.append(str(y))

    x_line = list_to_svmlight(x, 0)
    new_line += x_line

    pos_start = len(x)
    if pos_vects.shape[1] is not 0:
        pos_items = list_to_svmlight(pos_vects[i].todense().tolist()[0], pos_start)
        new_line += pos_items
    rants_start = pos_start + pos_vects.shape[1]

    if rants_vects.shape[1] is not 0:
        rants_items = list_to_svmlight(pos_vects[i].todense().tolist()[0], rants_start)
        new_line += rants_items

    svmlight_line = " ".join(new_line)
    svmlight_line += "\n"
    return svmlight_line


def list_to_svmlight(l, start_index):
    items = list()
    for i, item in enumerate(l, start_index):
        if float(item) == 0.0:
            continue
        new_item = "%s:%s" % (i + 1, item)
        items.append(new_item)
        start_index += 1
    return items


def sparse_to_svlight(vector, start_index):
    items = list()
    for i, j in enumerate(vector.nonzero(), start_index):
        item = vector[j]
        if item == '' or float(item) == 0.0:
            continue
        new_item = "%s:%s" % (i + 1, item)
        items.append(new_item)
        start_index += 1
    return items


def vector_headers(pos_features, word_features, simple=True):
    """
    Given an existing CSV string of headers, will concatenate the headers for
    the features from the pos and word features.

    :param headers: the current headers as a comma separated list of column names
    :param pos_features: the list of features of the pos vector (i.e. the vocabulary of the pos vectorizer)
    :param word_features: the list of features of the word vector (i.e. the vocabulary of the word vectorizer)
    :param simple: if True, features will be ['pos_','word_'] + N. if False, features are used as headers
    :return: the updated header list as a string
    """
    headers = ''
    pos_headers = pos_features
    word_headers = word_features

    if simple:
        pos_headers = ["pos_" + str(i) for i in range(len(pos_features))]
        word_headers = ["word_" + str(i) for i in range(len(word_features))]
    if len(pos_features):
        headers += ',' + ','.join(pos_headers)
    if len(word_features):
        headers += ',' + ','.join(word_headers)
    logging.debug("Header: {}".format(headers))
    return headers


def generate_vector_headers(v, prefix):
    if v.shape[1] is 0:
        return ""
    return ','.join([prefix + str(i) for i in range(v.shape[1])]) + ','


def save_dataset_metadata2(encode, output_path, dataset_type, pos_max_features, pos_min_df, pos_ngram, pos_vec_func,
                           pos_vectorizer, source_filepath, timestamp, tokenize_pos):
    dataset_meta = {
        'timestamp': timestamp,
        'dataset': dataset_type,
        'input': str(source_filepath),
        'pos_max_features': str(pos_max_features),
        'pos_min_df': str(pos_min_df),
        'pos_ngram': str(pos_ngram),
        'pos_tokenizer': tokenize_pos.__name__,
        'pos_vectorizer': str(pos_vectorizer),
        'pos_vectorizer_func': pos_vec_func.__name__,
    }
    if encode:
        dataset_meta['encode_categoricals'] = 'True'
    metadata_output = os.path.join(output_path, "metadata-{}.json".format(timestamp))
    metadata_json = json.dumps(dataset_meta, indent=4, separators=(',', ': '))
    logging.info(metadata_json)
    with open(metadata_output, 'w', encoding='utf-8') as out:
        out.write(metadata_json)
    logging.info("Metadata saved to {}".format(metadata_output))


def save_dataset_metadata(encode, output_path, dataset_type, pos_max_features, pos_min_df, pos_ngram, pos_vec_func,
                          pos_vectorizer, source_filepath, timestamp, word_max_features, word_min_df, word_vectorizer,
                          tokenize_rant, tokenize_pos):
    dataset_meta = {
        'timestamp': timestamp,
        'dataset': dataset_type,
        'input': str(source_filepath),
        'word_max_features': str(word_max_features),
        'pos_max_features': str(pos_max_features),
        'word_min_df': str(word_min_df),
        'pos_min_df': str(pos_min_df),
        'pos_ngram': str(pos_ngram),
        'word_tokenizer': tokenize_rant.__name__,
        'pos_tokenizer': tokenize_pos.__name__,
        'pos_vectorizer': pos_vectorizer.__name__,
        'word_vectorizer': word_vectorizer.__name__,
        'pos_vectorizer_func': pos_vec_func.__name__,
    }
    if encode:
        dataset_meta['encode_categoricals'] = 'True'
    metadata_output = os.path.join(output_path, "metadata-{}.json".format(timestamp))
    metadata_json = json.dumps(dataset_meta, indent=4, separators=(',', ': '))
    logging.info(metadata_json)
    with open(metadata_output, 'w', encoding='utf-8') as out:
        out.write(metadata_json)
    logging.info("Metadata saved to {}".format(metadata_output))


def save_features_json(filepath, feature_names):
    if not feature_names:
        return
    with open(filepath, mode='w') as out:
        out.write(json.dumps(feature_names, ensure_ascii=False, indent=4, separators=(',', ':')))
        logging.info("Saved {} features to JSON ({})".format(len(feature_names), filepath))
