import logging

from datasets.fuman_raw import get_header


def make_csv_row(x, i, pos_vects, rants_vects) -> str:
    features = ','.join(str(i) for i in x[:-1]) + ','
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


def generate_header(pos_vects, rants_vects):
    header = get_header()
    pos_header = generate_pos_headers(pos_vects)
    words_header = generate_word_headers(rants_vects)
    target_header = "target"
    final_header = ','.join((header, pos_header, words_header, target_header)) + '\n'
    logging.debug("Header: features:{} pos:{} wd:{} final:{}".format(len(header.split(',')),
                                                                     len(pos_header.split(',')),
                                                                     len(words_header.split(',')),
                                                                     len(final_header.split(','))))
    return final_header


def generate_pos_headers(pos_vect):
    return ','.join(["pos" + str(i) for i in range(pos_vect.shape[1])])


def generate_word_headers(word_vect):
    if word_vect.shape[1] is 0:
        return ""
    return ','.join(["wd" + str(i) for i in range(word_vect.shape[1])])
