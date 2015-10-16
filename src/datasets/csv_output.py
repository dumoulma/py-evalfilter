import logging

from datasets.fuman_raw import get_header


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


def generate_header(pos_vects, rants_vects):
    headers = get_header() + ','
    headers += generate_vector_headers(pos_vects, "pos")
    headers += generate_vector_headers(rants_vects, "word")
    headers += "target\n"
    logging.debug("Header: {}".format(headers))
    return headers


def generate_vector_headers(v, prefix):
    if v.shape[1] is 0:
        return ""
    return ','.join([prefix + str(i) for i in range(v.shape[1])]) + ','
