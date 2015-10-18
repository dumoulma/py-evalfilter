import os


def get_size(filename):
    st = os.stat(filename)
    return st.st_size >> 20
