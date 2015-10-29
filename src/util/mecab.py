from functools import partial

import MeCab

mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
STOPWORDS = {'の', 'が', 'て', '、', 'する', 'ある', 'です', 'ます', 'た', 'が', 'から', 'れる', 'いる', '「', '\u3000', '」',
             'と', 'くる', 'で', 'ない', 'を', 'に', 'なる', '。', 'だ', 'のに', 'でる', 'は', 'よう', 'も', 'しか', 'いう',
             'う', '・', 'ので', 'けど', 'こと', 'ので', 'など', 'ば', 'すでに', 'によって', 'くらい', 'さ', '－', '一', 'か',
             '。', '\\', 'n', 'etc.', 'etc', 'すぎる', 'これ', 'それ', 'あれ', 'この', 'その', 'あの', 'もっとも', 'もっと',
             'に関し', 'に関して', 'あなたと', 'あなた', 'あなたに', '打ち合せ', '打合せ', 'いただき', 'それぞれに', 'それぞれ',
             'よって', 'によって', 'て', 'として', 'for', 'や', 'のもの', 'そのもの', 'つくれる', 'くれる', '明らか', 'てる'}


def tokenize(field, text, min_length=2):
    text_striped = text.replace('\n', ' ').replace('\\n', ' ').replace('\r', '')
    tokenized = [w.split('\t') for w in mecab.parse(text_striped).split('\n')]
    return [w[field].lower() for w in tokenized if len(w) >= 2 and len(w[field]) >= min_length]


def tokenize_rant(text, min_length=2):
    return partial(tokenize, field=2)(text=text, min_length=min_length)


def tokenize_pos(text):
    return partial(tokenize, field=3)(text=text)
