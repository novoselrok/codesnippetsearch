import pickle
import json
import itertools

from spacy.lang.en.stop_words import STOP_WORDS

from vocabulary import Vocabulary


def get_docs(files):
    docs = []
    for file_path in files:
        with open(file_path) as f:
            docs.extend([json.loads(line) for line in f.readlines()])
    return docs


def build_and_serialize_vocabularies(docs, serialize_path):
    def build_vocabulary(tokens_key, ignored_tokens=None):
        tokens = itertools.chain.from_iterable([doc[tokens_key] for doc in docs])
        tokens = [token.lower() for token in tokens]
        return Vocabulary.create_vocabulary(tokens, ignored_tokens=ignored_tokens)

    code_vocabulary = build_vocabulary('code_tokens', ignored_tokens=set(list(':()[],.?{}=') + ['self', 'and', 'or']))
    docstring_vocabulary = build_vocabulary('docstring_tokens', ignored_tokens=STOP_WORDS)

    with open(serialize_path, 'wb') as f:
        pickle.dump((code_vocabulary, docstring_vocabulary), f)


def get_vocabularies(serialize_path):
    with open(serialize_path, 'rb') as f:
        return pickle.load(f)
