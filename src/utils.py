import os
import pickle
import json
import itertools

import shared


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def flatten(iterable):
    return itertools.chain.from_iterable(iterable)


def load_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def get_raw_docs(files):
    docs = []
    for file_path in files:
        with open(file_path, encoding='utf-8') as f:
            docs.extend([json.loads(line) for line in f.readlines()])
    return docs


def iter_jsonl(file_path):
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def get_pickled_object(serialize_path):
    with open(serialize_path, 'rb') as f:
        return pickle.load(f)


def pickle_object(obj, serialize_path):
    with open(serialize_path, 'wb') as f:
        pickle.dump(obj, f)


def get_code_data_path(filename):
    return os.path.join(shared.CODE_DATA_DIR, filename)


def get_saved_vocabulary_path(filename):
    return os.path.join(shared.SAVED_VOCABULARIES_DIR, filename)


def get_saved_seqs_path(filename):
    return os.path.join(shared.SAVED_SEQS_DIR, filename)


def get_saved_model_path(filename):
    return os.path.join(shared.SAVED_MODELS_DIR, filename)


def get_saved_docs_path(filename):
    return os.path.join(shared.SAVED_DOCS_DIR, filename)
