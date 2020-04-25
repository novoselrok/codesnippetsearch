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


def get_docs(files):
    docs = []
    for file_path in files:
        with open(file_path) as f:
            docs.extend([json.loads(line) for line in f.readlines()])
    return docs


def get_pickled_object(serialize_path):
    with open(serialize_path, 'rb') as f:
        return pickle.load(f)


def get_code_data_path(filename):
    return os.path.join(shared.CODE_DATA_DIR, filename)


def get_saved_vocabulary_path(filename):
    return os.path.join(shared.SAVED_VOCABULARIES_DIR, filename)


def get_saved_seqs_path(filename):
    return os.path.join(shared.SAVED_SEQS_DIR, filename)


def get_saved_model_path(filename):
    return os.path.join(shared.SAVED_MODELS_DIR, filename)
