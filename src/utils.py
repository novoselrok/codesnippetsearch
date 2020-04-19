import pickle
import json
import itertools


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
