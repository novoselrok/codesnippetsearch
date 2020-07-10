import pickle
import json
import gzip
from typing import Iterable

import numpy as np


def pickle_load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_serialize(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def jsonl_gzip_load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def jsonl_gzip_serialize(iterable: Iterable, path: str, compress_level: int = 5):
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=compress_level) as f:
        for item in iterable:
            f.write(json.dumps(item) + '\n')


def jsonl_load(path: str):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def jsonl_serialize(iterable: Iterable, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in iterable:
            f.write(json.dumps(item) + '\n')


def numpy_serialize(arr: np.ndarray, path: str):
    return np.save(path, arr)


SERIALIZERS = {
    'pickle': pickle_serialize,
    'numpy': numpy_serialize,
    'jsonl': jsonl_serialize,
    'jsonl-gzip': jsonl_gzip_serialize,
}

LOADERS = {
    'pickle': pickle_load,
    'numpy': np.load,
    'jsonl': jsonl_load,
    'jsonl-gzip': jsonl_gzip_load,
}

EXTENSIONS = {
    'pickle': 'pkl',
    'numpy': 'npy',
    'jsonl': 'jsonl',
    'jsonl-gzip': 'jsonl.gz',
}


def serialize(obj, format_: str, path: str):
    if format_ not in SERIALIZERS:
        raise Exception(f'{format_} is not a valid serialization format')

    SERIALIZERS[format_](obj, f'{path}.{EXTENSIONS[format_]}')


def load(format_: str, path: str):
    if format_ not in LOADERS:
        raise Exception(f'{format_} is not a valid serialization format')

    return LOADERS[format_](f'{path}.{EXTENSIONS[format_]}')
