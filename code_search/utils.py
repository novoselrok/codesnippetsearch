import os
import pickle
import json
import itertools
from typing import Iterable, TypeVar

import numpy as np
from annoy import AnnoyIndex

from code_search import shared
from code_search.bpevocabulary import BpeVocabulary


T = TypeVar('T')


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def flatten(iterable: Iterable[Iterable[T]]) -> Iterable[T]:
    return itertools.chain.from_iterable(iterable)


def iter_jsonl(file_path: str):
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(iterable, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in iterable:
            f.write(json.dumps(item) + '\n')


def load_pickled_object(serialize_path: str):
    with open(serialize_path, 'rb') as f:
        return pickle.load(f)


def pickle_object(obj, serialize_path):
    with open(serialize_path, 'wb') as f:
        pickle.dump(obj, f)


def get_raw_doc_path(language: str, set_: str, idx: int) -> str:
    return os.path.join(
        shared.ENV['CODESEARCHNET_DATA_DIR'], language, 'final', 'jsonl', set_, f'{language}_{set_}_{idx}.jsonl')


def get_evaluation_queries():
    with open(os.path.join(shared.ENV['CODESEARCHNET_RESOURCES_DIR'], 'queries.csv'), encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[1:]]


def get_raw_docs(language: str, set_: str):
    if set_ == 'train':
        file_paths = [get_raw_doc_path(language, set_, i) for i in range(shared.LANGUAGES_NUM_FILES[language])]
    else:
        file_paths = [get_raw_doc_path(language, set_, 0)]

    for file_path in file_paths:
        yield from iter_jsonl(file_path)


def get_cached_seqs_path(language: str, set_: str, type_: str) -> str:
    return os.path.join(
        shared.SEQS_CACHE_DIR, shared.SEQS_CACHE_FILENAME.format(language=language, set_=set_, type_=type_))


def load_cached_seqs(language: str, set_: str, type_: str) -> np.ndarray:
    return np.load(get_cached_seqs_path(language, set_, type_))


def cache_seqs(seqs: np.ndarray, language: str, set_: str, type_: str):
    np.save(get_cached_seqs_path(language, set_, type_), seqs)


def get_cached_vocabulary_path(language: str, type_: str) -> str:
    return os.path.join(
        shared.VOCABULARIES_CACHE_DIR, shared.VOCABULARY_CACHE_FILENAME.format(language=language, type_=type_))


def cache_vocabulary(vocabulary: BpeVocabulary, language: str, type_: str):
    pickle_object(vocabulary, get_cached_vocabulary_path(language, type_))


def load_cached_vocabulary(language: str, type_: str) -> BpeVocabulary:
    return load_pickled_object(get_cached_vocabulary_path(language, type_))


def get_cached_docs_path(language: str, set_: str):
    return os.path.join(
        shared.DOCS_CACHE_DIR, shared.DOCS_CACHE_FILENAME.format(language=language, set_=set_))


def cache_docs(docs, language: str, set_: str):
    write_jsonl(docs, get_cached_docs_path(language, set_))


def load_cached_docs(language: str, set_: str):
    return iter_jsonl(get_cached_docs_path(language, set_))


def get_cached_code_embeddings_path(language: str):
    return os.path.join(
        shared.CODE_EMBEDDINGS_CACHE_DIR, shared.CODE_EMBEDDINGS_CACHE_FILENAME.format(language=language))


def cache_code_embeddings(code_embeddings: np.ndarray, language: str):
    np.save(get_cached_code_embeddings_path(language), code_embeddings)


def load_cached_code_embeddings(language: str):
    return np.load(get_cached_code_embeddings_path(language))


def get_cached_model_path(language: str) -> str:
    return os.path.join(
        shared.MODELS_CACHE_DIR, shared.MODEL_CACHE_FILENAME.format(language=language))


def load_cached_model_weights(language: str, model):
    model.load_weights(get_cached_model_path(language), by_name=True)
    return model


def get_annoy_index():
    return AnnoyIndex(shared.EMBEDDING_SIZE, 'angular')


def get_cached_ann_path(language: str) -> str:
    return os.path.join(
        shared.ANNS_CACHE_DIR, shared.ANN_CACHE_FILENAME.format(language=language))


def load_cached_ann(language: str) -> AnnoyIndex:
    ann = get_annoy_index()
    ann.load(get_cached_ann_path(language))
    return ann


def cache_ann(ann: AnnoyIndex, language: str):
    ann.save(get_cached_ann_path(language))
