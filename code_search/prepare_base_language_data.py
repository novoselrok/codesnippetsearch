import argparse
import itertools
import os
from multiprocessing import Pool
from typing import List, Iterable, Callable

import numpy as np
from bpevocabulary import BpeVocabulary

from code_search import shared, utils, serialize
from code_search.prepare_data import preprocess_doc, build_vocabulary, preprocess_query_tokens
from code_search.preprocessing_tokens import preprocess_code_tokens


def _multiprocess_map_method(args):
    obj, method_name, arg = args
    method = getattr(obj, method_name)
    method(*arg)


def map_method(obj, method_name: str, args: Iterable, num_processes=4):
    if num_processes > 1:
        with Pool(num_processes) as p:
            p.map(_multiprocess_map_method, ((obj, method_name, arg) for arg in args))
    else:
        map(lambda arg: getattr(obj, method_name)(*arg), args)


def pad_encode_seqs(
        seqs: shared.TokensGenerator,
        max_length: int,
        vocabulary: BpeVocabulary,
        preprocess_tokens_fn: Callable[[Iterable[str]], shared.TokensGenerator]) -> np.ndarray:
    encoded_seqs = vocabulary.transform(
        (utils.flatten(preprocess_tokens_fn(seq)) for seq in seqs), fixed_length=max_length)
    return np.array(list(encoded_seqs))


def keep_valid_seqs(padded_encoded_code_seqs: np.ndarray, padded_encoded_query_seqs: np.ndarray):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_query_seqs = padded_encoded_query_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_query_seqs
    return padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_query_seqs[valid_seqs_indices, :]


class DataManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _get_language_dir(self, language: str) -> str:
        language_dir = os.path.join(self.base_dir, language)
        os.makedirs(language_dir, exist_ok=True)
        return language_dir

    def _get_language_corpus_path(self, language: str, set_: shared.DataSet) -> str:
        filename = shared.SERIALIZED_CORPUS_FILENAME.format(set_=set_)
        return os.path.join(self._get_language_dir(language), filename)

    def _get_preprocessed_language_corpus_path(self, language: str, set_: shared.DataSet) -> str:
        filename = shared.SERIALIZED_PREPROCESSED_CORPUS_FILENAME.format(set_=set_)
        return os.path.join(self._get_language_dir(language), filename)

    def _get_language_vocabulary_path(self, language: str) -> str:
        return os.path.join(
            self._get_language_dir(language), shared.SERIALIZED_VOCABULARY_FILENAME.format(type_=shared.DataType.CODE))

    def _get_query_vocabulary_path(self) -> str:
        return os.path.join(self.base_dir, shared.SERIALIZED_VOCABULARY_FILENAME.format(type_=shared.DataType.QUERY))

    def _get_language_seqs_path(self, language: str, type_: shared.DataType, set_: shared.DataSet) -> str:
        filename = shared.SERIALIZED_SEQS_FILENAME.format(type_=type_, set_=set_)
        return os.path.join(self._get_language_dir(language), filename)

    def get_language_corpus(self, language: str, set_: shared.DataSet):
        return serialize.load('jsonl-gzip', self._get_language_corpus_path(language, set_))

    def save_language_corpus(self, corpus: Iterable, language: str, set_: shared.DataSet):
        serialize.serialize(corpus, 'jsonl-gzip', self._get_language_corpus_path(language, set_))

    def save_preprocessed_language_corpus(self, corpus: Iterable, language: str, set_: shared.DataSet):
        serialize.serialize(corpus, 'jsonl-gzip', self._get_preprocessed_language_corpus_path(language, set_))

    def get_preprocessed_language_corpus(self, language: str, set_: shared.DataSet):
        return serialize.load('jsonl-gzip', self._get_preprocessed_language_corpus_path(language, set_))

    def get_language_vocabulary(self, language: str):
        return serialize.load('pickle', self._get_language_vocabulary_path(language))

    def save_language_vocabulary(self, vocabulary, language: str):
        serialize.serialize(vocabulary, 'pickle', self._get_language_vocabulary_path(language))

    def get_query_vocabulary(self):
        return serialize.load('pickle', self._get_query_vocabulary_path())

    def save_query_vocabulary(self, vocabulary):
        serialize.serialize(vocabulary, 'pickle', self._get_query_vocabulary_path())

    def get_language_seqs(self, language: str, type_: shared.DataType, set_: shared.DataSet):
        return serialize.load('numpy', self._get_language_seqs_path(language, type_, set_))

    def save_language_seqs(self, seqs: np.ndarray, language: str, type_: shared.DataType, set_: shared.DataSet):
        serialize.serialize(seqs, 'numpy', self._get_language_seqs_path(language, type_, set_))


class DataPreparer:
    def __init__(self, data_manager: DataManager, languages: List[str], num_processes=1, verbose=False):
        self.data_manager = data_manager
        self.languages = languages
        self.num_processes = num_processes
        self.verbose = verbose

    def prepare(self, **kwargs):
        self.prepare_corpora()
        self.prepare_vocabularies(
            kwargs['code_vocabulary_size'],
            kwargs['code_pct_bpe'],
            kwargs['query_vocabulary_size'],
            kwargs['query_pct_bpe'])
        self.prepare_seqs(kwargs['code_seq_max_length'], kwargs['query_seq_max_length'])

    def prepare_corpora(self):
        prepare_language_corpus_args = itertools.product(self.languages, shared.DataSet.sets())
        map_method(self, 'prepare_language_corpus', prepare_language_corpus_args, num_processes=self.num_processes)

    def prepare_vocabularies(
            self, code_vocabulary_size: int, code_pct_bpe: float, query_vocabulary_size: int, query_pct_bpe: float):
        prepare_language_vocabulary_args = ((language, code_vocabulary_size, code_pct_bpe)
                                            for language in self.languages)
        map_method(
            self, 'prepare_language_vocabulary', prepare_language_vocabulary_args, num_processes=self.num_processes)

        self.prepare_query_vocabulary(query_vocabulary_size, query_pct_bpe)

    def prepare_seqs(self, code_seq_max_length: int, query_seq_max_length: int):
        prepare_language_seqs_args = (
            (language, code_seq_max_length, query_seq_max_length, set_)
            for language, set_ in itertools.product(self.languages, shared.DataSet.sets()))
        map_method(self, 'prepare_language_seqs', prepare_language_seqs_args, num_processes=self.num_processes)

    def prepare_language_corpus(self, language: str, set_: shared.DataSet):
        if self.verbose:
            print(f'Preparing language corpus: {language}, {set_}')

        corpus = self.data_manager.get_language_corpus(language, set_)
        preprocessed_corpus = (preprocess_doc(doc, language) for doc in corpus)
        self.data_manager.save_preprocessed_language_corpus(preprocessed_corpus, language, set_)

    def prepare_language_vocabulary(self, language: str, vocabulary_size: int, pct_bpe: float):
        if self.verbose:
            print(f'Preparing language vocabulary: {language}')

        corpus = self.data_manager.get_preprocessed_language_corpus(language, set_=shared.DataSet.TRAIN)
        tokens = utils.flatten(doc['code_tokens'] for doc in corpus)
        vocabulary = build_vocabulary(tokens, vocabulary_size, pct_bpe)
        self.data_manager.save_language_vocabulary(vocabulary, language)

    def prepare_query_vocabulary(self, vocabulary_size: int, pct_bpe: float):
        if self.verbose:
            print('Preparing query vocabulary')

        corpora = utils.flatten(self.data_manager.get_preprocessed_language_corpus(language, set_=shared.DataSet.TRAIN)
                                for language in self.languages)
        tokens = utils.flatten(doc['query_tokens'] for doc in corpora)
        vocabulary = build_vocabulary(tokens, vocabulary_size, pct_bpe)
        self.data_manager.save_query_vocabulary(vocabulary)

    def prepare_language_seqs(
            self, language: str, code_seq_max_length: int, query_seq_max_length: int, set_: shared.DataSet):
        if self.verbose:
            print(f'Preparing language seqs: {language}, {set_}')

        corpus_fn = lambda: self.data_manager.get_preprocessed_language_corpus(language, set_)
        language_vocabulary = self.data_manager.get_language_vocabulary(language)

        code_seqs = (doc['code_tokens'] for doc in corpus_fn())
        padded_encoded_code_seqs = pad_encode_seqs(
            code_seqs, code_seq_max_length, language_vocabulary, preprocess_code_tokens)

        if set_ == shared.DataSet.ALL:
            # We do not have to prepare query seqs for entire corpus
            self.data_manager.save_language_seqs(padded_encoded_code_seqs, language, shared.DataType.CODE, set_)
        else:
            query_vocabulary = self.data_manager.get_query_vocabulary()
            query_seqs = (doc['query_tokens'] for doc in corpus_fn())
            padded_encoded_query_seqs = pad_encode_seqs(
                query_seqs, query_seq_max_length, query_vocabulary, preprocess_query_tokens)

            # Check for invalid sequences
            padded_encoded_code_seqs, padded_encoded_query_seqs = keep_valid_seqs(
                padded_encoded_code_seqs, padded_encoded_query_seqs)

            self.data_manager.save_language_seqs(padded_encoded_code_seqs, language, shared.DataType.CODE, set_)
            self.data_manager.save_language_seqs(padded_encoded_query_seqs, language, shared.DataType.QUERY, set_)


def main():
    parser = argparse.ArgumentParser(description='Prepare base language data before training the code search model.')
    utils.add_bool_arg(parser, 'prepare-all', default=True)
    utils.add_bool_arg(parser, 'prepare-corpora', default=False)
    utils.add_bool_arg(parser, 'prepare-vocabularies', default=False)
    utils.add_bool_arg(parser, 'prepare-seqs', default=False)
    args = vars(parser.parse_args())

    data_manager = DataManager(shared.BASE_LANGUAGES_DIR)
    data_preparer = DataPreparer(data_manager, shared.LANGUAGES, num_processes=4, verbose=True)

    if args['prepare-all']:
        data_preparer.prepare(
            code_vocabulary_size=shared.CODE_VOCABULARY_SIZE,
            code_pct_bpe=shared.VOCABULARY_PCT_BPE,
            query_vocabulary_size=shared.QUERY_VOCABULARY_SIZE,
            query_pct_bpe=shared.VOCABULARY_PCT_BPE,
            code_seq_max_length=shared.CODE_MAX_SEQ_LENGTH,
            query_seq_max_length=shared.QUERY_MAX_SEQ_LENGTH)
    else:
        if args['prepare-corpora']:
            data_preparer.prepare_corpora()

        if args['prepare-vocabularies']:
            data_preparer.prepare_vocabularies(
                shared.CODE_VOCABULARY_SIZE,
                shared.VOCABULARY_PCT_BPE,
                shared.QUERY_VOCABULARY_SIZE,
                shared.VOCABULARY_PCT_BPE)

        if args['prepare-seqs']:
            data_preparer.prepare_seqs(
                shared.CODE_MAX_SEQ_LENGTH, shared.QUERY_MAX_SEQ_LENGTH)


if __name__ == '__main__':
    main()
