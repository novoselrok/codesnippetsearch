import argparse
import itertools
from typing import List, Iterable, Callable
from collections import Counter

import numpy as np
from code_search.bpe_vocabulary import BpeVocabulary, merge_vocabularies

from code_search import shared, utils, torch_utils
from code_search.model import get_base_language_model, get_base_language_model_for_evaluation
from code_search.data_manager import DataManager, get_base_languages_data_manager
from code_search.preprocessing_tokens import preprocess_code_tokens, preprocess_query_tokens, remove_inline_comments, \
    extract_sub_tokens


def get_query_tokens(docstring_tokens: List[str], identifier: str):
    query_tokens = list(utils.flatten(preprocess_query_tokens(docstring_tokens)))
    if len(query_tokens) > 0:
        return query_tokens
    elif identifier and len(identifier) >= shared.MIN_FUNC_NAME_QUERY_LENGTH:
        return extract_sub_tokens(identifier)

    return []


def preprocess_doc(doc, language: str):
    identifier = doc['identifier']
    docstring_tokens = doc['docstring_tokens']
    code_tokens = doc['code_tokens']

    return {
        # func_name and url are needed for evaluation
        'identifier': identifier,
        'url': doc.get('url'),
        'query_tokens': get_query_tokens(docstring_tokens, identifier),
        'code_tokens': list(
            utils.flatten(preprocess_code_tokens(remove_inline_comments(language, code_tokens)))),
    }


def build_vocabulary(tokens: Iterable[str], vocabulary_size: int, pct_bpe: float):
    vocabulary = BpeVocabulary(vocab_size=vocabulary_size, pct_bpe=pct_bpe)
    vocabulary.fit(Counter(tokens))
    return vocabulary


def pad_encode_seqs(
        seqs: shared.TokensGenerator,
        max_length: int,
        vocabulary: BpeVocabulary,
        preprocess_tokens_fn: Callable[[Iterable[str]], shared.TokensGenerator]) -> np.ndarray:
    encoded_seqs = vocabulary.transform(
        (utils.flatten(preprocess_tokens_fn(seq)) for seq in seqs), fixed_length=max_length)
    return np.array(list(encoded_seqs))


def pad_encode_query(data_manager: DataManager, query: str, max_query_seq_length: int):
    return pad_encode_seqs(
        (seq.split(' ') for seq in [query]),
        max_query_seq_length,
        data_manager.get_query_vocabulary(),
        preprocess_query_tokens)


def keep_valid_seqs(padded_encoded_code_seqs: np.ndarray, padded_encoded_query_seqs: np.ndarray):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_query_seqs = padded_encoded_query_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_query_seqs
    return padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_query_seqs[valid_seqs_indices, :]


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
        utils.map_method(
            self, 'prepare_language_corpus', prepare_language_corpus_args, num_processes=self.num_processes)

    def prepare_vocabularies(
            self, code_vocabulary_size: int, code_pct_bpe: float, query_vocabulary_size: int, query_pct_bpe: float):
        prepare_language_vocabulary_args = ((language, code_vocabulary_size, code_pct_bpe)
                                            for language in self.languages)
        utils.map_method(
            self, 'prepare_language_vocabulary', prepare_language_vocabulary_args, num_processes=self.num_processes)

        self.prepare_query_vocabulary(query_vocabulary_size, query_pct_bpe)

    def prepare_seqs(self, code_seq_max_length: int, query_seq_max_length: int):
        prepare_language_seqs_args = (
            (language, code_seq_max_length, query_seq_max_length, set_)
            for language, set_ in itertools.product(self.languages, shared.DataSet.sets()))
        utils.map_method(self, 'prepare_language_seqs', prepare_language_seqs_args, num_processes=self.num_processes)

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


class RepositoryDataPreparer(DataPreparer):
    def __init__(
            self,
            data_manager: DataManager,
            base_data_manager: DataManager,
            languages: List[str],
            seed: int = 0,
            verbose: bool = True):
        super().__init__(data_manager, languages, verbose=verbose)
        self.base_data_manager = base_data_manager
        self.seed = seed

    @staticmethod
    def _is_valid_training_doc(doc):
        return len(get_query_tokens(doc['docstring_tokens'], doc['identifier'])) > 0

    def prepare(self, **kwargs):
        self.split_corpora_into_sets()
        self.prepare_corpora()
        self.prepare_vocabularies(
            kwargs['code_vocabulary_size'],
            kwargs['code_pct_bpe'],
            kwargs['query_vocabulary_size'],
            kwargs['query_pct_bpe'])
        self.merge_vocabularies()
        self.prepare_embedding_weights()
        self.prepare_seqs(kwargs['code_seq_max_length'], kwargs['query_seq_max_length'])

    @staticmethod
    def _get_embedding_weights(
            base_vocabulary: BpeVocabulary,
            repository_vocabulary: BpeVocabulary,
            base_embedding_weights: np.ndarray):

        sorted_base_words = utils.get_values_sorted_by_key(base_vocabulary.inverse_word_vocab)
        word_indices = [base_vocabulary.word_vocab[word] for word in sorted_base_words]
        mapped_word_indices = [repository_vocabulary.word_vocab[word] for word in sorted_base_words]

        sorted_base_bpes = utils.get_values_sorted_by_key(base_vocabulary.inverse_bpe_vocab)
        bpe_indices = [base_vocabulary.bpe_vocab[bpe] for bpe in sorted_base_bpes]
        mapped_bpe_indices = [repository_vocabulary.bpe_vocab[bpe] for bpe in sorted_base_bpes]

        embedding_weights = np.random.normal(size=(repository_vocabulary.vocab_size, base_embedding_weights.shape[1]))
        embedding_weights[mapped_word_indices, :] = base_embedding_weights[word_indices, :]
        embedding_weights[mapped_bpe_indices, :] = base_embedding_weights[bpe_indices, :]

        return embedding_weights

    def _get_base_model(self):
        return get_base_language_model_for_evaluation(self.base_data_manager)

    def split_corpora_into_sets(self):
        utils.map_method(self,
                         'split_language_corpus_into_sets',
                         ((language,) for language in self.languages),
                         num_processes=self.num_processes)

    def split_language_corpus_into_sets(self, language: str):
        corpus = lambda: (doc for doc in self.data_manager.get_language_corpus(language, shared.DataSet.ALL)
                          if RepositoryDataPreparer._is_valid_training_doc(doc))

        split_data_sets = shared.DataSet.split_data_sets()
        rnd = np.random.RandomState(self.seed)
        data_set_per_doc = rnd.choice(
            split_data_sets,
            utils.len_generator(corpus()),
            p=[shared.DATA_SETS_SPLIT_RATIO[data_set] for data_set in split_data_sets])

        for set_ in split_data_sets:
            set_corpus = (doc for doc, doc_set in zip(corpus(), data_set_per_doc) if doc_set == set_)
            self.data_manager.save_language_corpus(set_corpus, language, set_)

    def merge_vocabularies(self):
        self.merge_query_vocabularies()
        utils.map_method(self,
                         'merge_language_vocabularies',
                         ((language,) for language in self.languages),
                         num_processes=self.num_processes)

    def merge_language_vocabularies(self, language: str):
        base_language_vocabulary = self.base_data_manager.get_language_vocabulary(language)
        repository_language_vocabulary = self.data_manager.get_language_vocabulary(language)
        merged_vocabulary = merge_vocabularies(base_language_vocabulary, repository_language_vocabulary)
        self.data_manager.save_language_vocabulary(merged_vocabulary, language)

    def merge_query_vocabularies(self):
        base_query_vocabulary = self.base_data_manager.get_query_vocabulary()
        repository_query_vocabulary = self.data_manager.get_query_vocabulary()
        merged_vocabulary = merge_vocabularies(base_query_vocabulary, repository_query_vocabulary)
        self.data_manager.save_query_vocabulary(merged_vocabulary)

    def prepare_embedding_weights(self):
        self.prepare_query_embedding_weights()
        utils.map_method(
            self,
            'prepare_language_embedding_weights',
            ((language,) for language in self.languages),
            num_processes=self.num_processes)

    def prepare_query_embedding_weights(self):
        base_query_vocabulary = self.base_data_manager.get_query_vocabulary()
        repository_query_vocabulary = self.data_manager.get_query_vocabulary()

        base_embedding_weights = torch_utils.torch_gpu_to_np(
            self._get_base_model().get_query_embedding_weights().detach())

        embedding_weights = self._get_embedding_weights(
            base_query_vocabulary, repository_query_vocabulary, base_embedding_weights)

        self.data_manager.save_query_embedding_weights(embedding_weights)

    def prepare_language_embedding_weights(self, language: str):
        base_language_vocabulary = self.base_data_manager.get_language_vocabulary(language)
        repository_language_vocabulary = self.data_manager.get_language_vocabulary(language)

        base_embedding_weights = torch_utils.torch_gpu_to_np(
            self._get_base_model().get_language_embedding_weights(language).detach())

        embedding_weights = self._get_embedding_weights(
            base_language_vocabulary, repository_language_vocabulary, base_embedding_weights)

        self.data_manager.save_language_embedding_weights(embedding_weights, language)


def main():
    parser = argparse.ArgumentParser(description='Prepare base language data before training the code search model.')
    utils.add_bool_arg(parser, 'prepare-all', default=True)
    utils.add_bool_arg(parser, 'prepare-corpora', default=False)
    utils.add_bool_arg(parser, 'prepare-vocabularies', default=False)
    utils.add_bool_arg(parser, 'prepare-seqs', default=False)
    args = vars(parser.parse_args())

    data_manager = get_base_languages_data_manager()
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
