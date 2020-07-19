import os
from typing import Iterable

import numpy as np

from code_search import shared, serialize, utils


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

    def _get_query_embedding_weights_path(self) -> str:
        return os.path.join(self.base_dir, shared.SERIALIZED_EMBEDDING_WEIGHTS.format(type_=shared.DataType.QUERY))

    def _get_language_embedding_weights_path(self, language: str):
        return os.path.join(
            self._get_language_dir(language), shared.SERIALIZED_EMBEDDING_WEIGHTS.format(type_=shared.DataType.CODE))

    def _get_torch_model_path(self):
        return os.path.join(self.base_dir, shared.SERIALIZED_MODEL_FILENAME)

    def _get_language_code_embeddings_path(self, language: str):
        return os.path.join(self._get_language_dir(language), shared.SERIALIZED_CODE_EMBEDDINGS_FILENAME)

    def get_language_annoy_index_path(self, language: str):
        return os.path.join(self._get_language_dir(language), shared.SERIALIZED_ANNOY_INDEX_FILENAME)

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

    def save_torch_model(self, model):
        serialize.serialize(model.state_dict(), 'torch', self._get_torch_model_path())

    def get_torch_model(self, model):
        model.load_state_dict(serialize.load('torch', self._get_torch_model_path()))
        return model

    def get_language_code_embeddings(self, language: str):
        return serialize.load('numpy', self._get_language_code_embeddings_path(language))

    def save_language_code_embeddings(self, code_embeddings: np.ndarray, language: str):
        return serialize.serialize(code_embeddings, 'numpy', self._get_language_code_embeddings_path(language))

    def get_language_annoy_index(self, annoy_index, language: str):
        return serialize.load('annoy', self.get_language_annoy_index_path(language), annoy_index=annoy_index)

    def save_language_annoy_index(self, annoy_index, language: str):
        return serialize.serialize(annoy_index, 'annoy', self.get_language_annoy_index_path(language))

    def get_language_embedding_weights(self, language: str):
        return serialize.load('numpy', self._get_language_embedding_weights_path(language))

    def get_query_embedding_weights(self):
        return serialize.load('numpy', self._get_query_embedding_weights_path())

    def save_language_embedding_weights(self, embedding_weights: np.ndarray, language: str):
        serialize.serialize(embedding_weights, 'numpy', self._get_language_embedding_weights_path(language))

    def save_query_embedding_weights(self, embedding_weights: np.ndarray):
        serialize.serialize(embedding_weights, 'numpy', self._get_query_embedding_weights_path())


def get_base_languages_data_manager() -> DataManager:
    return DataManager(shared.BASE_LANGUAGES_DIR)


def get_repository_data_manager(organization: str, name: str) -> DataManager:
    return DataManager(utils.get_repository_directory(organization, name))
