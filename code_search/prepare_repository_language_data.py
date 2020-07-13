from typing import Dict, List

import numpy as np

from code_search.bpevocabulary import BpeVocabulary
from code_search import shared, utils
from code_search.prepare_data import preprocess_doc, build_vocabulary


def prepare_repository_language_corpora(repository_language: shared.RepositoryLanguage):
    language = repository_language[3]
    language_dir = utils.get_repository_language_serialized_data_path(repository_language)
    corpus = utils.load_serialized_corpus(language_dir)
    utils.serialize_corpus((preprocess_doc(doc, language) for doc in corpus), language_dir, is_preprocessed=True)

    valid_training_preprocessed_corpus = lambda: (
        doc for doc in utils.load_serialized_corpus(language_dir, is_preprocessed=True) if len(doc['query_tokens']) > 0)
    docs_data_sets = np.random.choice(
        shared.DATA_SETS, utils.len_generator(valid_training_preprocessed_corpus()), p=shared.DATA_SETS_SPLIT)

    for set_ in shared.DATA_SETS:
        utils.serialize_corpus(
            (doc for doc, doc_set in zip(valid_training_preprocessed_corpus(), docs_data_sets) if doc_set == set_),
            language_dir, is_preprocessed=True, set_=set_)


def inverse_dict(dict_: Dict[str, int]) -> Dict[int, str]:
    return {value: key for key, value in dict_.items()}


def sort_inverse_vocab(vocab_dict: Dict[int, str]) -> List[str]:
    return [value for _, value in sorted(vocab_dict.items())]


def merge_vocabularies(base_vocabulary: BpeVocabulary, additional_vocabulary: BpeVocabulary) -> BpeVocabulary:
    sorted_base_words = sort_inverse_vocab(base_vocabulary.inverse_word_vocab)
    additional_words = list(
        set(additional_vocabulary.inverse_word_vocab.values()) - set(sorted_base_words))
    words = sorted_base_words + additional_words

    sorted_base_bpes = sort_inverse_vocab(base_vocabulary.inverse_bpe_vocab)
    additional_bpes = list(
        set(additional_vocabulary.inverse_bpe_vocab.values()) - set(sorted_base_bpes))
    bpes = sorted_base_bpes + additional_bpes

    merged_vocab = list(
        enumerate([(word, True) for word in words] + [(bpe, False) for bpe in bpes]))

    merged_word_vocab = {word: idx for idx, (word, is_word) in merged_vocab if is_word}
    merged_bpe_vocab = {bpe: idx for idx, (bpe, is_word) in merged_vocab if not is_word}

    base_vocabulary.word_vocab = merged_word_vocab
    base_vocabulary.bpe_vocab = merged_bpe_vocab
    base_vocabulary.inverse_word_vocab = inverse_dict(merged_word_vocab)
    base_vocabulary.inverse_bpe_vocab = inverse_dict(merged_bpe_vocab)
    base_vocabulary.word_vocab_size = len(merged_word_vocab)
    base_vocabulary.bpe_vocab_size = len(merged_bpe_vocab)
    base_vocabulary.vocab_size = base_vocabulary.word_vocab_size + base_vocabulary.bpe_vocab_size
    return base_vocabulary


def prepare_repository_language_vocabularies(
        repository_language: shared.RepositoryLanguage, code_vocab_size: int, query_vocab_size: int):
    language = repository_language[3]
    repository_language_dir = utils.get_repository_language_serialized_data_path(repository_language)
    base_language_dir = utils.get_base_language_serialized_data_path(language)

    for type_ in ['code', 'query']:
        vocab_size = code_vocab_size if type_ == 'code' else query_vocab_size
        base_language_vocabulary = utils.load_serialized_vocabulary(base_language_dir, type_)
        repository_language_vocabulary = build_vocabulary(repository_language_dir, type_, vocab_size)
        merged_repository_language_vocabulary = merge_vocabularies(
            base_language_vocabulary, repository_language_vocabulary)

        utils.serialize_vocabulary(merged_repository_language_vocabulary, repository_language_dir, type_)


def prepare_repository_language_type_embedding_weights(
        base_language_dir: str, repository_language_dir: str, type_: str):
    model = utils.load_serialized_model_weights(base_language_dir, train_model.get_model())

    base_vocab = utils.load_serialized_vocabulary(base_language_dir, type_)
    base_embedding_weights = model.get_layer(f'{type_}_embedding').get_weights()[0]

    repository_vocab = utils.load_serialized_vocabulary(repository_language_dir, type_)
    repository_embedding_weights = np.random.normal(size=(repository_vocab.vocab_size, shared.EMBEDDING_SIZE))

    sorted_base_words = sort_inverse_vocab(base_vocab.inverse_word_vocab)
    sorted_base_bpes = sort_inverse_vocab(base_vocab.inverse_bpe_vocab)

    word_indices = [base_vocab.word_vocab[word] for word in sorted_base_words]
    mapped_word_indices = [repository_vocab.word_vocab[word] for word in sorted_base_words]

    bpe_indices = [base_vocab.bpe_vocab[bpe] for bpe in sorted_base_bpes]
    mapped_bpe_indices = [repository_vocab.bpe_vocab[bpe] for bpe in sorted_base_bpes]

    repository_embedding_weights[mapped_word_indices, :] = base_embedding_weights[word_indices, :]
    repository_embedding_weights[mapped_bpe_indices, :] = base_embedding_weights[bpe_indices, :]

    utils.serialize_embedding_weights(repository_embedding_weights, repository_language_dir, type_)


def prepare_repository_language_embedding_weights(repository_language: shared.RepositoryLanguage):
    language = repository_language[3]
    repository_language_dir = utils.get_repository_language_serialized_data_path(repository_language)
    base_language_dir = utils.get_base_language_serialized_data_path(language)

    prepare_repository_language_type_embedding_weights(base_language_dir, repository_language_dir, 'code')
    prepare_repository_language_type_embedding_weights(base_language_dir, repository_language_dir, 'query')
