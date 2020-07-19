import numpy as np

from code_search import shared, utils


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
