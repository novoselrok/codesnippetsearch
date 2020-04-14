import random

import numpy as np
from keras import Model
from keras.preprocessing.sequence import pad_sequences

import train_model
from utils import get_vocabularies, get_docs


def get_embedding_predictor(trained_model_path: str, get_input_and_embedding_layers_fn,
                            sequence_size: int, vocabulary_size: int, embedding_size: int = 50):
    input_layer, embedding_layer = get_input_and_embedding_layers_fn(sequence_size, vocabulary_size, embedding_size)
    predictor = Model(inputs=input_layer, outputs=embedding_layer)
    predictor.load_weights(trained_model_path, by_name=True)
    return predictor


def predict(model_path):
    docs = get_docs([f'../data/python_train_{i}.jsonl' for i in range(14)])
    code_vocabulary, docstring_vocabulary = get_vocabularies(train_model.VOCABULARIES_SERIALIZE_PATH)

    code_embedding_predictor = get_embedding_predictor(
        model_path,
        train_model.get_code_input_and_embedding_layer,
        train_model.MAX_CODE_SEQ_LENGTH,
        code_vocabulary.size,
        embedding_size=train_model.EMBEDDING_SIZE)
    docstring_embedding_predictor = get_embedding_predictor(
        model_path,
        train_model.get_docstring_input_and_embedding_layer,
        train_model.MAX_DOCSTRING_SEQ_LENGTH,
        docstring_vocabulary.size,
        embedding_size=train_model.EMBEDDING_SIZE)

    encoded_code_seqs = train_model.encode_tokens(docs, 'code_tokens', code_vocabulary)
    padded_encoded_code_seqs = np.array(pad_sequences(
        encoded_code_seqs, maxlen=train_model.MAX_CODE_SEQ_LENGTH, padding='post'))
    code_embeddings = code_embedding_predictor.predict(padded_encoded_code_seqs)

    encoded_docstring_seqs = train_model.encode_tokens(docs, 'docstring_tokens', docstring_vocabulary)
    padded_encoded_docstring_seqs = np.array(
        pad_sequences(encoded_docstring_seqs, maxlen=train_model.MAX_DOCSTRING_SEQ_LENGTH, padding='post'))
    docstring_embeddings = docstring_embedding_predictor.predict(padded_encoded_docstring_seqs)

    code_embeddings = code_embeddings / np.linalg.norm(code_embeddings, axis=1).reshape((-1, 1))
    docstring_embeddings = docstring_embeddings / np.linalg.norm(docstring_embeddings, axis=1).reshape((-1, 1))

    reciprocal_ranks = []
    for i in range(code_embeddings.shape[0]):
        test_embeddings = np.concatenate(
            (
                code_embeddings[i, :].reshape((1, -1)),
                code_embeddings[random.sample(range(code_embeddings.shape[0]), 999), :]
            ),
            axis=0
        )

        row = np.dot(
            test_embeddings,
            docstring_embeddings[i, :]
        )
        sorted_indices = np.argsort(-1 * row)  # sort descending by cosine similarity

        rank = np.where(sorted_indices == 0)[0][0] + 1
        # print(rank, end=', ')
        reciprocal_rank = 1. / rank
        reciprocal_ranks.append(reciprocal_rank)

        if i % 1000 == 0:
            print(i, np.mean(reciprocal_ranks))
    # print()
    print(np.mean(reciprocal_ranks))
