import gc
import itertools
import random
import time

import numpy as np
from keras import callbacks, layers, Model, optimizers
from keras.preprocessing.sequence import pad_sequences
from more_itertools import chunked
from scipy.spatial.distance import cdist

from utils import get_vocabularies, get_docs, build_and_serialize_vocabularies
from keras_utils import ZeroMaskedEntries, mask_aware_mean, mask_aware_mean_output_shape

random.seed(123)
np.random.seed(123)

EMBEDDING_SIZE = 128
MAX_CODE_SEQ_LENGTH = 200
MAX_DOCSTRING_SEQ_LENGTH = 20
VOCABULARY_SIZE = 10000

SAVED_MODELS_PREFIX_PATH = '../saved_models'
VOCABULARIES_SERIALIZE_PATH = '../saved_vocabularies/vocabularies.pckl'


class MrrEarlyStopping(callbacks.EarlyStopping):
    def __init__(self, padded_encoded_code_validation_seqs, padded_encoded_docstring_validation_seqs,
                 patience=5, batch_size=1000):
        super().__init__(monitor='val_mrr', mode='max', restore_best_weights=True, verbose=True, patience=patience)
        self.padded_encoded_code_validation_seqs = padded_encoded_code_validation_seqs
        self.padded_encoded_docstring_validation_seqs = padded_encoded_docstring_validation_seqs
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        code_embedding_predictor = Model(
           inputs=self.model.get_layer('code_input').input,
           outputs=self.model.get_layer('code_embedding_mean').output)

        docstring_embedding_predictor = Model(
           inputs=self.model.get_layer('docstring_input').input,
           outputs=self.model.get_layer('docstring_embedding_mean').output)

        n_samples = self.padded_encoded_code_validation_seqs.shape[0]
        mrrs = []
        for idx_chunk in chunked(list(range(n_samples)), self.batch_size):
            if len(idx_chunk) < self.batch_size:
                continue

            code_embeddings = code_embedding_predictor.predict(
                self.padded_encoded_code_validation_seqs[idx_chunk, :])
            docstring_embeddings = docstring_embedding_predictor.predict(
                self.padded_encoded_docstring_validation_seqs[idx_chunk, :])

            distance_matrix = cdist(docstring_embeddings, code_embeddings, 'cosine')
            correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
            ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
            mrrs.append(np.mean(1.0 / ranks))

        mean_mrr = np.mean(mrrs)
        print('Mean MRR:', mean_mrr)
        super().on_epoch_end(epoch, {**logs, 'val_mrr': mean_mrr})


def get_code_input_and_embedding_layer(code_sequence_size: int, code_vocabulary_size: int, embedding_size: int):
    code_input = layers.Input(shape=(code_sequence_size,), name='code_input')
    code_embedding = layers.Embedding(
        input_length=code_sequence_size,
        input_dim=code_vocabulary_size,
        output_dim=embedding_size,
        name='code_embedding',
        mask_zero=True)(code_input)
    code_embedding = ZeroMaskedEntries()(code_embedding)
    code_embedding = layers.Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='code_embedding_mean')(code_embedding)

    return code_input, code_embedding


def get_docstring_input_and_embedding_layer(
        docstring_sequence_size: int, docstring_vocabulary_size: int, embedding_size: int):
    docstring_input = layers.Input(shape=(docstring_sequence_size,), name='docstring_input')
    docstring_embedding = layers.Embedding(
        input_length=docstring_sequence_size,
        input_dim=docstring_vocabulary_size,
        output_dim=embedding_size,
        name='docstring_embedding',
        mask_zero=True)(docstring_input)
    docstring_embedding = ZeroMaskedEntries()(docstring_embedding)
    docstring_embedding = layers.Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='docstring_embedding_mean')(docstring_embedding)

    return docstring_input, docstring_embedding


def get_model(code_sequence_size: int, docstring_sequence_size: int, code_vocabulary_size: int,
              docstring_vocabulary_size: int, embedding_size: int = 50) -> Model:
    code_input, code_embedding = get_code_input_and_embedding_layer(
        code_sequence_size, code_vocabulary_size, embedding_size)

    docstring_input, docstring_embedding = get_docstring_input_and_embedding_layer(
        docstring_sequence_size, docstring_vocabulary_size, embedding_size)

    # cosine similarity
    merge_layer = layers.Dot(axes=1, normalize=True, name='dot_product')([
        code_embedding, docstring_embedding
    ])

    model = Model(inputs=[code_input, docstring_input], outputs=merge_layer)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mse')
    return model


def generate_batch(padded_encoded_code_seqs, padded_encoded_docstring_seqs,
                   batch_size: int, negative_ratio: float):
    n_samples = padded_encoded_code_seqs.shape[0]

    shuffled_indices = np.arange(0, n_samples)
    np.random.shuffle(shuffled_indices)
    padded_encoded_code_seqs = padded_encoded_code_seqs[shuffled_indices, :]
    padded_encoded_docstring_seqs = padded_encoded_docstring_seqs[shuffled_indices, :]

    idx = 0
    while True:
        end_idx = min(idx + batch_size, n_samples)
        n_positive = min(batch_size, end_idx - idx)
        n_negative = int(n_positive * negative_ratio)

        negative_samples_indices = random.sample(range(n_samples), n_negative * 2)
        code_negative_samples_indices = negative_samples_indices[:n_negative]
        docstring_negative_samples_indices = negative_samples_indices[n_negative:]

        batch_code_seqs = np.concatenate(
            (padded_encoded_code_seqs[idx:end_idx, :],
             padded_encoded_code_seqs[code_negative_samples_indices, :]), axis=0)

        batch_docstring_seqs = np.concatenate(
            (padded_encoded_docstring_seqs[idx:end_idx, :],
             padded_encoded_docstring_seqs[docstring_negative_samples_indices, :]), axis=0)

        target = np.concatenate((np.ones(n_positive), -1 * np.ones(n_negative)))
        yield {'code_input': batch_code_seqs, 'docstring_input': batch_docstring_seqs}, target

        idx += n_positive
        if idx >= n_samples:
            idx = 0


def encode_tokens(docs, tokens_key, vocabulary):
    encoded_seqs = []
    for doc in docs:
        encoded_seqs.append(
            [vocabulary.token_to_id[token] for token in doc[tokens_key]
             if token in vocabulary.token_to_id]
        )
    return encoded_seqs


def pad_encoded_seqs(seqs, maxlen):
    return np.array(pad_sequences(seqs, maxlen=maxlen, padding='post'))


def keep_valid_seqs(padded_encoded_code_seqs, padded_encoded_docstring_seqs):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_docstring_seqs = padded_encoded_docstring_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_docstring_seqs

    return padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_docstring_seqs[valid_seqs_indices, :]


def train(train_docs, validation_docs, verbose=True):
    code_vocabulary, docstring_vocabulary = get_vocabularies(VOCABULARIES_SERIALIZE_PATH)

    padded_encoded_code_seqs = pad_encoded_seqs(
        encode_tokens(train_docs, 'code_tokens', code_vocabulary), MAX_CODE_SEQ_LENGTH)
    padded_encoded_docstring_seqs = pad_encoded_seqs(
        encode_tokens(train_docs, 'docstring_tokens', docstring_vocabulary), MAX_DOCSTRING_SEQ_LENGTH)
    padded_encoded_code_seqs, padded_encoded_docstring_seqs = keep_valid_seqs(
        padded_encoded_code_seqs, padded_encoded_docstring_seqs)

    padded_encoded_code_validation_seqs = pad_encoded_seqs(
        encode_tokens(validation_docs, 'code_tokens', code_vocabulary), MAX_CODE_SEQ_LENGTH)
    padded_encoded_docstring_validation_seqs = pad_encoded_seqs(
        encode_tokens(validation_docs, 'docstring_tokens', docstring_vocabulary), MAX_DOCSTRING_SEQ_LENGTH)
    padded_encoded_code_validation_seqs, padded_encoded_docstring_validation_seqs = keep_valid_seqs(
        padded_encoded_code_validation_seqs, padded_encoded_docstring_validation_seqs)

    model = get_model(
        MAX_CODE_SEQ_LENGTH,
        MAX_DOCSTRING_SEQ_LENGTH,
        VOCABULARY_SIZE,
        VOCABULARY_SIZE,
        embedding_size=EMBEDDING_SIZE
    )
    if verbose:
        print(model.summary())

    del train_docs
    del docstring_vocabulary
    del code_vocabulary
    gc.collect()

    batch_size = 256
    model.fit_generator(
        generate_batch(
            padded_encoded_code_seqs, padded_encoded_docstring_seqs, batch_size=batch_size, negative_ratio=2),
        epochs=200,
        steps_per_epoch=padded_encoded_code_seqs.shape[0] // batch_size,
        verbose=2 if verbose else -1,
        callbacks=[
            MrrEarlyStopping(padded_encoded_code_validation_seqs, padded_encoded_docstring_validation_seqs, patience=25)
        ]
    )
    model_serialize_path = f'{SAVED_MODELS_PREFIX_PATH}/embedding_model_{int(time.time())}.hdf5'
    model.save(model_serialize_path)


def preprocess_docs(docs):
    for doc in docs:
        doc['code_tokens'] = list(itertools.chain.from_iterable(
            [token.split('_') for token in doc['code_tokens']]))
        doc['code_tokens'] = [token.lower() for token in doc['code_tokens']]
        doc['docstring_tokens'] = [token.lower() for token in doc['docstring_tokens']]

    return [doc for doc in docs if len(doc['code_tokens']) > 0 and len(doc['docstring_tokens']) > 0]


def main():
    train_docs = get_docs([f'../data/python_train_{i}.jsonl' for i in range(14)])
    validation_docs = get_docs(['../data/python_valid_0.jsonl'])

    train_docs = preprocess_docs(train_docs)
    validation_docs = preprocess_docs(validation_docs)

    build_and_serialize_vocabularies(train_docs, VOCABULARIES_SERIALIZE_PATH)
    train(train_docs, validation_docs)


if __name__ == '__main__':
    main()
