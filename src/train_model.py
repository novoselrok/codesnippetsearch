import random
import time

import numpy as np
import tensorflow as tf
from keras import callbacks, layers, Model, optimizers
from more_itertools import chunked
from scipy.spatial.distance import cdist

from keras_utils import ZeroMaskedEntries, mask_aware_mean, mask_aware_mean_output_shape
from shared import CODE_VOCABULARY_SIZE, DOCSTRING_VOCABULARY_SIZE, CODE_MAX_SEQ_LENGTH, DOCSTRING_MAX_SEQ_LENGTH, \
    EMBEDDING_SIZE

random.seed(123)
np.random.seed(123)


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


def get_code_input_and_embedding_layer():
    code_input = layers.Input(shape=(CODE_MAX_SEQ_LENGTH,), name='code_input')
    code_embedding = layers.Embedding(
        input_length=CODE_MAX_SEQ_LENGTH,
        input_dim=CODE_VOCABULARY_SIZE,
        output_dim=EMBEDDING_SIZE,
        name='code_embedding',
        mask_zero=True)(code_input)
    code_embedding = ZeroMaskedEntries()(code_embedding)
    code_embedding = layers.Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='code_embedding_mean')(code_embedding)

    return code_input, code_embedding


def get_docstring_input_and_embedding_layer():
    docstring_input = layers.Input(shape=(DOCSTRING_MAX_SEQ_LENGTH,), name='docstring_input')
    docstring_embedding = layers.Embedding(
        input_length=DOCSTRING_MAX_SEQ_LENGTH,
        input_dim=DOCSTRING_VOCABULARY_SIZE,
        output_dim=EMBEDDING_SIZE,
        name='docstring_embedding',
        mask_zero=True)(docstring_input)
    docstring_embedding = ZeroMaskedEntries()(docstring_embedding)
    docstring_embedding = layers.Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='docstring_embedding_mean')(docstring_embedding)

    return docstring_input, docstring_embedding


def cosine_similarity(x):
    code_embedding, docstring_embedding = x
    docstring_norms = tf.norm(docstring_embedding, axis=-1, keepdims=True) + 1e-10
    code_norms = tf.norm(code_embedding, axis=-1, keepdims=True) + 1e-10
    return tf.matmul(
        docstring_embedding / docstring_norms, code_embedding / code_norms, transpose_a=False, transpose_b=True)


def cosine_loss(_, cosine_similarity_matrix):
    neg_matrix = tf.linalg.diag(tf.fill(dims=[tf.shape(cosine_similarity_matrix)[0]], value=float('-inf')))

    # Distance between docstring and code snippet should be as small as possible
    diagonal_cosine_distance = 1. - tf.linalg.diag_part(cosine_similarity_matrix)
    # Max. similarity between docstring and non-corresponding code snippet should be as small as possible
    max_positive_non_diagonal_similarity_in_row = tf.reduce_max(
        tf.nn.relu(cosine_similarity_matrix + neg_matrix), axis=-1)

    # Combined distance and similarity should be as small as possible as well
    per_sample_loss = tf.maximum(0., diagonal_cosine_distance + max_positive_non_diagonal_similarity_in_row)
    return tf.reduce_mean(per_sample_loss)


def get_model() -> Model:
    code_input, code_embedding = get_code_input_and_embedding_layer()
    docstring_input, docstring_embedding = get_docstring_input_and_embedding_layer()

    merge_layer = layers.Lambda(cosine_similarity, name='cosine_similarity')([
        code_embedding, docstring_embedding
    ])

    model = Model(inputs=[code_input, docstring_input], outputs=merge_layer)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss=cosine_loss)
    return model


def generate_batch(padded_encoded_code_seqs, padded_encoded_docstring_seqs, batch_size: int):
    n_samples = padded_encoded_code_seqs.shape[0]

    shuffled_indices = np.arange(0, n_samples)
    np.random.shuffle(shuffled_indices)
    padded_encoded_code_seqs = padded_encoded_code_seqs[shuffled_indices, :]
    padded_encoded_docstring_seqs = padded_encoded_docstring_seqs[shuffled_indices, :]

    idx = 0
    while True:
        end_idx = min(idx + batch_size, n_samples)
        n_batch_samples = min(batch_size, end_idx - idx)

        batch_code_seqs = padded_encoded_code_seqs[idx:end_idx, :]
        batch_docstring_seqs = padded_encoded_docstring_seqs[idx:end_idx, :]

        yield {'code_input': batch_code_seqs, 'docstring_input': batch_docstring_seqs}, np.zeros(n_batch_samples)

        idx += n_batch_samples
        if idx >= n_samples:
            idx = 0


def train(language, verbose=True):
    model = get_model()
    if verbose:
        print(model.summary())

    train_code_seqs = np.load(f'../saved_seqs/{language}_train_code_seqs.npy')
    train_docstring_seqs = np.load(f'../saved_seqs/{language}_train_docstring_seqs.npy')

    valid_code_seqs = np.load(f'../saved_seqs/{language}_valid_code_seqs.npy')
    valid_docstring_seqs = np.load(f'../saved_seqs/{language}_valid_docstring_seqs.npy')

    num_samples = train_code_seqs.shape[0]
    batch_size = 1000
    model.fit_generator(
        generate_batch(train_code_seqs, train_docstring_seqs, batch_size=batch_size),
        epochs=200,
        steps_per_epoch=num_samples // batch_size,
        verbose=2 if verbose else -1,
        callbacks=[
            MrrEarlyStopping(valid_code_seqs, valid_docstring_seqs, patience=25)
        ]
    )
    model_serialize_path = f'../saved_models/embedding_model_{int(time.time())}.hdf5'
    model.save(model_serialize_path)


if __name__ == '__main__':
    train('python')
