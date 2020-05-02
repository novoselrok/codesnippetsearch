import random
import sys

import numpy as np
import tensorflow as tf
import wandb
from keras import callbacks, layers, Model, optimizers
from wandb.keras import WandbCallback

from code_search import evaluate_model
from code_search import shared
from code_search import utils
from code_search.keras_utils import ZeroMaskedEntries, mask_aware_mean, mask_aware_mean_output_shape


np.random.seed(0)
random.seed(0)


class MrrEarlyStopping(callbacks.EarlyStopping):
    def __init__(self,
                 padded_encoded_code_validation_seqs,
                 padded_encoded_query_validation_seqs,
                 patience=5,
                 batch_size=1000):
        super().__init__(monitor='val_mrr', mode='max', restore_best_weights=True, verbose=True, patience=patience)
        self.padded_encoded_code_validation_seqs = padded_encoded_code_validation_seqs
        self.padded_encoded_query_validation_seqs = padded_encoded_query_validation_seqs
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        mean_mrr = evaluate_model.evaluate_model_mean_mrr(
            self.model,
            self.padded_encoded_code_validation_seqs,
            self.padded_encoded_query_validation_seqs,
            batch_size=self.batch_size)

        print('Mean MRR:', mean_mrr)
        super().on_epoch_end(epoch, {**logs, 'val_mrr': mean_mrr})


def get_code_input_and_embedding_layer():
    code_input = layers.Input(shape=(shared.CODE_MAX_SEQ_LENGTH,), name='code_input')
    code_embedding = layers.Embedding(
        input_length=shared.CODE_MAX_SEQ_LENGTH,
        input_dim=shared.CODE_VOCABULARY_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name='code_embedding',
        mask_zero=True)(code_input)
    code_embedding = ZeroMaskedEntries()(code_embedding)
    code_embedding = layers.Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='code_embedding_mean')(code_embedding)

    return code_input, code_embedding


def get_query_input_and_embedding_layer():
    query_input = layers.Input(shape=(shared.QUERY_MAX_SEQ_LENGTH,), name='query_input')
    query_embedding = layers.Embedding(
        input_length=shared.QUERY_MAX_SEQ_LENGTH,
        input_dim=shared.QUERY_VOCABULARY_SIZE,
        output_dim=shared.EMBEDDING_SIZE,
        name='query_embedding',
        mask_zero=True)(query_input)
    query_embedding = ZeroMaskedEntries()(query_embedding)
    query_embedding = layers.Lambda(
        mask_aware_mean, mask_aware_mean_output_shape, name='query_embedding_mean')(query_embedding)

    return query_input, query_embedding


def get_code_embedding_predictor(model):
    return Model(
        inputs=model.get_layer('code_input').input,
        outputs=model.get_layer('code_embedding_mean').output)


def get_query_embedding_predictor(model):
    return Model(
        inputs=model.get_layer('query_input').input,
        outputs=model.get_layer('query_embedding_mean').output)


def cosine_similarity(x):
    code_embedding, query_embedding = x
    query_norms = tf.norm(query_embedding, axis=-1, keepdims=True) + 1e-10
    code_norms = tf.norm(code_embedding, axis=-1, keepdims=True) + 1e-10
    return tf.matmul(
        query_embedding / query_norms, code_embedding / code_norms, transpose_a=False, transpose_b=True)


def cosine_loss(_, cosine_similarity_matrix):
    neg_matrix = tf.linalg.diag(tf.fill(dims=[tf.shape(cosine_similarity_matrix)[0]], value=float('-inf')))

    # Distance between query and code snippet should be as small as possible
    diagonal_cosine_distance = 1. - tf.linalg.diag_part(cosine_similarity_matrix)
    # Max. similarity between query and non-corresponding code snippet should be as small as possible
    max_positive_non_diagonal_similarity_in_row = tf.reduce_max(
        tf.nn.relu(cosine_similarity_matrix + neg_matrix), axis=-1)

    # Combined distance and similarity should be as small as possible as well
    per_sample_loss = tf.maximum(0., diagonal_cosine_distance + max_positive_non_diagonal_similarity_in_row)
    return tf.reduce_mean(per_sample_loss)


def get_model() -> Model:
    code_input, code_embedding = get_code_input_and_embedding_layer()
    query_input, query_embedding = get_query_input_and_embedding_layer()

    merge_layer = layers.Lambda(cosine_similarity, name='cosine_similarity')([
        code_embedding, query_embedding
    ])

    model = Model(inputs=[code_input, query_input], outputs=merge_layer)
    model.compile(optimizer=optimizers.Adam(learning_rate=shared.LEARNING_RATE), loss=cosine_loss)
    return model


def generate_batch(padded_encoded_code_seqs, padded_encoded_query_seqs, batch_size: int):
    n_samples = padded_encoded_code_seqs.shape[0]

    shuffled_indices = np.arange(0, n_samples)
    np.random.shuffle(shuffled_indices)
    padded_encoded_code_seqs = padded_encoded_code_seqs[shuffled_indices, :]
    padded_encoded_query_seqs = padded_encoded_query_seqs[shuffled_indices, :]

    idx = 0
    while True:
        end_idx = min(idx + batch_size, n_samples)
        n_batch_samples = min(batch_size, end_idx - idx)

        batch_code_seqs = padded_encoded_code_seqs[idx:end_idx, :]
        batch_query_seqs = padded_encoded_query_seqs[idx:end_idx, :]

        yield {'code_input': batch_code_seqs, 'query_input': batch_query_seqs}, np.zeros(n_batch_samples)

        idx += n_batch_samples
        if idx >= n_samples:
            idx = 0


def train(language, model_callbacks, verbose=True):
    model = get_model()

    train_code_seqs = utils.load_cached_seqs(language, 'train', 'code')
    train_query_seqs = utils.load_cached_seqs(language, 'train', 'query')

    valid_code_seqs = utils.load_cached_seqs(language, 'valid', 'code')
    valid_query_seqs = utils.load_cached_seqs(language, 'valid', 'query')

    num_samples = train_code_seqs.shape[0]
    model.fit_generator(
        generate_batch(train_code_seqs, train_query_seqs, batch_size=shared.TRAIN_BATCH_SIZE),
        epochs=200,
        steps_per_epoch=num_samples // shared.TRAIN_BATCH_SIZE,
        verbose=2 if verbose else -1,
        callbacks=[
            MrrEarlyStopping(valid_code_seqs, valid_query_seqs, patience=5)
        ] + model_callbacks
    )

    model.save(utils.get_cached_model_path(language))


if __name__ == '__main__':
    # todo: add --wandb flag
    USE_WANDB = False

    if USE_WANDB:
        if len(sys.argv) == 1:
            # todo: add optional notes
            raise Exception('Running with WANDB enabled requires notes.')

        notes = sys.argv[1]
        # todo: project name should be in env
        wandb.init(project='glorified-code-search', notes=notes, config=shared.get_wandb_config())
        additional_callbacks = [WandbCallback(monitor='val_loss', save_model=False)]
    else:
        additional_callbacks = []

    for language_ in shared.LANGUAGES:
        print(f'Training {language_}')
        train(language_, additional_callbacks)

    # TODO: add --no-evaluate flag
    evaluate_model.evaluate_mean_mrr(USE_WANDB)
    evaluate_model.emit_ndcg_model_predictions(USE_WANDB)
