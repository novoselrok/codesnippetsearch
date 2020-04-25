import glob
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb
from keras import Model
from more_itertools import chunked
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

import prepare_data
import train_model
import utils

random.seed(123)


def get_model_paths_and_languages():
    model_paths = list(glob.glob(utils.get_saved_model_path('*.hdf5')))

    language_to_model_paths = defaultdict(list)

    for model_path in model_paths:
        _, language, ext = os.path.basename(model_path).split('_')
        language_to_model_paths[language].append((int(ext.split('.')[0]), model_path))

    for language, language_model_paths in language_to_model_paths.items():
        yield list(sorted(language_model_paths))[-1][1], language


def get_code_embedding_predictor(model):
    return Model(
        inputs=model.get_layer('code_input').input,
        outputs=model.get_layer('code_embedding_mean').output)


def get_docstring_embedding_predictor(model):
    return Model(
        inputs=model.get_layer('docstring_input').input,
        outputs=model.get_layer('docstring_embedding_mean').output)


def emit_ndcg_model_predictions(use_wandb=False):
    with open(utils.get_code_data_path('queries.csv'), encoding='utf-8') as f:
        queries = [line.strip() for line in f.readlines()[1:]]
        query_tokens = [line.split(' ') for line in queries]

    predictions = []
    for model_path, language in get_model_paths_and_languages():
        print(f'Evaluating {language}')

        docs = utils.get_pickled_object(utils.get_saved_seqs_path(f'{language}_valid_docs.pckl'))

        model = train_model.get_model()
        model.load_weights(model_path, by_name=True)
        code_embedding_predictor = get_code_embedding_predictor(model)
        docstring_embedding_predictor = get_docstring_embedding_predictor(model)

        train_code_seqs = np.load(utils.get_saved_seqs_path(f'{language}_train_code_seqs.npy'))
        code_embeddings = code_embedding_predictor.predict(train_code_seqs)

        query_seqs = prepare_data.pad_encode_docstring_seqs(language, query_tokens)
        query_embeddings = docstring_embedding_predictor.predict(query_seqs)

        nn = NearestNeighbors(n_neighbors=100, metric='cosine', n_jobs=-1)
        nn.fit(code_embeddings)

        _dists, nearest_neigh_indices = nn.kneighbors(query_embeddings)

        for query_idx, query in enumerate(queries):
            for query_nearest_code_idx in nearest_neigh_indices[query_idx, :]:
                predictions.append({
                    'query': query,
                    'language': language,
                    'identifier': docs[query_nearest_code_idx]['identifier'],
                    'url': docs[query_nearest_code_idx]['url'],
                })

    df_predictions = pd.DataFrame(predictions, columns=['query', 'language', 'identifier', 'url'])
    save_path = os.path.join(wandb.run.dir, 'model_predictions.csv') if use_wandb else '../model_predictions.csv'
    df_predictions.to_csv(save_path, index=False)


def evaluate_mean_mrr(
        model, padded_encoded_code_validation_seqs, padded_encoded_docstring_validation_seqs, batch_size=1000):
    code_embedding_predictor = get_code_embedding_predictor(model)
    docstring_embedding_predictor = get_docstring_embedding_predictor(model)

    n_samples = padded_encoded_code_validation_seqs.shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)
    mrrs = []
    for idx_chunk in chunked(indices, batch_size):
        if len(idx_chunk) < batch_size:
            continue

        code_embeddings = code_embedding_predictor.predict(
            padded_encoded_code_validation_seqs[idx_chunk, :])
        docstring_embeddings = docstring_embedding_predictor.predict(
            padded_encoded_docstring_validation_seqs[idx_chunk, :])

        distance_matrix = cdist(docstring_embeddings, code_embeddings, 'cosine')
        correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
        ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
        mrrs.append(np.mean(1.0 / ranks))

    return np.mean(mrrs)


def evaluate(language, model_path):
    model = train_model.get_model()
    model.load_weights(model_path, by_name=True)

    test_code_seqs = np.load(utils.get_saved_seqs_path(f'{language}_valid_code_seqs.npy'))
    test_docstring_seqs = np.load(utils.get_saved_seqs_path(f'{language}_valid_docstring_seqs.npy'))

    print('Test Mean MRR: ', evaluate_mean_mrr(model, test_code_seqs, test_docstring_seqs))


def main():
    for model_path, language in get_model_paths_and_languages():
        print(f'Evaluating {language}')
        evaluate(language, model_path)


if __name__ == '__main__':
    main()
