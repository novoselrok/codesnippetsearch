import gc
import os
import random

import torch
import numpy as np
import pandas as pd
import wandb
from more_itertools import chunked
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from code_search import prepare_data
from code_search import shared
from code_search import train
from code_search import utils
from code_search.build_base_language_code_embeddings import build_code_embeddings

np.random.seed(0)
random.seed(0)


def emit_ndcg_model_predictions(use_wandb=False):
    queries = utils.get_evaluation_queries()

    predictions = []
    for language in shared.LANGUAGES:
        language_dir = utils.get_base_language_serialized_data_path(language)
        print(f'Evaluating {language}')

        evaluation_docs = [{'url': doc['url'], 'identifier': doc['identifier']}
                           for doc in utils.load_serialized_corpus(language_dir)]

        code_embeddings = utils.load_serialized_code_embeddings(language_dir)

        model: train.CodeSearchNet = train.get_model().to(train.device)
        model.load_state_dict(utils.load_serialized_model(shared.BASE_LANGUAGES_DIR))
        model.eval()

        query_seqs = prepare_data.pad_encode_seqs(
            prepare_data.preprocess_query_tokens,
            lambda: (line.split(' ') for line in queries),
            shared.QUERY_MAX_SEQ_LENGTH,
            language_dir,
            'query')
        query_embeddings = model.encode_query(
            torch.from_numpy(query_seqs).to(train.device)).detach().cpu().numpy()

        nn = NearestNeighbors(n_neighbors=150, metric='cosine', n_jobs=-1)
        nn.fit(code_embeddings)
        _, nearest_neighbor_indices = nn.kneighbors(query_embeddings)
        # ann = utils.load_serialized_ann(language_dir)

        for query_idx, query in enumerate(queries):
            # nearest_neighbor_indices = ann.get_nns_by_vector(query_embeddings[query_idx, :], 150)
            for query_nearest_code_idx in nearest_neighbor_indices[query_idx, :]:
                # for query_nearest_code_idx in nearest_neighbor_indices:
                predictions.append({
                    'query': query,
                    'language': language,
                    'identifier': evaluation_docs[query_nearest_code_idx]['identifier'],
                    'url': evaluation_docs[query_nearest_code_idx]['url'],
                })

        del evaluation_docs
        gc.collect()

    df_predictions = pd.DataFrame(predictions, columns=['query', 'language', 'identifier', 'url'])
    save_path = os.path.join(wandb.run.dir, 'model_predictions.csv') if use_wandb else '../model_predictions.csv'
    df_predictions.to_csv(save_path, index=False)


# def evaluate_model_mean_mrr(
#         model, padded_encoded_code_validation_seqs, padded_encoded_query_validation_seqs, batch_size=1000):
#     code_embedding_predictor = train_model.get_code_embedding_predictor(model)
#     query_embedding_predictor = train_model.get_query_embedding_predictor(model)
#
#     n_samples = padded_encoded_code_validation_seqs.shape[0]
#     indices = list(range(n_samples))
#     random.shuffle(indices)
#     mrrs = []
#     for idx_chunk in chunked(indices, batch_size):
#         if len(idx_chunk) < batch_size:
#             continue
#
#         code_embeddings = code_embedding_predictor.predict(padded_encoded_code_validation_seqs[idx_chunk, :])
#         query_embeddings = query_embedding_predictor.predict(padded_encoded_query_validation_seqs[idx_chunk, :])
#
#         distance_matrix = cdist(query_embeddings, code_embeddings, 'cosine')
#         correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
#         ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
#         mrrs.append(np.mean(1.0 / ranks))
#
#     return np.mean(mrrs)
#
#
# def evaluate_language_mean_mrr(language):
#     language_dir = utils.get_base_language_serialized_data_path(language)
#     model = utils.load_serialized_model_weights(language_dir, train_model.get_model())
#
#     valid_code_seqs = utils.load_serialized_seqs(language_dir, 'code', set_='valid')
#     valid_query_seqs = utils.load_serialized_seqs(language_dir, 'query', set_='valid')
#     valid_mean_mrr = evaluate_model_mean_mrr(model, valid_code_seqs, valid_query_seqs)
#
#     test_code_seqs = utils.load_serialized_seqs(language_dir, 'code', set_='test')
#     test_query_seqs = utils.load_serialized_seqs(language_dir, 'query', set_='test')
#     test_mean_mrr = evaluate_model_mean_mrr(model, test_code_seqs, test_query_seqs)
#
#     print(f'Evaluating {language} - Valid Mean MRR: {valid_mean_mrr}, Test Mean MRR: {test_mean_mrr}')
#     return valid_mean_mrr, test_mean_mrr
#
#
# def evaluate_mean_mrr(use_wandb=False):
#     language_valid_mrrs = {}
#     language_test_mrrs = {}
#
#     for language in shared.LANGUAGES:
#         valid_mrr, test_mrr = evaluate_language_mean_mrr(language)
#         language_valid_mrrs[f'{language}_valid_mrr'] = valid_mrr
#         language_test_mrrs[f'{language}_test_mrr'] = test_mrr
#
#     valid_mean_mrr = np.mean(list(language_valid_mrrs.values()))
#     test_mean_mrr = np.mean(list(language_test_mrrs.values()))
#     print(f'All languages - Valid Mean MRR: {valid_mean_mrr}, Test Mean MRR: {test_mean_mrr}')
#
#     if use_wandb:
#         wandb.log({
#             'valid_mean_mrr': valid_mean_mrr,
#             'test_mean_mrr': test_mean_mrr,
#             **language_valid_mrrs,
#             **language_test_mrrs
#         })


if __name__ == '__main__':
    # evaluate_mean_mrr()

    build_code_embeddings()
    wandb.init(project=shared.ENV['WANDB_PROJECT_NAME'], config=shared.get_wandb_config())
    emit_ndcg_model_predictions(True)
