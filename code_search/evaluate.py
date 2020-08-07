import argparse
import gc
import json
import os
import random
from collections import defaultdict
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
import wandb
from more_itertools import chunked
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from code_search import shared, prepare_data, utils, preprocessing_tokens
from code_search.code_embedding import get_annoy_index, build_code_embeddings
from code_search.model import CodeSearchNN, get_base_language_model_for_evaluation
from code_search.torch_utils import get_device, np_to_torch, torch_gpu_to_np
from code_search.data_manager import DataManager, get_base_languages_data_manager

np.random.seed(0)


def get_language_mrrs(model: CodeSearchNN,
                      language: str,
                      code_seqs: torch.Tensor,
                      query_seqs: torch.Tensor,
                      batch_size=1000,
                      seed=0) -> List[float]:
    indices = list(range(code_seqs.shape[0]))
    random.Random(seed).shuffle(indices)

    mrrs = []
    for idx_chunk in chunked(indices, batch_size):
        if len(idx_chunk) < batch_size:
            continue

        code_embeddings = torch_gpu_to_np(model.encode_code(language, code_seqs[idx_chunk]))
        query_embeddings = torch_gpu_to_np(model.encode_query(query_seqs[idx_chunk]))

        distance_matrix = cdist(query_embeddings, code_embeddings, 'cosine')
        correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
        ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
        ranks = ranks[np.invert(np.isnan(ranks)) & (ranks >= 1)]  # Make sure we only use valid ranks
        mrrs.append(float(np.mean(1.0 / ranks)))

    return mrrs


def evaluate_mrr(model: CodeSearchNN,
                 language_code_seqs: Dict[str, np.ndarray],
                 language_query_seqs: Dict[str, np.ndarray],
                 device: torch.device,
                 batch_size: int = 1000):
    mrrs_per_language = {}
    for language in language_code_seqs.keys():
        code_seqs = np_to_torch(language_code_seqs[language], device)
        query_seqs = np_to_torch(language_query_seqs[language], device)
        mrrs_per_language[language] = get_language_mrrs(model, language, code_seqs, query_seqs, batch_size=batch_size)

    mean_mrr = np.mean(list(utils.flatten(mrrs_per_language.values())))
    mean_mrr_per_language = {language: np.mean(values) for language, values in mrrs_per_language.items()}
    return mean_mrr, mean_mrr_per_language


def ndcg(predictions: Dict[str, List[str]], relevance_scores: Dict[str, Dict[str, float]]) -> float:
    num_results = 0
    ndcg_sum = 0

    for query, query_relevance_annotations in relevance_scores.items():
        current_rank = 1
        query_dcg = 0
        for url in predictions[query]:
            if url in query_relevance_annotations:
                query_dcg += (2**query_relevance_annotations[url] - 1) / np.log2(current_rank + 1)
            current_rank += 1

        query_idcg = 0
        for i, ideal_relevance in enumerate(sorted(query_relevance_annotations.values(), reverse=True), start=1):
            query_idcg += (2 ** ideal_relevance - 1) / np.log2(i + 1)
        if query_idcg == 0:
            # We have no positive annotations for the given query, so we should probably not penalize anyone about this.
            continue
        num_results += 1
        ndcg_sum += query_dcg / query_idcg

    return ndcg_sum / num_results


def get_ndcg_predictions(
        queries,
        model: CodeSearchNN,
        data_manager: DataManager,
        device: torch.device,
        nn_lib: str = 'scikit',
        n_neighbors: int = 150,
        search_k: int = -1):
    predictions = []
    for language in shared.LANGUAGES:
        print(f'Evaluating {language}')

        evaluation_docs = [{'url': doc['url'], 'identifier': doc['identifier']}
                           for doc in data_manager.get_language_corpus(language, shared.DataSet.ALL)]

        with torch.no_grad():
            query_seqs = prepare_data.pad_encode_seqs(
                (line.split(' ') for line in queries),
                shared.QUERY_MAX_SEQ_LENGTH,
                data_manager.get_query_vocabulary(),
                preprocessing_tokens.preprocess_query_tokens)
            query_embeddings = torch_gpu_to_np(model.encode_query(np_to_torch(query_seqs, device)))

        if nn_lib == 'scikit':
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
            nn.fit(data_manager.get_language_code_embeddings(language))
            _, nearest_neighbor_indices_per_query = nn.kneighbors(query_embeddings)

            for query_idx, query in enumerate(queries):
                for query_nearest_code_idx in nearest_neighbor_indices_per_query[query_idx, :]:
                    predictions.append({
                        'query': query,
                        'language': language,
                        'identifier': evaluation_docs[query_nearest_code_idx]['identifier'],
                        'url': evaluation_docs[query_nearest_code_idx]['url'],
                    })
        elif nn_lib == 'annoy':
            annoy_index = data_manager.get_language_annoy_index(get_annoy_index(query_embeddings.shape[1]), language)
            for query_idx, query in enumerate(queries):
                nearest_neighbor_indices = annoy_index.get_nns_by_vector(
                    query_embeddings[query_idx, :], n_neighbors, search_k=search_k)

                for query_nearest_code_idx in nearest_neighbor_indices:
                    predictions.append({
                        'query': query,
                        'language': language,
                        'identifier': evaluation_docs[query_nearest_code_idx]['identifier'],
                        'url': evaluation_docs[query_nearest_code_idx]['url'],
                    })
        else:
            raise Exception('Unknown nearest neighbors library.')

        del evaluation_docs
        gc.collect()

    return predictions


def evaluate_ndcg(model: CodeSearchNN, data_manager: DataManager, languages: List[str], device: torch.device):
    build_code_embeddings(model, data_manager, languages, device)
    validation_queries = [
        'sort string list'
    ]
    relevance_scores = json.load(open(os.path.join(shared.ROOT_DIR, 'relevance_annotations.json')))
    ndcg_predictions = get_ndcg_predictions(validation_queries, model, data_manager, device)
    language_ndcg = []
    for language in languages:
        language_predictions_per_query = defaultdict(list)
        for prediction in ndcg_predictions:
            if prediction['language'] == language:
                language_predictions_per_query[prediction['query']].append(prediction['url'])

        score = ndcg(language_predictions_per_query, relevance_scores[language])
        print(f'{language} NDCG: {score}')
        language_ndcg.append(score)

    print(f'Mean NDCG: {np.mean(language_ndcg)}')


def get_evaluation_queries():
    with open(os.path.join(shared.CODESEARCHNET_DATA_DIR, 'queries.csv'), encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[1:]]


def main():
    parser = argparse.ArgumentParser(description='Evaluate code search model.')
    utils.add_bool_arg(parser, 'wandb', default=False)
    args = vars(parser.parse_args())

    if args['wandb']:
        wandb.init(project=shared.ENV['WANDB_PROJECT_NAME'], config=shared.get_wandb_config())

    device = get_device()
    data_manager = get_base_languages_data_manager()
    model = get_base_language_model_for_evaluation(data_manager, device)
    queries = get_evaluation_queries()

    ndcg_predictions = get_ndcg_predictions(queries, model, data_manager, device)

    df_predictions = pd.DataFrame(ndcg_predictions, columns=['query', 'language', 'identifier', 'url'])
    save_path = os.path.join(wandb.run.dir, 'model_predictions.csv') if args['wandb'] else '../model_predictions.csv'
    df_predictions.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
