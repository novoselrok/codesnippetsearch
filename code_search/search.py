from typing import List, Dict, Tuple

import torch

from code_search.code_embedding import get_annoy_index
from code_search.data_manager import DataManager
from code_search.model import CodeSearchNN
from code_search.prepare_data import pad_encode_query
from code_search.torch_utils import torch_gpu_to_np, np_to_torch


def get_nearest_query_neighbors_per_language(
        model: CodeSearchNN,
        data_manager: DataManager,
        languages: List[str],
        query: str,
        max_query_seq_length: int,
        device: torch.device,
        results_per_language: int = 20) -> Dict[str, Tuple[List[int], List[float]]]:
    nearest_neighbors_per_language = {}

    for language in languages:
        with torch.no_grad():
            padded_encoded_query = np_to_torch(pad_encode_query(data_manager, query, max_query_seq_length), device)
            query_embedding = torch_gpu_to_np(model.encode_query(padded_encoded_query))[0, :]

        ann = data_manager.get_language_annoy_index(get_annoy_index(model.embedding_size), language)
        nearest_neighbors_per_language[language] = ann.get_nns_by_vector(
            query_embedding, results_per_language, include_distances=True)

    return nearest_neighbors_per_language


def get_nearest_code_neighbors(
        data_manager: DataManager,
        language: str,
        embedding_row_index: int,
        embedding_size: int,
        n_results: int = 20) -> Tuple[List[int], List[float]]:
    ann = data_manager.get_language_annoy_index(get_annoy_index(embedding_size), language)
    indices, distances = ann.get_nns_by_item(
        embedding_row_index, n_results + 1, include_distances=True)
    return indices[1:], distances[1:]  # Exclude the first result since it belongs to the embedding row
