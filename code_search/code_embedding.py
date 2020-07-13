import argparse
from typing import List

import torch
import numpy as np
from annoy import AnnoyIndex

from code_search import shared, utils
from code_search.torch_utils import np_to_torch, torch_gpu_to_np, get_device
from code_search.data_manager import DataManager
from code_search.model import CodeSearchNN, get_base_language_model


def get_annoy_index(embedding_size: int):
    return AnnoyIndex(embedding_size, 'angular')


def batch_encode_code_seqs(
        model: CodeSearchNN, language: str, code_seqs: np.ndarray, device: torch.device, batch_size=1000):
    n_seqs = code_seqs.shape[0]
    code_embeddings = np.zeros((n_seqs, model.embedding_size))

    idx = 0
    for _ in range((n_seqs // batch_size) + 1):
        end_idx = min(n_seqs, idx + batch_size)

        with torch.no_grad():
            batch_code_seqs = np_to_torch(code_seqs[idx:end_idx, :], device)
            code_embeddings[idx:end_idx, :] = torch_gpu_to_np(model.encode_code(language, batch_code_seqs))

        idx += batch_size

    return code_embeddings


def build_language_code_embeddings(
        model: CodeSearchNN, data_manager: DataManager, languages: List[str], device: torch.device, verbose=True):
    for language in languages:
        if verbose:
            print(f'Building {language} code embeddings')
        code_seqs = data_manager.get_language_seqs(language, shared.DataType.CODE, shared.DataSet.ALL)
        code_embeddings = batch_encode_code_seqs(model, language, code_seqs, device)
        data_manager.save_language_code_embeddings(code_embeddings, language)


def build_language_annoy_indices(data_manager: DataManager, languages: List[str], n_trees=200, verbose=True):
    for language in languages:
        if verbose:
            print(f'Building {language} AnnoyIndex')
        code_embeddings = data_manager.get_language_code_embeddings(language)
        n_samples, embedding_size = code_embeddings.shape
        annoy_index = get_annoy_index(embedding_size)
        for i in range(n_samples):
            annoy_index.add_item(i, code_embeddings[i, :])
        annoy_index.build(n_trees)

        data_manager.save_language_annoy_index(annoy_index, language)


def main():
    parser = argparse.ArgumentParser(description='Build code embeddings and AnnoyIndex.')
    utils.add_bool_arg(parser, 'code-embeddings', default=True)
    utils.add_bool_arg(parser, 'annoy-index', default=False)
    args = vars(parser.parse_args())

    data_manager = DataManager(shared.BASE_LANGUAGES_DIR)
    device = get_device()
    model: CodeSearchNN = data_manager.get_torch_model(get_base_language_model(device))
    model.eval()

    if args['code-embeddings']:
        build_language_code_embeddings(model, data_manager, shared.LANGUAGES, device)

    if args['annoy-index']:
        build_language_annoy_indices(data_manager, shared.LANGUAGES)


if __name__ == '__main__':
    main()
