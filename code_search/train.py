import argparse
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

import wandb
from torch import optim

from code_search import shared, utils
from code_search.data_manager import DataManager, get_base_languages_data_manager
from code_search.model import CodeSearchNN, get_base_language_model
from code_search.torch_utils import get_device, np_to_torch
from code_search.evaluate import evaluate_mrr


class EarlyStopping:
    """Early stops the training if score doesn't improve after a given patience."""
    def __init__(self, model, data_manager: DataManager, patience=5, verbose=False):
        self.model = model
        self.data_manager = data_manager
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        """Returns True if the model should stop training."""
        if self.best_score is None:
            self.save_model()
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1

            if self.counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping with best score: {self.best_score}.')
                return True
        else:
            self.save_model()
            self.best_score = score
            self.counter = 0

        return False

    def save_model(self):
        self.data_manager.save_torch_model(self.model)


def generate_batch(
        language_code_seqs: Dict[str, np.ndarray], language_query_seqs: Dict[str, np.ndarray], batch_size: int):
    languages = list(language_code_seqs.keys())
    n_language_samples = {language: code_seqs.shape[0] for language, code_seqs in language_code_seqs.items()}
    n_language_remaining_samples = n_language_samples.copy()

    # Shuffle each of the languages
    for language in languages:
        shuffled_indices = np.arange(0, n_language_samples[language])
        np.random.shuffle(shuffled_indices)

        language_code_seqs[language] = language_code_seqs[language][shuffled_indices, :]
        language_query_seqs[language] = language_query_seqs[language][shuffled_indices, :]

    language_idx = {language: 0 for language in languages}
    remaining_samples = sum(n_language_remaining_samples.values())

    while remaining_samples > 0:
        batch_language_code_seqs = OrderedDict()
        batch_language_query_seqs = OrderedDict()

        for language in languages:
            language_batch_size = int(batch_size * (n_language_remaining_samples[language] / float(remaining_samples)))
            idx = language_idx[language]
            end_idx = min(idx + language_batch_size, n_language_samples[language])
            n_batch_samples = min(language_batch_size, end_idx - idx)

            if n_batch_samples > 0:
                batch_language_code_seqs[language] = language_code_seqs[language][idx:end_idx, :]
                batch_language_query_seqs[language] = language_query_seqs[language][idx:end_idx, :]

                language_idx[language] += n_batch_samples
                n_language_remaining_samples[language] -= n_batch_samples
            else:
                batch_language_code_seqs[language] = None
                batch_language_query_seqs[language] = None

        yield batch_language_code_seqs, batch_language_query_seqs

        remaining_samples = sum(n_language_remaining_samples.values())


def cosine_loss(cosine_similarity_matrix, device: torch.device):
    neg_matrix = torch.zeros(*list(cosine_similarity_matrix.shape), requires_grad=True).to(device)
    neg_matrix.fill_diagonal_(float('-inf'))

    # Distance between query and code snippet should be as small as possible
    diagonal_cosine_distance = 1. - torch.diag(cosine_similarity_matrix)
    # Max. similarity between query and non-corresponding code snippet should be as small as possible
    max_positive_non_diagonal_similarity_in_row, _ = torch.max(F.relu(cosine_similarity_matrix + neg_matrix), dim=1)
    # Combined distance and similarity should be as small as possible as well
    per_sample_loss = F.relu(diagonal_cosine_distance + max_positive_non_diagonal_similarity_in_row)

    return torch.mean(per_sample_loss)


def load_language_set_seqs(data_manager: DataManager, languages: List[str], set_: shared.DataSet):
    language_code_seqs = OrderedDict()
    language_query_seqs = OrderedDict()
    for language in languages:
        language_code_seqs[language] = data_manager.get_language_seqs(language, shared.DataType.CODE, set_)
        language_query_seqs[language] = data_manager.get_language_seqs(language, shared.DataType.QUERY, set_)
    return language_code_seqs, language_query_seqs


def train_model(model: CodeSearchNN,
                train_language_seqs: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
                valid_language_seqs: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
                data_manager: DataManager,
                device: torch.device,
                learning_rate=1e-3,
                batch_size=1000,
                max_epochs=100,
                mrr_eval_batch_size=1000,
                verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    es = EarlyStopping(model, data_manager, verbose=verbose)

    train_language_code_seqs, train_language_query_seqs = train_language_seqs
    valid_language_code_seqs, valid_language_query_seqs = valid_language_seqs

    for epoch in range(max_epochs):
        if verbose:
            print(f'=== Epoch {epoch} ===')
        epoch_start = time.time()

        model.train()

        loss_per_batch = []
        for batch_language_code_seqs, batch_language_query_seqs in generate_batch(
                train_language_code_seqs, train_language_query_seqs, batch_size):
            for language in batch_language_code_seqs.keys():
                if batch_language_code_seqs[language] is None:
                    continue

                batch_language_code_seqs[language] = np_to_torch(batch_language_code_seqs[language], device)
                batch_language_query_seqs[language] = np_to_torch(batch_language_query_seqs[language], device)

            optimizer.zero_grad()
            output = model(batch_language_code_seqs, batch_language_query_seqs)
            loss = cosine_loss(output, device)
            loss.backward()
            optimizer.step()

            loss_per_batch.append(loss.item())

        model.eval()
        with torch.no_grad():
            validation_mean_mrr, validation_mean_mrr_per_language = evaluate_mrr(
                model, valid_language_code_seqs, valid_language_query_seqs, device, batch_size=mrr_eval_batch_size)

        if verbose:
            mean_loss_per_batch = np.mean(loss_per_batch)
            epoch_duration = time.time() - epoch_start
            print(f'Duration: {epoch_duration:.1f}s')
            print(f'Train loss: {mean_loss_per_batch:.4f}, Valid MRR: {validation_mean_mrr:.4f}')
            print(f'Valid MRR per language: {validation_mean_mrr_per_language}')

        if es(validation_mean_mrr):
            break


def train(model: CodeSearchNN, data_manager: DataManager, languages: List[str], device: torch.device, **kwargs):
    train_language_seqs = load_language_set_seqs(
        data_manager, languages, shared.DataSet.TRAIN)
    valid_language_seqs = load_language_set_seqs(
        data_manager, languages, shared.DataSet.VALID)

    train_model(
        model,
        train_language_seqs,
        valid_language_seqs,
        data_manager,
        device,
        **kwargs)

    test_language_code_seqs, test_language_query_seqs = load_language_set_seqs(
        data_manager, languages, shared.DataSet.TEST)

    best_model = data_manager.get_torch_model(model)
    model.eval()
    with torch.no_grad():
        test_mean_mrr, test_mean_mrr_per_language = evaluate_mrr(
            best_model,
            test_language_code_seqs,
            test_language_query_seqs,
            device,
            batch_size=kwargs.get('mrr_eval_batch_size', 1000))

        if kwargs['verbose']:
            print(f'Test MRR: {test_mean_mrr:.4f}')
            print(f'Test MRR: {test_mean_mrr_per_language}')


def main():
    parser = argparse.ArgumentParser(description='Train code search model from prepared data.')
    parser.add_argument('--notes', default='')
    utils.add_bool_arg(parser, 'wandb', default=False)
    args = vars(parser.parse_args())

    data_manager = get_base_languages_data_manager()
    device = get_device()
    model = get_base_language_model(device)

    if args['wandb']:
        wandb.init(project=shared.ENV['WANDB_PROJECT_NAME'], notes=args['notes'], config=shared.get_wandb_config())
        wandb.watch(model)

    train(
        model,
        data_manager,
        shared.LANGUAGES,
        device,
        learning_rate=shared.LEARNING_RATE,
        batch_size=shared.TRAIN_BATCH_SIZE,
        verbose=True)


if __name__ == '__main__':
    main()
