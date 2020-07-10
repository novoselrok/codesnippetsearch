import argparse
import random
import time
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np

import wandb
from more_itertools import chunked
from scipy.spatial.distance import cdist
from torch import optim

from code_search import shared, utils
from code_search.model import CodeSearchNet


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    """Early stops the training if score doesn't improve after a given patience."""

    def __init__(self, model, path, patience=5, verbose=False):
        self.model = model
        self.path = path
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
        utils.serialize_model(self.model.state_dict(), self.path)


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


def cosine_loss(cosine_similarity_matrix):
    neg_matrix = torch.zeros(*list(cosine_similarity_matrix.shape), requires_grad=True).to(get_device())
    neg_matrix.fill_diagonal_(float('-inf'))

    # Distance between query and code snippet should be as small as possible
    diagonal_cosine_distance = 1. - torch.diag(cosine_similarity_matrix)
    # Max. similarity between query and non-corresponding code snippet should be as small as possible
    max_positive_non_diagonal_similarity_in_row, _ = torch.max(F.relu(cosine_similarity_matrix + neg_matrix), dim=1)
    # Combined distance and similarity should be as small as possible as well
    per_sample_loss = F.relu(diagonal_cosine_distance + max_positive_non_diagonal_similarity_in_row)

    return torch.mean(per_sample_loss)


def get_language_mrrs(model: CodeSearchNet,
                      language: str,
                      code_seqs: torch.Tensor,
                      query_seqs: torch.Tensor,
                      batch_size=1000,
                      seed=0):
    n_samples = code_seqs.shape[0]
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)

    mrrs = []
    for idx_chunk in chunked(indices, batch_size):
        if len(idx_chunk) < batch_size:
            continue

        code_embeddings = model.encode_code(language, code_seqs[idx_chunk]).cpu().numpy()
        query_embeddings = model.encode_query(query_seqs[idx_chunk]).cpu().numpy()

        distance_matrix = cdist(query_embeddings, code_embeddings, 'cosine')
        correct_elements = np.expand_dims(np.diag(distance_matrix), axis=-1)
        ranks = np.sum(distance_matrix <= correct_elements, axis=-1)
        mrrs.append(np.mean(1.0 / ranks))

    return mrrs


def get_mrrs_per_language(model: CodeSearchNet,
                          language_code_seqs: Dict[str, np.ndarray],
                          language_query_seqs: Dict[str, np.ndarray]):
    device = get_device()
    language_mrrs = {}
    for language in language_code_seqs.keys():
        valid_code_seqs = np_to_torch(language_code_seqs[language], device)
        valid_query_seqs = np_to_torch(language_query_seqs[language], device)
        language_mrrs[language] = get_language_mrrs(model, language, valid_code_seqs, valid_query_seqs)
    return language_mrrs


def np_to_torch(arr: np.ndarray, device: torch.device):
    return torch.from_numpy(arr).to(device)


def get_model():
    return CodeSearchNet(
        shared.LANGUAGES, shared.EMBEDDING_SIZE, shared.CODE_VOCABULARY_SIZE, shared.QUERY_VOCABULARY_SIZE)


def load_language_set_seqs(languages: List[str], set_: str):
    language_code_seqs = OrderedDict()
    language_query_seqs = OrderedDict()
    for language in languages:
        language_dir = utils.get_base_language_serialized_data_path(language)
        language_code_seqs[language] = utils.load_serialized_seqs(language_dir, 'code', set_=set_)
        language_query_seqs[language] = utils.load_serialized_seqs(language_dir, 'query', set_=set_)
    return language_code_seqs, language_query_seqs


def train(model: CodeSearchNet,
          language_code_seqs: Dict[str, np.ndarray],
          language_query_seqs: Dict[str, np.ndarray],
          model_save_path: str,
          learning_rate=1e-3,
          batch_size=1000,
          max_epochs=100,
          verbose=True):
    device = get_device()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    es = EarlyStopping(model, model_save_path, verbose=verbose)

    for epoch in range(max_epochs):
        if verbose:
            print(f'Epoch {epoch}')
        epoch_start = time.time()

        model.train()

        loss_per_batch = []
        for batch_language_code_seqs, batch_language_query_seqs in generate_batch(
                language_code_seqs, language_query_seqs, batch_size):
            for language in batch_language_code_seqs.keys():
                if batch_language_code_seqs[language] is None:
                    continue

                batch_language_code_seqs[language] = np_to_torch(batch_language_code_seqs[language], device)
                batch_language_query_seqs[language] = np_to_torch(batch_language_query_seqs[language], device)

            optimizer.zero_grad()
            output = model(batch_language_code_seqs, batch_language_query_seqs)
            loss = cosine_loss(output)
            loss.backward()
            optimizer.step()

            loss_per_batch.append(loss.item())

        model.eval()
        with torch.no_grad():
            validation_mrr = get_mrrs_per_language(model, 'valid')

        if es(validation_mrr):
            break

        if verbose:
            mean_loss_per_batch = np.mean(loss_per_batch)
            epoch_duration = time.time() - epoch_start
            print(f'Duration: {epoch_duration:.1f}s, Train loss: {mean_loss_per_batch:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Train code search model from prepared data.')
    parser.add_argument('--notes', default='')
    utils.add_bool_arg(parser, 'wandb', default=False)
    args = vars(parser.parse_args())

    language_code_seqs = OrderedDict()
    language_query_seqs = OrderedDict()

    for language in shared.LANGUAGES:
        language_dir = utils.get_base_language_serialized_data_path(language)
        language_code_seqs[language] = utils.load_serialized_seqs(language_dir, 'code', set_='train')
        language_query_seqs[language] = utils.load_serialized_seqs(language_dir, 'query', set_='train')

    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=shared.LEARNING_RATE)
    es = EarlyStopping(model, shared.BASE_LANGUAGES_DIR, verbose=True)

    if args['wandb']:
        wandb.init(project=shared.ENV['WANDB_PROJECT_NAME'], notes=args['notes'], config=shared.get_wandb_config())
        wandb.watch(model)

    for epoch in range(100):
        print(f'Epoch {epoch}')
        epoch_start = time.time()

        model.train()
        losses = []

        for batch_language_code_seqs, batch_language_query_seqs in generate_batch(
                language_code_seqs, language_query_seqs, shared.TRAIN_BATCH_SIZE):

            for language in shared.LANGUAGES:
                if batch_language_code_seqs[language] is None:
                    continue

                batch_language_code_seqs[language] = torch.from_numpy(batch_language_code_seqs[language]).to(device)
                batch_language_query_seqs[language] = torch.from_numpy(
                    batch_language_query_seqs[language]).to(device)

            optimizer.zero_grad()
            output = model(batch_language_code_seqs, batch_language_query_seqs)
            loss = cosine_loss(output)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        epoch_duration = time.time() - epoch_start
        mean_loss = np.mean(losses)
        print(f'Duration: {epoch_duration:.1f}s, Loss: {mean_loss:.4f}')

        model.eval()
        with torch.no_grad():
            validation_mrr = evaluate_mrr(model, 'valid')

        if args['wandb']:
            wandb.log({'loss': mean_loss, 'mrr': validation_mrr})

        if es(validation_mrr):
            break

    # TODO: Load best saved model
    model.eval()
    with torch.no_grad():
        evaluate_mrr(model, 'test')


if __name__ == '__main__':
    main()
