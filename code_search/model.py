from typing import List, Optional, Dict

import torch
import torch.nn as nn

from code_search import shared
from code_search.data_manager import DataManager

SMALL_NUMBER = 1e-8


class CodeSearchNN(nn.Module):
    def __init__(
            self,
            languages: List[str],
            embedding_size: int,
            code_vocabulary_size: Dict[str, int],
            code_seq_length: int,
            query_vocabulary_size: int,
            query_seq_length: int):
        super().__init__()

        self.embedding_size = embedding_size
        self.languages = languages
        for language in languages:
            setattr(self, f'{language}_embedding_dropout', nn.Dropout(p=0.1))
            setattr(self, f'{language}_embedding', nn.Embedding(
                num_embeddings=code_vocabulary_size[language], embedding_dim=embedding_size, padding_idx=0))
            setattr(self, f'{language}_weights_layer', nn.Linear(embedding_size, 1, bias=False))
            setattr(self, f'{language}_weights_layer_bn', nn.BatchNorm1d(code_seq_length))

        self.query_embedding_dropout = nn.Dropout(p=0.1)
        self.query_embedding = nn.Embedding(
            num_embeddings=query_vocabulary_size, embedding_dim=embedding_size, padding_idx=0)
        self.query_weights_layer = nn.Linear(embedding_size, 1, bias=False)
        self.query_weights_layer_bn = nn.BatchNorm1d(query_seq_length)

    def get_query_weights_layer(self):
        return self.query_weights_layer.weight

    def set_query_weights_layer(self, weight: torch.Tensor):
        self.query_weights_layer.weight = nn.Parameter(weight)

    def get_language_weights_layer(self, language: str):
        return getattr(self, f'{language}_weights_layer').weight

    def set_language_weights_layer(self, language: str, weight: torch.Tensor):
        getattr(self, f'{language}_weights_layer').weight = nn.Parameter(weight)

    def get_query_embedding_weights(self):
        return self.query_embedding.weight

    def set_query_embedding_weights(self, embedding_weights: torch.Tensor):
        self.query_embedding.weight = nn.Parameter(embedding_weights)

    def get_language_embedding_weights(self, language: str):
        return getattr(self, f'{language}_embedding').weight

    def set_language_embedding_weights(self, language: str, embedding_weights: torch.Tensor):
        getattr(self, f'{language}_embedding').weight = nn.Parameter(embedding_weights)

    @staticmethod
    def mask(seqs: torch.Tensor):
        return (seqs != 0).float().unsqueeze(dim=2)

    @staticmethod
    def weighted_mean(embedding: torch.Tensor, weights: torch.Tensor):
        weighted_sum = torch.sum(embedding * weights, dim=1)
        return weighted_sum / (torch.sum(weights, dim=1) + SMALL_NUMBER)

    def encode_code(self, language: str, code_seqs: torch.Tensor):
        mask = CodeSearchNN.mask(code_seqs)
        embedding = getattr(self, f'{language}_embedding')(code_seqs)
        embedding = getattr(self, f'{language}_embedding_dropout')(embedding)
        bn = getattr(self, f'{language}_weights_layer_bn')
        weights = torch.sigmoid(bn(getattr(self, f'{language}_weights_layer')(embedding))) * mask
        return CodeSearchNN.weighted_mean(embedding, weights)

    def encode_query(self, query_seqs):
        mask = CodeSearchNN.mask(query_seqs)
        embedding = self.query_embedding(query_seqs)
        embedding = self.query_embedding_dropout(embedding)
        weights = torch.sigmoid(self.query_weights_layer_bn(self.query_weights_layer(embedding))) * mask
        return CodeSearchNN.weighted_mean(embedding, weights)

    def forward(self,
                language_code_seqs: Dict[str, Optional[torch.Tensor]],
                language_query_seqs: Dict[str, Optional[torch.Tensor]]):
        query_seqs = torch.cat([language_query_seqs[language] for language in self.languages
                                if language_query_seqs[language] is not None])
        query_embeds_mean = self.encode_query(query_seqs)

        code_embeds_mean = torch.cat(
            [self.encode_code(language, language_code_seqs[language]) for language in self.languages
             if language_code_seqs[language] is not None])

        query_embeds_mean_norm = torch.norm(query_embeds_mean, dim=1, keepdim=True) + SMALL_NUMBER
        code_embeds_mean_norm = torch.norm(code_embeds_mean, dim=1, keepdim=True) + SMALL_NUMBER

        return torch.matmul(
            query_embeds_mean / query_embeds_mean_norm, (code_embeds_mean / code_embeds_mean_norm).t())


def get_model(
        languages: List[str],
        embedding_size: int,
        code_vocabulary_size: Dict[str, int],
        code_seq_length: int,
        query_vocabulary_size: int,
        query_seq_length: int,
        device: Optional[torch.device] = None):
    model = CodeSearchNN(
        languages,
        embedding_size,
        code_vocabulary_size,
        code_seq_length,
        query_vocabulary_size,
        query_seq_length)

    if device is not None:
        model = model.to(device)

    return model


def get_base_language_model(device: Optional[torch.device] = None):
    return get_model(
        shared.LANGUAGES,
        shared.EMBEDDING_SIZE,
        {language: shared.CODE_VOCABULARY_SIZE for language in shared.LANGUAGES},
        shared.CODE_MAX_SEQ_LENGTH,
        shared.QUERY_VOCABULARY_SIZE,
        shared.QUERY_MAX_SEQ_LENGTH,
        device)


def get_base_language_model_for_evaluation(data_manager: DataManager, device: Optional[torch.device] = None):
    model = data_manager.get_torch_model(get_base_language_model(device))
    model.eval()
    return model


def get_repository_model(data_manager: DataManager, languages: List[str], device: Optional[torch.device] = None):
    query_vocabulary_size = data_manager.get_query_vocabulary().vocab_size
    code_vocabulary_size = {language: data_manager.get_language_vocabulary(language).vocab_size
                            for language in languages}

    model = get_model(
        languages,
        shared.EMBEDDING_SIZE,
        code_vocabulary_size,
        shared.CODE_MAX_SEQ_LENGTH,
        query_vocabulary_size,
        shared.QUERY_MAX_SEQ_LENGTH)

    if device is not None:
        model = model.to(device)

    return model


def get_repository_model_for_evaluation(
        data_manager: DataManager,
        languages: List[str],
        device: Optional[torch.device] = None) -> CodeSearchNN:
    model = data_manager.get_torch_model(get_repository_model(data_manager, languages, device))
    model.eval()
    return model
