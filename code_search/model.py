from typing import List, Optional, Dict

import torch
import torch.nn as nn


SMALL_NUMBER = 1e-10


class CodeSearchNet(nn.Module):
    def __init__(
            self, languages: List[str], embedding_size: int, code_vocabulary_size: int, query_vocabulary_size: int):
        super().__init__()

        self.languages = languages
        for language in languages:
            setattr(self, f'{language}_embedding_dropout', nn.Dropout(p=0.2))
            setattr(self, f'{language}_embedding', nn.Embedding(
                num_embeddings=code_vocabulary_size, embedding_dim=embedding_size, padding_idx=0))

        self.query_embedding_dropout = nn.Dropout(p=0.2)
        self.query_embedding = nn.Embedding(
            num_embeddings=query_vocabulary_size, embedding_dim=embedding_size, padding_idx=0)

    @staticmethod
    def mask_aware_mean(embedding: torch.Tensor):
        mask = (torch.sum(torch.abs(embedding), dim=2, keepdim=True) != 0).float()
        non_masked_rows = torch.sum(mask, dim=1)
        mean = torch.sum(embedding, dim=1) / (non_masked_rows + SMALL_NUMBER)
        return mean

    def encode_code(self, language: str, code_seqs: torch.Tensor):
        embedding = getattr(self, f'{language}_embedding')(code_seqs)
        embedding = getattr(self, f'{language}_embedding_dropout')(embedding)
        return CodeSearchNet.mask_aware_mean(embedding)

    def encode_query(self, query_seqs):
        embedding = self.query_embedding(query_seqs)
        embedding = self.query_embedding_dropout(embedding)
        return CodeSearchNet.mask_aware_mean(embedding)

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
