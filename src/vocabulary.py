from typing import List, Dict, Counter as TypingCounter, Union, Optional, Iterator
from collections import Counter

MASK_TOKEN = '%MASK%'


class Vocabulary:
    def __init__(self):
        self.token_to_id: Dict[str, int] = {MASK_TOKEN: 0}
        self.id_to_token: List[str] = [MASK_TOKEN]

    def add_token(self, token: str):
        if token in self.token_to_id:
            return

        token_id = len(self.id_to_token)
        self.token_to_id[token] = token_id
        self.id_to_token.append(token)

    def get_token_id(self, token: str) -> Optional[int]:
        return self.token_to_id.get(token)

    def add_tokens(self, tokens: TypingCounter[str], vocabulary_size: int, count_threshold: int):
        for token, count in tokens.most_common(vocabulary_size):
            if count >= count_threshold:
                self.add_token(token)
            else:
                break

    @property
    def size(self):
        return len(self.id_to_token)

    @staticmethod
    def create_vocabulary(tokens: Iterator[str], ignored_tokens=None,
                          vocabulary_size: int = 10000, count_threshold: int = 10):
        counter = Counter([token
                           for token in tokens
                           if not ignored_tokens or (ignored_tokens and token not in ignored_tokens)])
        vocabulary = Vocabulary()
        vocabulary.add_tokens(counter, vocabulary_size, count_threshold)
        return vocabulary
