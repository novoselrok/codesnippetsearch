import random
from collections import Counter
from typing import Iterable

import numpy as np

from code_search import utils
from code_search.bpevocabulary import BpeVocabulary
from code_search.preprocessing_tokens import remove_inline_comments, preprocess_query_tokens, \
    preprocess_code_tokens

random.seed(0)
np.random.seed(0)


def preprocess_doc(doc, language: str):
    identifier = doc['identifier'] if 'identifier' in doc else doc['func_name']
    query_tokens = doc['docstring_tokens']
    code_tokens = doc['code_tokens']

    return {
        # func_name and url are needed for evaluation
        'identifier': identifier,
        'url': doc.get('url'),
        'query_tokens': list(utils.flatten(preprocess_query_tokens(query_tokens))),
        'code_tokens': list(
            utils.flatten(preprocess_code_tokens(remove_inline_comments(language, code_tokens)))),
    }


def build_vocabulary(tokens: Iterable[str], vocabulary_size: int, pct_bpe: float):
    vocabulary = BpeVocabulary(vocab_size=vocabulary_size, pct_bpe=pct_bpe)
    vocabulary.fit(Counter(tokens))
    return vocabulary
