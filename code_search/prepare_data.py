import argparse
import gc
import itertools
import re
from collections import Counter
from multiprocessing import Pool
from typing import Iterable, Callable

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

from code_search import shared
from code_search import utils
from code_search.bpevocabulary import BpeVocabulary, DEFAULT_UNK

np.random.seed(0)

IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
IDENTIFIER_CAMEL_CASE_SPLIT = re.compile('(?<=[a-z0-9])([A-Z])')
PUNCTUATION_TOKENS = set('.?:;,')
CODE_PUNCTUATION = set('(){}[].:=,')


def extract_sub_tokens(token):
    sub_tokens = re.split('[._]', token)
    sub_tokens = [IDENTIFIER_CAMEL_CASE_SPLIT.sub(r' \1', sub_token).split(' ')
                  if IDENTIFIER_TOKEN_REGEX.match(sub_token) else [sub_token]
                  for sub_token in sub_tokens]

    return [token.lower() for token in utils.flatten(sub_tokens) if len(token.strip()) > 0]


def preprocess_code_tokens(tokens: Iterable[str]) -> shared.TokensGenerator:
    for token in tokens:
        if token in CODE_PUNCTUATION:
            continue

        yield [token.lower().strip()]


def preprocess_query_tokens(tokens: Iterable[str]) -> shared.TokensGenerator:
    for token in tokens:
        token_lower = token.lower().strip()

        if token_lower in STOP_WORDS or token_lower in PUNCTUATION_TOKENS:
            continue

        yield [token_lower]


def extract_doc(raw_doc, use_func_name_as_query=False):
    func_name = raw_doc['func_name']

    is_func_name_as_query = use_func_name_as_query and func_name and len(func_name) > shared.MIN_FUNC_NAME_QUERY_LENGTH
    if is_func_name_as_query:
        query_tokens = extract_sub_tokens(func_name)
        # Replace function name occurrences in code
        code_tokens = [token if token != func_name else DEFAULT_UNK for token in raw_doc['code_tokens']]
    else:
        query_tokens = raw_doc['docstring_tokens']
        code_tokens = raw_doc['code_tokens']

    return {
        'func_name': func_name,
        'query_tokens': query_tokens,
        'code_tokens': code_tokens,
        'url': raw_doc['url'],
        'is_func_name_as_query': is_func_name_as_query,
    }


def prepare_set_docs(args):
    language, set_ = args
    print(f'Building docs for {language} {set_}')

    docs = [extract_doc(raw_doc, set_ == 'train' and np.random.uniform(0., 1.) < shared.FUNC_NAME_AS_QUERY_PCT)
            for raw_doc in utils.get_raw_docs(language, set_)]
    utils.cache_docs(docs, language, set_)

    print(f'Done building for {language} {set_}')


def prepare_docs():
    with Pool(8) as p:
        p.map(prepare_set_docs, itertools.product(shared.LANGUAGES, shared.DATA_SETS))


def prepare_language_vocabulary(args):
    language, (tokens_key, vocab_size, type_) = args
    print(f'Building vocabulary for {language} {type_}')

    docs = utils.load_cached_docs(language, 'train')
    tokens = utils.flatten(preprocess_query_tokens(utils.flatten(doc[tokens_key] for doc in docs)))
    vocabulary = BpeVocabulary(vocab_size=vocab_size, pct_bpe=shared.VOCABULARY_PCT_BPE)
    vocabulary.fit(Counter(tokens))
    utils.cache_vocabulary(vocabulary, language, type_)

    print(f'Done building vocabulary for {language} {type_}')


def prepare_vocabularies():
    vocabulary_configs = [
        ('code_tokens', shared.CODE_VOCABULARY_SIZE, 'code'),
        ('query_tokens', shared.QUERY_VOCABULARY_SIZE, 'query'),
    ]

    with Pool(8) as p:
        p.map(prepare_language_vocabulary, itertools.product(shared.LANGUAGES, vocabulary_configs))


def keep_valid_seqs(padded_encoded_code_seqs, padded_encoded_query_seqs):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_query_seqs = padded_encoded_query_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_query_seqs

    return padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_query_seqs[valid_seqs_indices, :]


def pad_encode_seqs(
        preprocess_tokens_fn: Callable[[Iterable[str]], shared.TokensGenerator],
        seqs: shared.TokensGenerator,
        max_length: int,
        language: str,
        type_: str) -> np.ndarray:
    bpe = utils.load_cached_vocabulary(language, type_)
    encoded_seqs = bpe.transform(
        (utils.flatten(preprocess_tokens_fn(seq)) for seq in seqs), fixed_length=max_length)
    return np.array(list(encoded_seqs))


def prepare_set_seqs(args):
    language, set_ = args
    print(f'Building sequences for {language} {set_}')

    # Prepare code seqs
    code_seqs = (doc['code_tokens'] for doc in utils.load_cached_docs(language, set_))
    padded_encoded_code_seqs = pad_encode_seqs(
        preprocess_code_tokens, code_seqs, shared.CODE_MAX_SEQ_LENGTH, language, 'code')

    # Prepare query seqs
    query_seqs = (doc['query_tokens'] for doc in utils.load_cached_docs(language, set_))
    padded_encoded_query_seqs = pad_encode_seqs(
        preprocess_query_tokens, query_seqs, shared.QUERY_MAX_SEQ_LENGTH, language, 'query')

    # Check for invalid sequences
    padded_encoded_code_seqs, padded_encoded_query_seqs = keep_valid_seqs(
        padded_encoded_code_seqs, padded_encoded_query_seqs)

    utils.cache_seqs(padded_encoded_code_seqs, language, set_, 'code')
    utils.cache_seqs(padded_encoded_query_seqs, language, set_, 'query')

    print(f'Done building sequences for {language} {set_}')


def prepare_evaluation_seqs(language):
    print(f'Building evaluation sequences for {language}')

    evaluation_docs = utils.load_cached_docs(language, 'evaluation')
    evaluation_code_seqs = (doc['function_tokens'] for doc in evaluation_docs)

    evaluation_padded_encoded_code_seqs = pad_encode_seqs(
        preprocess_code_tokens, evaluation_code_seqs, shared.CODE_MAX_SEQ_LENGTH, language, 'code')
    utils.cache_seqs(evaluation_padded_encoded_code_seqs, language, 'evaluation', 'code')

    # Just to be safe
    del evaluation_code_seqs
    gc.collect()

    print(f'Done building evaluation sequences for {language}')


def prepare_seqs():
    with Pool(4) as p:
        p.map(prepare_set_seqs, itertools.product(shared.LANGUAGES, shared.DATA_SETS))

    with Pool(2) as p:
        p.map(prepare_evaluation_seqs, shared.LANGUAGES)


def main():
    parser = argparse.ArgumentParser(description='Prepare data before training the code search models.')
    utils.add_bool_arg(parser, 'prepare-all', default=False)
    utils.add_bool_arg(parser, 'prepare-docs', default=False)
    utils.add_bool_arg(parser, 'prepare-vocabularies', default=False)
    utils.add_bool_arg(parser, 'prepare-seqs', default=False)
    args = vars(parser.parse_args())

    if args['prepare-all'] or args['prepare-docs']:
        prepare_docs()

    if args['prepare-all'] or args['prepare-vocabularies']:
        prepare_vocabularies()

    if args['prepare-all'] or args['prepare-seqs']:
        prepare_seqs()


if __name__ == '__main__':
    main()
