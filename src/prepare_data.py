import argparse
import gc
import json
import re
from collections import Counter

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

import shared
import utils
from bpevocabulary import BpeVocabulary, DEFAULT_UNK

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


def preprocess_code_tokens(tokens):
    for token in tokens:
        if token in CODE_PUNCTUATION:
            continue

        yield [token.lower()]


def preprocess_query_tokens(tokens):
    for token in tokens:
        token_lower = token.lower()

        if token_lower in STOP_WORDS or token_lower in PUNCTUATION_TOKENS:
            continue

        yield [token_lower]


def build_and_serialize_bpe_vocabulary(tokens, vocab_size, serialize_path):
    vocabulary = BpeVocabulary(vocab_size=vocab_size, pct_bpe=0.5)
    vocabulary.fit(Counter(tokens))
    utils.pickle_object(vocabulary, serialize_path)


def extract_doc(raw_doc, use_func_name_as_query=False):
    func_name = raw_doc['func_name']

    is_func_name_as_query = use_func_name_as_query and func_name and len(func_name) > 12
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


def prepare_docs():
    for language in shared.LANGUAGES:
        print(f'Building {language} docs')

        for set_ in ['train', 'valid', 'test']:
            print(f'Set {set_}')

            if set_ == 'train':
                files = [utils.get_code_data_path(f'{language}_{set_}_{i}.jsonl')
                         for i in range(shared.LANGUAGES_NUM_FILES[language])]
            else:
                files = [utils.get_code_data_path(f'{language}_{set_}_0.jsonl')]

            raw_docs = utils.get_raw_docs(files)
            docs = [extract_doc(raw_doc, set_ == 'train' and np.random.uniform(0., 1.) < 0.1) for raw_doc in raw_docs]

            with open(utils.get_saved_docs_path(f'{language}_{set_}_docs.json'), 'w', encoding='utf-8') as f:
                json.dump(docs, f)


def prepare_vocabularies():
    for language in shared.LANGUAGES:
        docs = utils.load_json(utils.get_saved_docs_path(f'{language}_train_docs.json'))

        print(f'Building {language} code vocabulary')
        build_and_serialize_bpe_vocabulary(
            utils.flatten(preprocess_code_tokens(utils.flatten([doc['code_tokens'] for doc in docs]))),
            shared.CODE_VOCABULARY_SIZE,
            utils.get_saved_vocabulary_path(f'{language}_code.pckl'),
        )

        print(f'Building {language} query vocabulary')
        build_and_serialize_bpe_vocabulary(
            utils.flatten(preprocess_query_tokens(utils.flatten([doc['query_tokens'] for doc in docs]))),
            shared.QUERY_VOCABULARY_SIZE,
            utils.get_saved_vocabulary_path(f'{language}_query.pckl'),
        )


def keep_valid_seqs(padded_encoded_code_seqs, padded_encoded_query_seqs):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_query_seqs = padded_encoded_query_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_query_seqs

    return padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_query_seqs[valid_seqs_indices, :]


def pad_encode_seqs(language, seq_type, preprocess_tokens_fn, seqs, max_length):
    bpe: BpeVocabulary = utils.get_pickled_object(
        utils.get_saved_vocabulary_path(f'{language}_{seq_type}.pckl'))

    encoded_seqs = bpe.transform(
        (utils.flatten(preprocess_tokens_fn(seq)) for seq in seqs), fixed_length=max_length)
    return np.array(list(encoded_seqs))


def prepare_seqs():
    for language in shared.LANGUAGES:
        print(f'Building {language} sequences.')

        for set_ in ['train', 'valid', 'test']:
            print(set_)
            docs = utils.load_json(utils.get_saved_docs_path(f'{language}_{set_}_docs.json'))

            # Prepare code seqs
            code_seqs = [doc['code_tokens'] for doc in docs]
            padded_encoded_code_seqs = pad_encode_seqs(
                language, 'code', preprocess_code_tokens, code_seqs, shared.CODE_MAX_SEQ_LENGTH)

            # Prepare query seqs
            query_seqs = [doc['query_tokens'] for doc in docs]
            padded_encoded_query_seqs = pad_encode_seqs(
                language, 'query', preprocess_query_tokens, query_seqs, shared.QUERY_MAX_SEQ_LENGTH)

            # Check for invalid sequences
            padded_encoded_code_seqs, padded_encoded_query_seqs = keep_valid_seqs(
                padded_encoded_code_seqs, padded_encoded_query_seqs)

            np.save(utils.get_saved_seqs_path(f'{language}_{set_}_code_seqs.npy'), padded_encoded_code_seqs)
            np.save(utils.get_saved_seqs_path(f'{language}_{set_}_query_seqs.npy'), padded_encoded_query_seqs)

            del padded_encoded_code_seqs
            del padded_encoded_query_seqs
            del docs
            del code_seqs
            del query_seqs
            gc.collect()

        # Build docs and seqs for NCDG evaluation
        print('Building evaluation seqs')
        evaluation_docs_iter = utils.iter_jsonl(
            utils.get_code_data_path(f'{language}_definitions.jsonl'))
        evaluation_code_seqs_gen = (doc['function_tokens'] for doc in evaluation_docs_iter)

        evaluation_padded_encoded_code_seqs = pad_encode_seqs(
            language, 'code', preprocess_code_tokens, evaluation_code_seqs_gen, shared.CODE_MAX_SEQ_LENGTH)

        np.save(utils.get_saved_seqs_path(
                    f'{language}_evaluation_code_seqs.npy'), evaluation_padded_encoded_code_seqs)

        del evaluation_padded_encoded_code_seqs
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data before training the code search models.')
    utils.add_bool_arg(parser, 'prepare-docs', default=False)
    utils.add_bool_arg(parser, 'prepare-vocabularies', default=False)
    utils.add_bool_arg(parser, 'prepare-seqs', default=False)
    args = vars(parser.parse_args())

    if args['prepare-docs']:
        prepare_docs()

    if args['prepare-vocabularies']:
        prepare_vocabularies()

    if args['prepare-seqs']:
        prepare_seqs()
