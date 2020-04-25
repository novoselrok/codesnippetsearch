import argparse
import pickle
import re

import numpy as np
import tokenizers
from keras_preprocessing.sequence import pad_sequences
from spacy.lang.en.stop_words import STOP_WORDS

import shared
import utils
from vocabulary import Vocabulary

IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
IDENTIFIER_CAMEL_CASE_SPLIT = re.compile('(?<=[a-z0-9])([A-Z])')
PUNCTUATION_TOKENS = set('.?:;,')
CODE_PUNCTUATION = set('(){}[].:=,')


def build_and_serialize_vocabulary(tokens, vocabulary_size, count_threshold, serialize_path, ignored_tokens=None):
    vocabulary = Vocabulary.create_vocabulary(tokens,
                                              vocabulary_size=vocabulary_size,
                                              count_threshold=count_threshold,
                                              ignored_tokens=ignored_tokens)
    with open(serialize_path, 'wb') as f:
        pickle.dump(vocabulary, f)


def extract_sub_tokens(token):
    sub_tokens = token.split('_')
    sub_tokens = [IDENTIFIER_CAMEL_CASE_SPLIT.sub(r' \1', sub_token).split(' ')
                  if IDENTIFIER_TOKEN_REGEX.match(sub_token) else [sub_token]
                  for sub_token in sub_tokens]

    return [token.lower() for token in utils.flatten(sub_tokens) if len(token.strip()) > 0]


def preprocess_code_tokens(tokens):
    for token in tokens:
        if token in CODE_PUNCTUATION:
            continue

        yield extract_sub_tokens(token)


def preprocess_docstring_tokens(tokens):
    for token in tokens:
        token_lower = token.lower()

        if token_lower in STOP_WORDS or token_lower in PUNCTUATION_TOKENS:
            continue

        yield extract_sub_tokens(token)


def prepare_vocabularies():
    for language in shared.LANGUAGES:
        print(f'Building {language} vocabulary')
        docs = utils.get_docs([utils.get_code_data_path(f'{language}_train_{i}.jsonl')
                               for i in range(shared.LANGUAGES_NUM_FILES[language])])

        build_and_serialize_vocabulary(
            utils.flatten(preprocess_code_tokens(utils.flatten([doc['code_tokens'] for doc in docs]))),
            shared.CODE_VOCABULARY_SIZE,
            shared.CODE_TOKEN_COUNT_THRESHOLD,
            utils.get_saved_vocabulary_path(f'{language}_vocabulary.pckl')
        )

        print(f'Building {language} docstrings vocabulary')
        docstring_tokens = [doc['docstring_tokens'] for doc in docs]
        docstring_tokens_joined = ' '.join(utils.flatten(preprocess_docstring_tokens(utils.flatten(docstring_tokens))))

        language_docstring_corpus = utils.get_saved_vocabulary_path(f'{language}_docstring_corpus.txt')
        with open(language_docstring_corpus, 'w', encoding='utf-8') as f:
            f.write(docstring_tokens_joined)

        bpe = tokenizers.CharBPETokenizer()
        bpe.train(language_docstring_corpus, show_progress=False,
                  vocab_size=shared.DOCSTRING_VOCABULARY_SIZE, min_frequency=shared.DOCSTRING_TOKEN_COUNT_THRESHOLD)
        bpe.enable_padding(max_length=shared.DOCSTRING_MAX_SEQ_LENGTH)
        bpe.enable_truncation(max_length=shared.DOCSTRING_MAX_SEQ_LENGTH)
        bpe.save(shared.SAVED_VOCABULARIES_DIR, f'{language}_docstring')


def keep_valid_seqs(padded_encoded_code_seqs, padded_encoded_docstring_seqs):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_docstring_seqs = padded_encoded_docstring_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_docstring_seqs

    return np.arange(0, padded_encoded_code_seqs.shape[0])[valid_seqs_indices], \
        padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_docstring_seqs[valid_seqs_indices, :]


def encode_code_tokens(tokens, vocabulary):
    return [vocabulary.get_token_id(token) for token in utils.flatten(preprocess_code_tokens(tokens))]


def pad_code_seqs(seqs):
    return np.array(pad_sequences(seqs, maxlen=shared.CODE_MAX_SEQ_LENGTH, padding='post'))


def pad_encode_docstring_seqs(language, seqs):
    bpe = tokenizers.CharBPETokenizer(
        utils.get_saved_vocabulary_path(f'{language}_docstring-vocab.json'),
        utils.get_saved_vocabulary_path(f'{language}_docstring-merges.txt'))
    bpe.enable_padding(max_length=shared.DOCSTRING_MAX_SEQ_LENGTH)
    bpe.enable_truncation(max_length=shared.DOCSTRING_MAX_SEQ_LENGTH)

    encoded_seqs = bpe.encode_batch([' '.join(utils.flatten(preprocess_docstring_tokens(seq))) for seq in seqs])
    return np.array([output.ids for output in encoded_seqs])


def prepare_seqs():
    for language in shared.LANGUAGES:
        print(f'Building {language} sequences')

        for set_ in ['train', 'valid', 'test']:
            if set_ == 'train':
                files = [utils.get_code_data_path(f'{language}_{set_}_{i}.jsonl')
                         for i in range(shared.LANGUAGES_NUM_FILES[language])]
            else:
                files = [utils.get_code_data_path(f'{language}_{set_}_0.jsonl')]

            docs = utils.get_docs(files)
            language_vocabulary: Vocabulary = utils.get_pickled_object(
                utils.get_saved_vocabulary_path(f'{language}_vocabulary.pckl'))

            encoded_code_seqs = [encode_code_tokens(doc['code_tokens'], language_vocabulary) for doc in docs]
            padded_encoded_code_seqs = pad_code_seqs(encoded_code_seqs)

            docstring_seqs = [doc['docstring_tokens'] for doc in docs]
            padded_encoded_docstring_seqs = pad_encode_docstring_seqs(language, docstring_seqs)

            valid_indices, padded_encoded_code_seqs, padded_encoded_docstring_seqs = keep_valid_seqs(
                padded_encoded_code_seqs, padded_encoded_docstring_seqs)

            if set_ == 'train':
                valid_docs = [{'url': docs[idx]['url'], 'identifier': docs[idx]['func_name']}
                              for idx in valid_indices]
                with open(utils.get_saved_seqs_path(f'{language}_valid_docs.pckl'), 'wb') as f:
                    pickle.dump(valid_docs, f)

            np.save(utils.get_saved_seqs_path(f'{language}_{set_}_code_seqs.npy'), padded_encoded_code_seqs)
            np.save(utils.get_saved_seqs_path(f'{language}_{set_}_docstring_seqs.npy'), padded_encoded_docstring_seqs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data before training the code search models.')
    utils.add_bool_arg(parser, 'prepare-vocabularies', default=True)
    utils.add_bool_arg(parser, 'prepare-seqs', default=True)
    args = vars(parser.parse_args())

    if args['prepare-vocabularies']:
        prepare_vocabularies()

    if args['prepare-seqs']:
        prepare_seqs()
