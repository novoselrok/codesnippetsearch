import pickle
import re

import numpy as np
import tokenizers
from keras_preprocessing.sequence import pad_sequences
from spacy.lang.en.stop_words import STOP_WORDS

from shared import LANGUAGES, LANGUAGES_NUM_FILES, CODE_VOCABULARY_SIZE, CODE_TOKEN_COUNT_THRESHOLD, \
    DOCSTRING_TOKEN_COUNT_THRESHOLD, DOCSTRING_VOCABULARY_SIZE, CODE_MAX_SEQ_LENGTH, DOCSTRING_MAX_SEQ_LENGTH
from utils import get_docs, get_pickled_object, flatten
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


def preprocess_code_tokens(tokens):
    for token in tokens:
        if token in CODE_PUNCTUATION:
            continue

        sub_tokens = token.split('_')
        sub_tokens = [IDENTIFIER_CAMEL_CASE_SPLIT.sub(r' \1', sub_token).split(' ')
                      if IDENTIFIER_TOKEN_REGEX.match(sub_token) else [sub_token]
                      for sub_token in sub_tokens]

        yield [token.lower() for token in flatten(sub_tokens) if len(token.strip()) > 0]


def preprocess_docstring_tokens(tokens):
    for token in tokens:
        token_lower = token.lower()

        if token_lower in STOP_WORDS or token_lower in PUNCTUATION_TOKENS:
            continue

        yield [token_lower]


def prepare_dictionaries():
    docstring_tokens = []
    for language in LANGUAGES:
        print(f'Building {language} vocabulary')
        docs = get_docs([f'../data/{language}_train_{i}.jsonl' for i in range(LANGUAGES_NUM_FILES[language])])

        build_and_serialize_vocabulary(
            flatten(preprocess_code_tokens(flatten([doc['code_tokens'] for doc in docs]))),
            CODE_VOCABULARY_SIZE,
            CODE_TOKEN_COUNT_THRESHOLD,
            f'../saved_vocabularies/{language}_vocabulary.pckl'
        )

        docstring_tokens.extend([doc['docstring_tokens'] for doc in docs])

    print('Building docstrings vocabulary')
    docstring_tokens_joined = ' '.join(flatten(preprocess_docstring_tokens(flatten(docstring_tokens))))

    with open('../saved_vocabularies/docstring_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(docstring_tokens_joined)

    bpe = tokenizers.CharBPETokenizer()
    bpe.train('../saved_vocabularies/docstring_corpus.txt', show_progress=False,
              vocab_size=DOCSTRING_VOCABULARY_SIZE, min_frequency=DOCSTRING_TOKEN_COUNT_THRESHOLD)
    bpe.save('../saved_vocabularies', 'docstring')


def keep_valid_seqs(padded_encoded_code_seqs, padded_encoded_docstring_seqs):
    # Keep seqs with at least one valid token
    valid_code_seqs = padded_encoded_code_seqs.astype(bool).sum(axis=1) > 0
    valid_docstring_seqs = padded_encoded_docstring_seqs.astype(bool).sum(axis=1) > 0
    valid_seqs_indices = valid_code_seqs & valid_docstring_seqs

    return padded_encoded_code_seqs[valid_seqs_indices, :], padded_encoded_docstring_seqs[valid_seqs_indices, :]


def encode_code_tokens(tokens, vocabulary):
    return list(sorted(set([vocabulary.get_token_id(token)
                            for token in flatten(preprocess_code_tokens(tokens))])))


def pad_code_seqs(seqs):
    return np.array(pad_sequences(seqs, maxlen=CODE_MAX_SEQ_LENGTH, padding='post'))


def pad_encode_docstring_seqs(seqs):
    bpe = tokenizers.CharBPETokenizer(
        '../saved_vocabularies/docstring-vocab.json', '../saved_vocabularies/docstring-merges.txt')
    bpe.enable_padding()
    bpe.enable_truncation(max_length=DOCSTRING_MAX_SEQ_LENGTH)

    encoded_seqs = bpe.encode_batch([' '.join(flatten(preprocess_docstring_tokens(sorted(set(seq)))))
                                     for seq in seqs])
    return np.array([output.ids for output in encoded_seqs])


def prepare_seqs():
    for language in LANGUAGES:
        print(f'Building {language} sequences')

        for set_ in ['train', 'valid']:
            files = [f'../data/{language}_{set_}_{i}.jsonl' for i in range(LANGUAGES_NUM_FILES[language])] \
                if set_ == 'train' else [f'../data/{language}_{set_}_0.jsonl']
            docs = get_docs(files)
            language_vocabulary: Vocabulary = get_pickled_object(f'../saved_vocabularies/{language}_vocabulary.pckl')

            encoded_code_seqs = [encode_code_tokens(doc['code_tokens'], language_vocabulary)
                                 for doc in docs]
            padded_encoded_code_seqs = pad_code_seqs(encoded_code_seqs)

            docstring_seqs = [doc['docstring_tokens'] for doc in docs]
            padded_encoded_docstring_seqs = pad_encode_docstring_seqs(docstring_seqs)

            padded_encoded_code_seqs, padded_encoded_docstring_seqs = keep_valid_seqs(
                padded_encoded_code_seqs, padded_encoded_docstring_seqs)

            np.save(f'../saved_seqs/{language}_{set_}_code_seqs.npy', padded_encoded_code_seqs)
            np.save(f'../saved_seqs/{language}_{set_}_docstring_seqs.npy', padded_encoded_docstring_seqs)


if __name__ == '__main__':
    prepare_dictionaries()
    prepare_seqs()
