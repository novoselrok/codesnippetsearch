import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DATA_DIR = os.path.join(ROOT_DIR, 'data')
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
SAVED_SEQS_DIR = os.path.join(ROOT_DIR, 'saved_seqs')
SAVED_VOCABULARIES_DIR = os.path.join(ROOT_DIR, 'saved_vocabularies')


LANGUAGES_NUM_FILES = {
    'python': 14,
    'ruby': 2,
    'php': 18,
    'go': 11,
    'javascript': 5,
    'java': 16,
}
LANGUAGES = list(sorted(LANGUAGES_NUM_FILES.keys()))

# Model constants
EMBEDDING_SIZE = 128

CODE_VOCABULARY_SIZE = 10000
CODE_TOKEN_COUNT_THRESHOLD = 10
CODE_MAX_SEQ_LENGTH = 200

DOCSTRING_VOCABULARY_SIZE = 10000
DOCSTRING_TOKEN_COUNT_THRESHOLD = 10
DOCSTRING_MAX_SEQ_LENGTH = 30
