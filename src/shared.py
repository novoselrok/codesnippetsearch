import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DATA_DIR = os.path.join(ROOT_DIR, 'data')
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
SAVED_SEQS_DIR = os.path.join(ROOT_DIR, 'saved_seqs')
SAVED_VOCABULARIES_DIR = os.path.join(ROOT_DIR, 'saved_vocabularies')
SAVED_DOCS_DIR = os.path.join(ROOT_DIR, 'saved_docs')


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
TRAIN_BATCH_SIZE = 1000
EMBEDDING_SIZE = 256

CODE_VOCABULARY_SIZE = 10000
CODE_TOKEN_COUNT_THRESHOLD = 10
CODE_MAX_SEQ_LENGTH = 200

QUERY_VOCABULARY_SIZE = 10000
QUERY_TOKEN_COUNT_THRESHOLD = 10
QUERY_MAX_SEQ_LENGTH = 30


def get_wandb_config():
    return {
        'train_batch_size': TRAIN_BATCH_SIZE,
        'embedding_size': EMBEDDING_SIZE,
        'code_vocabulary_size': CODE_VOCABULARY_SIZE,
        'code_token_count_threshold': CODE_TOKEN_COUNT_THRESHOLD,
        'code_max_seq_length': CODE_MAX_SEQ_LENGTH,
        'query_vocabulary_size': QUERY_VOCABULARY_SIZE,
        'query_token_count_threshold': QUERY_TOKEN_COUNT_THRESHOLD,
        'query_max_seq_length': QUERY_MAX_SEQ_LENGTH,
    }
