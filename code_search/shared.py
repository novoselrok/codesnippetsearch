import enum
import json
import os
from typing import Generator, Iterable

TokensGenerator = Generator[Iterable[str], None, None]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT_DIR, 'env.json')
SERIALIZED_DATA_DIR = os.path.join(ROOT_DIR, 'serialized_data')
VENDOR_DIR = os.path.join(ROOT_DIR, 'vendor')
BUILD_DIR = os.path.join(ROOT_DIR, 'build')
CODESEARCHNET_DATA_DIR = os.path.join(ROOT_DIR, 'codesearchnet_data')
BASE_LANGUAGES_DIR = os.path.join(SERIALIZED_DATA_DIR, 'languages')

# TODO: Filenames should be without extensions
GLOVE_EMBEDDINGS_FILENAME = 'glove_embeddings'
GLOVE_VOCABULARY_FILENAME = 'glove_vocabulary'

SERIALIZED_SEQS_FILENAME = 'seqs_{set_}_{type_}'

SERIALIZED_MODEL_FILENAME = 'model'

SERIALIZED_EMBEDDING_WEIGHTS = 'embedding_weights_{type_}'

SERIALIZED_VOCABULARY_FILENAME = 'vocabulary_{type_}'
SERIALIZED_QUERY_EMBEDDING_WEIGHTS_FILENAME = 'embedding_weights_query'

SERIALIZED_CORPUS_FILENAME = 'corpus_{set_}'
SERIALIZED_PREPROCESSED_CORPUS_FILENAME = 'preprocessed_corpus_{set_}'

SERIALIZED_CODE_EMBEDDINGS_FILENAME = 'code_embeddings'

SERIALIZED_ANN_FILENAME = 'ann'

LANGUAGES_NUM_FILES = {
    'python': 14,
    'ruby': 2,
    'php': 18,
    'go': 11,
    'javascript': 5,
    'java': 16,
}
LANGUAGES = list(sorted(LANGUAGES_NUM_FILES.keys()))


class DataType(enum.Enum):
    CODE = 'code'
    QUERY = 'query'

    def __str__(self):
        return self.value


class DataSet(enum.Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    ALL = 'all'

    @staticmethod
    def sets():
        return list(DataSet)

    @staticmethod
    def split_data_sets():
        return [DataSet.TRAIN, DataSet.VALID, DataSet.TEST]

    def __str__(self):
        return self.value


# DATA_SETS_SPLIT = [0.8, 0.1, 0.1]

# Model constants
TRAIN_BATCH_SIZE = 1000
EMBEDDING_SIZE = 128
LEARNING_RATE = 0.01

FUNC_NAME_AS_QUERY_PCT = 0.1
MIN_FUNC_NAME_QUERY_LENGTH = 12
VOCABULARY_PCT_BPE = 0.5

CODE_VOCABULARY_SIZE = 10000
CODE_TOKEN_COUNT_THRESHOLD = 10
CODE_MAX_SEQ_LENGTH = 200

QUERY_VOCABULARY_SIZE = CODE_VOCABULARY_SIZE
QUERY_TOKEN_COUNT_THRESHOLD = 10
QUERY_MAX_SEQ_LENGTH = 30


def get_wandb_config():
    return {
        'learning_rate': LEARNING_RATE,
        'train_batch_size': TRAIN_BATCH_SIZE,
        'embedding_size': EMBEDDING_SIZE,
        'func_name_as_query_pct': FUNC_NAME_AS_QUERY_PCT,
        'min_func_name_query_length': MIN_FUNC_NAME_QUERY_LENGTH,
        'vocabulary_pct_bpe': VOCABULARY_PCT_BPE,
        'code_vocabulary_size': CODE_VOCABULARY_SIZE,
        'code_token_count_threshold': CODE_TOKEN_COUNT_THRESHOLD,
        'code_max_seq_length': CODE_MAX_SEQ_LENGTH,
        'query_vocabulary_size': QUERY_VOCABULARY_SIZE,
        'query_token_count_threshold': QUERY_TOKEN_COUNT_THRESHOLD,
        'query_max_seq_length': QUERY_MAX_SEQ_LENGTH,
    }


with open(ENV_FILE, encoding='utf-8') as f:
    ENV = json.load(f)
