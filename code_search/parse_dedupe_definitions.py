import sys
import os

from code_search import shared
from code_search import utils

language = sys.argv[1]
evaluation_docs_pkl_path = os.path.join(shared.ENV['CODESEARCHNET_DATA_DIR'], f'{language}_dedupe_definitions_v2.pkl')
evaluation_docs = utils.load_pickled_object(evaluation_docs_pkl_path)

utils.write_jsonl(
    evaluation_docs,
    utils.get_cached_docs_path(language, 'evaluation')
)
