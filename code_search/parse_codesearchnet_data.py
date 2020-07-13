import os

from code_search import shared, serialize
from code_search.prepare_data import DataManager


def rename_dedupe_definitions_keys(doc):
    doc['code'] = doc.pop('function')
    doc['code_tokens'] = doc.pop('function_tokens')
    return doc


def rename_set_doc_keys(doc):
    doc['identifier'] = doc.pop('func_name')
    return doc


def get_base_language_doc_path(language: str, set_: shared.DataSet.TRAIN, idx: int) -> str:
    return os.path.join(
        shared.CODESEARCHNET_DATA_DIR, language, 'final', 'jsonl', str(set_), f'{language}_{set_}_{idx}')


def get_codesearchnet_language_set_corpus(language: str, set_: shared.DataSet):
    if set_ == shared.DataSet.TRAIN:
        file_paths = [get_base_language_doc_path(language, set_, i)
                      for i in range(shared.LANGUAGES_NUM_FILES[language])]
    else:
        file_paths = [get_base_language_doc_path(language, set_, 0)]

    for file_path in file_paths:
        yield from serialize.load('jsonl-gzip', file_path)


def combine_language_set_corpus(data_manager: DataManager, language: str, set_: shared.DataSet):
    corpus = (rename_set_doc_keys(doc) for doc in get_codesearchnet_language_set_corpus(language, set_))
    data_manager.save_language_corpus(corpus, language, set_)


def parse_dedupe_definitions(data_manager: DataManager, language: str):
    dedupe_definitions_pkl_path = os.path.join(
        shared.CODESEARCHNET_DATA_DIR, f'{language}_dedupe_definitions_v2')
    dedupe_definitions = serialize.load('pickle', dedupe_definitions_pkl_path)

    corpus = (rename_dedupe_definitions_keys(doc) for doc in dedupe_definitions)
    data_manager.save_language_corpus(corpus, language, shared.DataSet.ALL)


def main():
    data_manager = DataManager(shared.BASE_LANGUAGES_DIR)

    for language in shared.LANGUAGES:
        print(f'Preparing {language}')
        parse_dedupe_definitions(data_manager, language)

        for set_ in shared.DataSet.split_data_sets():
            print(set_)
            combine_language_set_corpus(data_manager, language, set_)


if __name__ == '__main__':
    main()
