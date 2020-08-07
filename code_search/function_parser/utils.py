import os

from code_search import shared


def get_tree_sitter_languages_lib():
    return os.path.join(shared.BUILD_DIR, 'py-tree-sitter-languages.so')


def walk(directory: str, ext: str):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.' + ext):
                yield os.path.join(root, f)
