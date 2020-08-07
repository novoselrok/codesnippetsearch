import os

from tree_sitter import Language

from code_search import shared
from code_search.function_parser.utils import get_tree_sitter_languages_lib

languages = [
    os.path.join(shared.VENDOR_DIR, 'tree-sitter-python'),
    os.path.join(shared.VENDOR_DIR, 'tree-sitter-javascript'),
    os.path.join(shared.VENDOR_DIR, 'tree-sitter-go'),
    os.path.join(shared.VENDOR_DIR, 'tree-sitter-ruby'),
    os.path.join(shared.VENDOR_DIR, 'tree-sitter-java'),
    os.path.join(shared.VENDOR_DIR, 'tree-sitter-php')
]

Language.build_library(
    # Store the library in the directory
    get_tree_sitter_languages_lib(),
    # Include one or more languages
    languages
)
