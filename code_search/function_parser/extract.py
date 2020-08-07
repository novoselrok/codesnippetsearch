from typing import Optional, Type, List, Dict, Any

from tree_sitter import Parser, Language

from code_search.function_parser.parsers.language_parser import LanguageParser, tokenize_docstring, tokenize_code
from code_search.function_parser.utils import walk, get_tree_sitter_languages_lib
from code_search.function_parser.language_data import LANGUAGE_METADATA


class DirectoryCodeDocumentsExtractor:
    MAX_FILE_BLOB_LENGTH = 100000

    def __init__(self, parser: Parser, language: str, language_parser: Type[LanguageParser]):
        self.parser = parser
        self.language = language
        self.language_parser = language_parser

    def process_directory(self, directory: str, ext: str, ignore_paths: List[str]) -> List[Dict[str, Any]]:
        docs = []

        for file_path in walk(directory, ext):
            if any(fp in file_path.lower() for fp in ignore_paths):
                continue

            function_definitions = self.get_function_definitions(file_path)

            if function_definitions is None:
                continue

            relative_file_path = file_path[len(directory):].lstrip('/')
            docs.extend(
                (self.get_code_document(func, relative_file_path)
                 for func in function_definitions if len(func['function_tokens']) > 1))

        return docs

    def get_code_document(self, function_definition: Dict[str, Any], path: str):
        return {
            'language': self.language,
            'path': path,
            'identifier': function_definition['identifier'],
            'docstring': function_definition['docstring'].strip(),
            'docstring_summary': function_definition['docstring_summary'].strip(),
            'docstring_tokens': tokenize_docstring(function_definition['docstring_summary']),
            'code': function_definition['function'].strip(),
            'code_tokens': function_definition['function_tokens'],
            'start_line': function_definition['start_point'][0] + 1,
            'end_line': function_definition['end_point'][0] + 1,
        }

    def get_function_definitions(self, file_path: str) -> Optional[List]:
        try:
            with open(file_path) as source_code:
                blob = source_code.read()

            if len(blob) > DirectoryCodeDocumentsExtractor.MAX_FILE_BLOB_LENGTH:
                return None

            tree = self.parser.parse(blob.encode())
            return self.language_parser.get_definitions(tree, blob)
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError) as e:
            print(e)
            return None


def extract_tokens_from_blob(blob: str, language: str) -> List[str]:
    parser = Parser()
    parser.set_language(Language(get_tree_sitter_languages_lib(), language))
    tree = parser.parse(blob.encode())
    return tokenize_code(tree.root_node, blob)


def extract(repository_dir_path: str, language: str):
    parser = Parser()
    parser.set_language(Language(get_tree_sitter_languages_lib(), language))

    language_metadata = LANGUAGE_METADATA[language]
    processor = DirectoryCodeDocumentsExtractor(parser, language, language_metadata['language_parser'])
    return processor.process_directory(
        repository_dir_path, language_metadata['ext'], language_metadata['ignore_paths'])
