from typing import Optional, Type, List, Dict, Any

from tree_sitter import Parser, Language

from code_search.function_parser.parsers.language_parser import LanguageParser, tokenize_docstring
from code_search.function_parser.utils import walk, get_tree_sitter_languages_lib
from code_search.function_parser.language_data import LANGUAGE_METADATA


class DirectoryCodeDocumentsExtractor:
    MAX_FILE_BLOB_LENGTH = 100000

    def __init__(self, parser: Parser, language: str, language_parser: Type[LanguageParser]):
        self.parser = parser
        self.language = language
        self.language_parser = language_parser

    def process_directory(self, directory: str, ext: str) -> List[Dict[str, Any]]:
        docs = []

        for file_path in walk(directory, ext):
            relative_file_path = file_path[len(directory):].lstrip('/')
            function_definitions = self.get_function_definitions(file_path)

            if function_definitions is None:
                continue

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
        if any(fp in file_path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None

        try:
            with open(file_path) as source_code:
                blob = source_code.read()

            if len(blob) > DirectoryCodeDocumentsExtractor.MAX_FILE_BLOB_LENGTH:
                return None

            tree = self.parser.parse(blob.encode())
            return self.language_parser.get_definitions(tree, blob)
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError) as e:
            # TODO: Log error
            print(e)
            return None


def extract(repository_dir_path: str, language: str):
    parser = Parser()
    parser.set_language(Language(get_tree_sitter_languages_lib(), language))
    processor = DirectoryCodeDocumentsExtractor(parser, language, LANGUAGE_METADATA[language]['language_parser'])
    return processor.process_directory(repository_dir_path, LANGUAGE_METADATA[language]['ext'])
