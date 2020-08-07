from typing import Dict, Iterable, Optional, Iterator, Any, List

from code_search.function_parser.parsers.language_parser import LanguageParser, match_from_span, tokenize_code
from code_search.function_parser.parsers.comment_utils import get_docstring_summary


class PythonParser(LanguageParser):
    @staticmethod
    def __get_docstring_node(function_node):
        block_nodes = [node for node in function_node.children if node.type == 'block']
        if len(block_nodes) == 0:
            return

        block_node = block_nodes[0]
        docstring_node = [node for node in block_node.children if
                          node.type == 'expression_statement' and node.children[0].type == 'string']

        if len(docstring_node) > 0:
            return docstring_node[0].children[0]

        return None

    @staticmethod
    def get_docstring(docstring_node, blob: str) -> str:
        docstring = ''
        if docstring_node is not None:
            docstring = match_from_span(docstring_node, blob)
            docstring = docstring.strip().strip('"').strip("'")
        return docstring

    @staticmethod
    def get_function_metadata(function_node, blob: str) -> Dict[str, str]:
        metadata = {
            'identifier': '',
            'parameters': '',
            'return_statement': ''
        }
        is_header = False
        for child in function_node.children:
            if is_header:
                if child.type == 'identifier':
                    metadata['identifier'] = match_from_span(child, blob)
                elif child.type == 'parameters':
                    metadata['parameters'] = match_from_span(child, blob)
            if child.type == 'def':
                is_header = True
            elif child.type == ':':
                is_header = False
            elif child.type == 'return_statement':
                metadata['return_statement'] = match_from_span(child, blob)
        return metadata

    @staticmethod
    def get_class_metadata(class_node, blob: str) -> Dict[str, str]:
        metadata = {
            'identifier': '',
            'argument_list': '',
        }
        is_header = False
        for child in class_node.children:
            if is_header:
                if child.type == 'identifier':
                    metadata['identifier'] = match_from_span(child, blob)
                elif child.type == 'argument_list':
                    metadata['argument_list'] = match_from_span(child, blob)
            if child.type == 'class':
                is_header = True
            elif child.type == ':':
                break
        return metadata

    @staticmethod
    def is_function_empty(function_node) -> bool:
        seen_header_end = False
        for child in function_node.children:
            if seen_header_end and (child.type == 'pass_statement' or child.type == 'raise_statement'):
                return True
            elif seen_header_end:
                return False

            if child.type == ':':
                seen_header_end = True
        return False

    @staticmethod
    def __process_functions(functions: Iterable, blob: str, func_identifier_scope: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        for function_node in functions:
            if PythonParser.is_function_empty(function_node):
                continue
            function_metadata = PythonParser.get_function_metadata(function_node, blob)
            if func_identifier_scope is not None:
                function_metadata['identifier'] = '{}.{}'.format(func_identifier_scope,
                                                                 function_metadata['identifier'])

            docstring_node = PythonParser.__get_docstring_node(function_node)
            function_metadata['docstring'] = PythonParser.get_docstring(docstring_node, blob)
            function_metadata['docstring_summary'] = get_docstring_summary(function_metadata['docstring'])
            function_metadata['function'] = match_from_span(function_node, blob)
            function_metadata['function_tokens'] = tokenize_code(function_node, blob)
            function_metadata['start_point'] = function_node.start_point
            function_metadata['end_point'] = function_node.end_point

            yield function_metadata

    @staticmethod
    def get_function_definitions(node):
        for child in node.children:
            if child.type == 'function_definition':
                yield child
            elif child.type == 'decorated_definition' or child.type == 'block':
                for c in child.children:
                    if c.type == 'function_definition':
                        yield c

    @staticmethod
    def get_class_definitions(node):
        for child in node.children:
            if child.type == 'class_definition':
                yield child
            elif child.type == 'decorated_definition':
                for c in child.children:
                    if c.type == 'class_definition':
                        yield c

    @staticmethod
    def get_definitions(tree, blob: str) -> List[Dict[str, Any]]:
        functions = PythonParser.get_function_definitions(tree.root_node)
        classes = PythonParser.get_class_definitions(tree.root_node)

        definitions = list(PythonParser.__process_functions(functions, blob))

        for _class in classes:
            class_metadata = PythonParser.get_class_metadata(_class, blob)
            docstring_node = PythonParser.__get_docstring_node(_class)
            class_metadata['docstring'] = PythonParser.get_docstring(docstring_node, blob)
            class_metadata['docstring_summary'] = get_docstring_summary(class_metadata['docstring'])
            class_metadata['function'] = ''
            class_metadata['function_tokens'] = []
            class_metadata['start_point'] = _class.start_point
            class_metadata['end_point'] = _class.end_point
            definitions.append(class_metadata)

            functions = PythonParser.get_function_definitions(_class)
            definitions.extend(PythonParser.__process_functions(functions, blob, class_metadata['identifier']))

        return definitions
