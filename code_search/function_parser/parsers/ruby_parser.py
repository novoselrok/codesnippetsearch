from typing import List, Dict, Any

from code_search.function_parser.parsers.language_parser import LanguageParser, match_from_span, tokenize_code, \
    traverse_type
from code_search.function_parser.parsers.comment_utils import get_docstring_summary


class RubyParser(LanguageParser):
    FILTER_PATHS = ('test', 'vendor')

    BLACKLISTED_FUNCTION_NAMES = {}

    @staticmethod
    def get_docstring(trait_node, blob: str, idx: int) -> str:
        raise NotImplementedError("Not used for Ruby.")

    @staticmethod
    def get_methods(module_or_class_node, blob: str) -> List[Dict[str, Any]]:
        definitions = []
        comment_buffer = []
        module_or_class_name = match_from_span(module_or_class_node.children[1], blob)
        for child in module_or_class_node.children:
            if child.type == 'comment':
                comment_buffer.append(child)
            elif child.type == 'method':
                docstring = '\n'.join([match_from_span(comment, blob).strip().strip('#') for comment in comment_buffer])
                docstring_summary = get_docstring_summary(docstring)

                metadata = RubyParser.get_function_metadata(child, blob)
                if metadata['identifier'] in RubyParser.BLACKLISTED_FUNCTION_NAMES:
                    continue

                definitions.append({
                    'type': 'class',
                    'identifier': '{}.{}'.format(module_or_class_name, metadata['identifier']),
                    'parameters': metadata['parameters'],
                    'function': match_from_span(child, blob),
                    'function_tokens': tokenize_code(child, blob),
                    'docstring': docstring,
                    'docstring_summary': docstring_summary,
                    'start_point': child.start_point,
                    'end_point': child.end_point
                })
                comment_buffer = []
            else:
                comment_buffer = []

        return definitions

    @staticmethod
    def get_definitions(tree, blob: str) -> List[Dict[str, Any]]:
        if 'ERROR' in set([child.type for child in tree.root_node.children]):
            return []

        definitions = []
        modules = []
        traverse_type(tree.root_node, modules, 'module')
        for module in modules:
            if not module.children:
                continue

            definitions.extend(RubyParser.get_methods(module, blob))

            classes = [child for child in module.children if child.type == 'class']
            for class_node in classes:
                definitions.extend(RubyParser.get_methods(class_node, blob))

        return definitions

    @staticmethod
    def get_function_metadata(function_node, blob: str) -> Dict[str, str]:
        metadata = {'identifier': match_from_span(function_node.children[1], blob), 'parameters': ''}
        if function_node.children[2].type == 'method_parameters':
            metadata['parameters'] = match_from_span(function_node.children[2], blob)
        return metadata
