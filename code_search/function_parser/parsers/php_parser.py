from typing import List, Dict, Any

from code_search.function_parser.parsers.language_parser import LanguageParser, match_from_span, tokenize_code
from code_search.function_parser.parsers.comment_utils import strip_c_style_comment_delimiters, get_docstring_summary


class PhpParser(LanguageParser):

    FILTER_PATHS = ('test', 'tests')

    BLACKLISTED_FUNCTION_NAMES = {'__construct', '__destruct', '__call', '__callStatic',
                                  '__get', '__set', '__isset', '__unset',
                                  '__sleep', '__wakeup', '__toString', '__invoke',
                                  '__set_state', '__clone', '__debugInfo', '__serialize',
                                  '__unserialize'}

    @staticmethod
    def get_docstring(trait_node, blob: str, idx: int) -> str:
        docstring = ''
        if idx - 1 >= 0 and trait_node.children[idx - 1].type == 'comment':
            docstring = match_from_span(trait_node.children[idx - 1], blob)
            docstring = strip_c_style_comment_delimiters(docstring)
        return docstring

    @staticmethod
    def get_declarations(declaration_node, blob: str, node_type: str) -> List[Dict[str, Any]]:
        declaration_lists = [child for child in declaration_node.children if child.type == 'declaration_list']
        if len(declaration_lists) == 0:
            return []
        declaration_list = declaration_lists[0]
        declaration_name = PhpParser.get_declaration_name(declaration_node, blob)

        declarations = []
        for idx, child in enumerate(declaration_list.children):
            if child.type == 'method_declaration':
                docstring = PhpParser.get_docstring(declaration_list, blob, idx)
                docstring_summary = get_docstring_summary(docstring)
                metadata = PhpParser.get_function_metadata(child, blob)

                declarations.append({
                    'type': child.type,
                    'identifier': '{}.{}'.format(declaration_name, metadata['identifier']),
                    'parameters': metadata['parameters'],
                    'function': match_from_span(child, blob),
                    'function_tokens': tokenize_code(child, blob),
                    'docstring': docstring,
                    'docstring_summary': docstring_summary,
                    'start_point': child.start_point,
                    'end_point': child.end_point
                })

        return declarations

    @staticmethod
    def get_definitions(tree, blob: str) -> List[Dict[str, Any]]:
        trait_declarations = [child for child in tree.root_node.children if child.type == 'trait_declaration']
        class_declarations = [child for child in tree.root_node.children if child.type == 'class_declaration']
        definitions = []
        for trait_declaration in trait_declarations:
            definitions.extend(PhpParser.get_declarations(trait_declaration, blob, trait_declaration.type))
        for class_declaration in class_declarations:
            definitions.extend(PhpParser.get_declarations(class_declaration, blob, class_declaration.type))
        return definitions

    @staticmethod
    def get_declaration_name(declaration_node, blob: str):
        for child in declaration_node.children:
            if child.type == 'name':
                return match_from_span(child, blob)
        return ''

    @staticmethod
    def get_function_metadata(function_node, blob: str) -> Dict[str, str]:
        metadata = {'identifier': match_from_span(function_node.children[1], blob),
                    'parameters': match_from_span(function_node.children[2], blob)}
        return metadata
