import re
from typing import List, Iterable


from code_search import shared, utils


IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
IDENTIFIER_CAMEL_CASE_SPLIT = re.compile('(?<=[a-z0-9])([A-Z])')
PUNCTUATION_TOKENS = set('.?:;,')
CODE_PUNCTUATION = set('(){}[].:=,;$')


def is_comment_token(language: str, token: str) -> bool:
    len_token = len(token)

    if language in ['python', 'ruby', 'php'] and len_token >= 1 and token.startswith('#'):
        return True
    if language in ['java', 'javascript', 'go', 'php'] \
            and len_token >= 2 and (token.startswith('//') or token.startswith('/*')):
        return True

    return False


def remove_inline_comments(language: str, code_tokens: List[str]) -> List[str]:
    return [token for token in code_tokens if not is_comment_token(language, token)]


def extract_sub_tokens(token):
    # Skip strings
    if len(token) > 0 and (token[0] in ['\'', '"'] or token[:2] in ['r\'', 'r"', 'f\'', 'f"']):
        return [token]

    sub_tokens = re.split('[._]', token)
    sub_tokens = [IDENTIFIER_CAMEL_CASE_SPLIT.sub(r' \1', sub_token).split(' ')
                  if IDENTIFIER_TOKEN_REGEX.match(sub_token) else [sub_token]
                  for sub_token in sub_tokens]

    return [token.strip() for token in utils.flatten(sub_tokens) if len(token.strip()) > 0]


def preprocess_code_tokens(tokens: Iterable[str]) -> shared.TokensGenerator:
    for token in tokens:
        token_lower = token.lower().strip()

        yield [token_lower]


def preprocess_query_tokens(tokens: Iterable[str]) -> shared.TokensGenerator:
    for token in tokens:
        token_lower = token.lower().strip()

        yield [token_lower]
