import os

from django import template
from pygments import highlight
from pygments.lexers import get_lexer_by_name

from code_search_app.shared import get_pygments_html_formatter

register = template.Library()


@register.filter(name='syntax_highlight')
def syntax_highlight(code_document):
    lexer = get_lexer_by_name(code_document.language, startinline=True)
    formatter = get_pygments_html_formatter()
    return highlight(code_document.code, lexer, formatter)


@register.filter(name='basename')
def basename(file_path):
    return os.path.basename(file_path)


@register.filter(name='cosine_match_rating')
def cosine_match_rating(distance):
    # 2.0 is the maximum cosine distance
    return str(round(((2.0 - distance) / 2.0) * 100, 2))


@register.filter(name='range')
def num_range(max_num):
    return range(max_num)


@register.filter(name='get_item')
def get_item(dictionary, key):
    return dictionary.get(key)
