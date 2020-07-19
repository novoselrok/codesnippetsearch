from code_search.function_parser.parsers.go_parser import GoParser
from code_search.function_parser.parsers.java_parser import JavaParser
from code_search.function_parser.parsers.javascript_parser import JavascriptParser
from code_search.function_parser.parsers.php_parser import PhpParser
from code_search.function_parser.parsers.python_parser import PythonParser
from code_search.function_parser.parsers.ruby_parser import RubyParser


LANGUAGE_METADATA = {
    'python': {
        'platform': 'pypi',
        'ext': 'py',
        'language_parser': PythonParser
    },
    'java': {
        'platform': 'maven',
        'ext': 'java',
        'language_parser': JavaParser
    },
    'go': {
        'platform': 'go',
        'ext': 'go',
        'language_parser': GoParser
    },
    'javascript': {
        'platform': 'npm',
        'ext': 'js',
        'language_parser': JavascriptParser
    },
    'php': {
        'platform': 'packagist',
        'ext': 'php',
        'language_parser': PhpParser
    },
    'ruby': {
        'platform': 'rubygems',
        'ext': 'rb',
        'language_parser': RubyParser
    }
}
