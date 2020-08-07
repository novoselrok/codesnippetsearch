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
        'language_parser': PythonParser,
        'ignore_paths': ['test']
    },
    'java': {
        'platform': 'maven',
        'ext': 'java',
        'language_parser': JavaParser,
        'ignore_paths': ['test', 'tests']
    },
    'go': {
        'platform': 'go',
        'ext': 'go',
        'language_parser': GoParser,
        'ignore_paths': ['test', 'vendor']
    },
    'javascript': {
        'platform': 'npm',
        'ext': 'js',
        'language_parser': JavascriptParser,
        'ignore_paths': ['vendor', 'test', 'node_modules', '.min', 'dist']
    },
    'php': {
        'platform': 'packagist',
        'ext': 'php',
        'language_parser': PhpParser,
        'ignore_paths': ['test', 'tests']
    },
    'ruby': {
        'platform': 'rubygems',
        'ext': 'rb',
        'language_parser': RubyParser,
        'ignore_paths': ['test', 'vendor']
    }
}
