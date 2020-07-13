import os
import itertools
from typing import Iterable
from multiprocessing import Pool

from code_search import shared


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def flatten(iterable: Iterable[Iterable]) -> Iterable:
    return itertools.chain.from_iterable(iterable)


def len_generator(generator):
    return sum(1 for _ in generator)


def _multiprocess_map_method(args):
    obj, method_name, arg = args
    method = getattr(obj, method_name)
    method(*arg)


def map_method(obj, method_name: str, args: Iterable, num_processes=4):
    if num_processes > 1:
        with Pool(num_processes) as p:
            p.map(_multiprocess_map_method, ((obj, method_name, arg) for arg in args))
    else:
        map(lambda arg: getattr(obj, method_name)(*arg), args)


def get_base_language_serialized_data_path(language: str):
    return os.path.join(shared.SERIALIZED_DATA_DIR, 'languages', language)


def get_evaluation_queries():
    with open(os.path.join(shared.CODESEARCHNET_DATA_DIR, 'queries.csv'), encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[1:]]
