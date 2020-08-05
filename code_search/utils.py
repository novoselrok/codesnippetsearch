import os
import itertools
from typing import Iterable, Dict, List
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


def get_values_sorted_by_key(dict_: Dict[int, str]) -> List[str]:
    return [value for _, value in sorted(dict_.items())]


def _multiprocess_map_method(args):
    obj, method_name, arg = args
    method = getattr(obj, method_name)
    method(*arg)


def map_method(obj, method_name: str, args: Iterable, num_processes=4):
    if num_processes > 1:
        with Pool(num_processes) as p:
            p.map(_multiprocess_map_method, ((obj, method_name, arg) for arg in args))
    else:
        list(map(lambda arg: getattr(obj, method_name)(*arg), args))


def get_repository_directory(organization: str, name: str):
    return os.path.join(shared.REPOSITORIES_DIR, organization, name)
