
from typing import Tuple, List

import re


def get_input(path: str = './input.txt') -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def get_input_lines(path: str = './input.txt') -> List[str]:
    return get_input(path).split('\n')

def ints(text: str, sign_prefixes: bool = True) -> Tuple[int, ...]:
    regex = r'([\-+]?\d+)' if sign_prefixes else r'(\d+)'
    return tuple(map(int, re.findall(regex, text)))
