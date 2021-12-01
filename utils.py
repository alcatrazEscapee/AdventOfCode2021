
from typing import Tuple

import re


def get_input(path: str = './input.txt') -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def ints(text: str, sign_prefixes: bool = True) -> Tuple[int, ...]:
    regex = r'([\-+]?\d+)' if sign_prefixes else r'(\d+)'
    return tuple(map(int, re.findall(regex, text)))
