# Day 16: Packet Decoder
# Leaderboard Rank: 96 / 83

from utils import get_input

from math import prod
from collections import deque
from typing import List, Dict, Callable, Deque


class BitQueue:
    """
    A Fifo queue of single bits, built from a MSB -> LSB ordered hex string
    """

    def __init__(self, hex_string: str):
        self.bits: Deque[int] = deque()
        for digit in hex_string:
            value = int(digit, 16)
            for i in range(4):
                self.bits.append((value >> (3 - i)) & 1)

    def pop(self, num_bits: int = 1) -> int:
        """ Pop a single value of num_bits length off the front of the queue """
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.bits.popleft()
        return value

    def __len__(self):
        return len(self.bits)

    def __str__(self):
        return 'BitQueue[' + ''.join(map(str, self.bits)) + ']'


PACKET_OPS: Dict[int, Callable[[List[int]], int]] = {
    0: sum,
    1: prod,
    2: min,
    3: max,
    # 4 represents a literal packet
    5: lambda ps: ps[0] > ps[1],
    6: lambda ps: ps[0] < ps[1],
    7: lambda ps: ps[0] == ps[1]
}

def main(text: str):
    result = packet(BitQueue(text), versions := [])
    print('Part 1:', sum(versions))
    print('Part 2:', int(result))


def packet(bits: BitQueue, versions: List[int]) -> int:
    """ Resolves a single packet recursively. Appends the version to the versions list and returns the packet result """

    version = bits.pop(3)
    type_id = bits.pop(3)

    versions.append(version)

    if type_id == 4:
        # Literal packet
        # Represents a single numeric value, encoded in 5-bit fields of (parity, a0..a3)
        # A field with parity = 1 indicates there are more fields remaining
        parity = 1
        value = 0
        while parity == 1:
            parity = bits.pop()
            value = (value << 4) | bits.pop(4)
        return value
    else:
        # Operator packet
        # Represents an operation to be applied to multiple sub-packets
        sub_packets = []
        if bits.pop() == 0:
            # Length type Id = 0: A single 15-bit field which encodes the total bit count of sub-packets
            value = bits.pop(15)
            remaining = len(bits)
            while remaining - len(bits) < value:
                sub_packets.append(packet(bits, versions))

            assert remaining - len(bits) == value, 'Packet (version=%d, type_id=%d, length_type_id=0, bits=%d) has mismatched sub-packets' % (version, type_id, value)
        else:
            # Length Type Id = 1: A single 11-bit field which encodes the total number of sub-packets
            value = bits.pop(11)
            for _ in range(value):
                sub_packets.append(packet(bits, versions))

        return PACKET_OPS[type_id](sub_packets)


if __name__ == '__main__':
    main(get_input())
