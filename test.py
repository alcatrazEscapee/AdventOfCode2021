
from main import run_day, run_day_with_example

# Tests for all AoC Puzzles

def test_day1():
    assert run_day(1, '199 200 208 210 200 207 240 269 260 263') == ('7', '5')
    assert run_day(1) == ('1482', '1518')

def test_day2():
    assert run_day(2) == ('1714680', '1963088820')

def test_day3():
    assert run_day(3) == ('3374136', '4432698')

def test_day4():
    assert run_day(4) == ('41503', '3178')

def test_day5():
    assert run_day(5) == ('7644', '18627')

def test_day6():
    assert run_day(6) == ('385391', '1728611055389')

def test_day7():
    assert run_day(7) == ('325528', '85015836')

def test_day8():
    assert run_day(8) == ('381', '1023686')

def test_day9():
    assert run_day(9) == ('566', '891684')

def test_day10():
    assert run_day(10) == ('469755', '2762335572')

def test_day11():
    assert run_day(11) == ('1683', '788')

def test_day12():
    assert run_day_with_example(12, 1) == ('10', '36')
    assert run_day_with_example(12, 2) == ('19', '103')
    assert run_day_with_example(12, 3) == ('226', '3509')
    assert run_day(12) == ('4338', '114189')

def test_day13():
    assert run_day(13) == ('810', 'HLBUBGFR')

def test_day14():
    assert run_day_with_example(14, 1) == ('1588', '2188189693529')
    assert run_day(14) == ('2712', '8336623059567')

def test_day15():
    assert run_day_with_example(15, 1) == ('40', '315')
    assert run_day(15) == ('503', '2853')

def test_day16():
    assert run_day(16, 'D2FE28') == ('6', '2021')
    assert run_day(16, '38006F45291200') == ('9', '1')
    assert run_day(16, 'EE00D40C823060') == ('14', '3')

    assert run_day(16, '8A004A801A8002F478')[0] == '16'
    assert run_day(16, '620080001611562C8802118E34')[0] == '12'
    assert run_day(16, 'C0015000016115A2E0802F182340')[0] == '23'
    assert run_day(16, 'A0016C880162017C3686B18A3D4780')[0] == '31'

    assert run_day(16, 'C200B40A82')[1] == '3'
    assert run_day(16, '04005AC33890')[1] == '54'
    assert run_day(16, '880086C3E88112')[1] == '7'
    assert run_day(16, 'CE00C43D881120')[1] == '9'
    assert run_day(16, 'D8005AC2A8F0')[1] == '1'
    assert run_day(16, 'F600BC2D8F')[1] == '0'
    assert run_day(16, '9C005AC2F8F0')[1] == '0'
    assert run_day(16, '9C0141080250320F1802104A08')[1] == '1'

    assert run_day(16) == ('879', '539051801941')
