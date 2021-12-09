
from main import run_day

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