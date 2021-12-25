
from main import run_day, run_day_with_example, get_day_input, get_example

from day18 import day18
from day22 import day22
from day23 import day23
from day24 import day24

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

def test_day17():
    assert run_day(17, 'target area: x=20..30, y=-10..-5') == ('45', '112')
    assert run_day(17, 'target area: x=352..377, y=-49..-30') == ('66', '820')  # Edge case inducing input from https://www.reddit.com/r/adventofcode/comments/rid0g3/2021_day_17_part_1_an_input_that_might_break_your/
    assert run_day(17) == ('15931', '2555')

def test_day18_explode():
    assert day18_explode([[[[[9, 8], 1], 2], 3], 4]) == [[[[0, 9], 2], 3], 4]
    assert day18_explode([7, [6, [5, [4, [3, 2]]]]]) == [7, [6, [5, [7, 0]]]]
    assert day18_explode([[6, [5, [4, [3, 2]]]], 1]) == [[6, [5, [7, 0]]], 3]
    assert day18_explode([[3, [2, [1, [7, 3]]]], [6, [5, [4, [3, 2]]]]]) == [[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]]
    assert day18_explode([[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]]) == [[3, [2, [8, 0]]], [9, [5, [7, 0]]]]
    assert day18_explode([[[[[4, 3], 4], 4], [7, [[8, 4], 9]]], [1, 1]]) == [[[[0, 7], 4], [7, [[8, 4], 9]]], [1, 1]]
    assert day18_explode([[[[0, 7], 4], [7, [[8, 4], 9]]], [1, 1]]) == [[[[0, 7], 4], [15, [0, 13]]], [1, 1]]

def test_day18_split():
    assert day18_split([[[[0, 7], 4], [15, [0, 13]]], [1, 1]]) == [[[[0, 7], 4], [[7, 8], [0, 13]]], [1, 1]]
    assert day18_split([[[[0, 7], 4], [[7, 8], [0, 13]]], [1, 1]]) == [[[[0, 7], 4], [[7, 8], [0, [6, 7]]]], [1, 1]]

def test_day18_magnitude():
    assert day18_magnitude([[1, 2], [[3, 4], 5]]) == 143
    assert day18_magnitude([[[[0, 7], 4], [[7, 8], [6, 0]]], [8, 1]]) == 1384
    assert day18_magnitude([[[[1, 1], [2, 2]], [3, 3]], [4, 4]]) == 445
    assert day18_magnitude([[[[3, 0], [5, 3]], [4, 4]], [5, 5]]) == 791
    assert day18_magnitude([[[[5, 0], [7, 4]], [5, 5]], [6, 6]]) == 1137
    assert day18_magnitude([[[[8, 7], [7, 7]], [[8, 6], [7, 7]]], [[[0, 7], [6, 6]], [8, 7]]]) == 3488

def test_day18():
    assert run_day(18) == ('4017', '4583')

def day18_explode(lists: day18.SnailfishNumber) -> day18.SnailfishNumber:
    day18.try_explode(nodes := day18.convert(lists))
    return nodes.listify()

def day18_split(lists: day18.SnailfishNumber) -> day18.SnailfishNumber:
    day18.try_split(nodes := day18.convert(lists))
    return nodes.listify()

def day18_magnitude(lists: day18.SnailfishNumber) -> int:
    return day18.convert(lists).magnitude()

def test_day19():
    assert run_day_with_example(19, 0) == ('79', '3621')
    assert run_day(19) == ('365', '11060')

def test_day20():
    assert run_day_with_example(20, 0) == ('35', '3351')
    assert run_day(20) == ('5846', '21149')

def test_day21():
    assert run_day(21, '1 4\n 2 8') == ('739785', '444356092776315')
    assert run_day(21) == ('571032', '49975322685009')

def test_day22():
    assert run_day_with_example(22, 1) == ('39', '39')
    assert run_day_with_example(22, 2) == ('590784', '39769202357779')
    assert run_day_with_example(22, 3) == ('474140', '2758514936282235')
    assert run_day(22) == ('570915', '1268313839428137')

def test_day22_coordinate_subdivisions():
    assert day22_coordinate_subdivision(1) == 39
    assert day22_coordinate_subdivision(2) == 39769202357779
    assert day22_coordinate_subdivision(3) == 2758514936282235

def day22_coordinate_subdivision(example: int) -> int:
    return day22.solve_part2_coordinate_subdivision(day22.parse(get_example(22, example)))

def test_day23():
    assert run_day_with_example(23, 1) == ('12521', '44169')
    assert run_day(23) == ('11120', '49232')

def test_day23_to_index_to_position():
    for i in range(1 + 10):
        assert day23.to_index(i, 0) == i
        assert day23.to_position(i) == (i, 0)

    for i, x in zip(range(4), (2, 4, 6, 8)):
        assert day23.to_index(x, 1) == 11 + i
        assert day23.to_index(x, 2) == 15 + i
        assert day23.to_position(11 + i) == (x, 1)
        assert day23.to_position(15 + i) == (x, 2)

def test_day24():
    assert run_day(24) == ('92969593497992', '81514171161381')

def test_day24_interpret_answer():
    inp = get_day_input(24)
    assert day24.run(inp, 92969593497992) == 0
    assert day24.run(inp, 81514171161381) == 0

def test_day25():
    assert run_day_with_example(25, 1) == ('58', None)
    assert run_day(25) == ('337', None)
