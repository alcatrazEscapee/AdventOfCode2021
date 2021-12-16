from utils import get_input
from math import prod

def main(text: str):
    bits = []
    for c in text:
        h = int(c, 16)
        for i in range(4):
            bits.append((h >> (3 - i)) & 1)

    print(''.join(map(str, bits)))

    vers = []
    print(bits)
    ps = packet(bits, vers)
    print('sum', sum(vers))
    print('s', ps)

def packet(bits, vers):
    a0 = bits.pop(0)
    a1 = bits.pop(0)
    a2 = bits.pop(0)
    ver = (a0 << 2) | (a1 << 1) | a2

    a0 = bits.pop(0)
    a1 = bits.pop(0)
    a2 = bits.pop(0)
    type_id = (a0 << 2) | (a1 << 1) | a2
    #print('packet ver', ver, 'type', type_id, bits)

    vers.append(ver)

    if type_id == 4:
        # literal
        #print('literal', bits)
        num = 0
        while True:
            a0 = bits.pop(0)
            a1 = bits.pop(0)
            a2 = bits.pop(0)
            a3 = bits.pop(0)
            a4 = bits.pop(0)
            num = (num << 4) | (a1 << 3) | (a2 << 2) | (a3 << 1) | a4
            if a0 == 0:
                break
        return num
    else:
        nums = []

        # operator
        #print('operator', bits)
        len_type_id = bits.pop(0)
        # 0 = total length, 15 bits
        if len_type_id == 0:
            num = 0
            for i in range(15):
                b = bits.pop(0)
                num = (num << 1) | b

            curr_rem = len(bits)
            while curr_rem - len(bits) < num:
                nums.append(packet(bits, vers))
            assert curr_rem - len(bits) == num
        # 1 = number of subpackets
        else:
            num = 0
            for i in range(11):
                b = bits.pop(0)
                num = (num << 1) | b

            for subp in range(num):
                nums.append(packet(bits, vers))

        #print('nums', nums)
        if type_id == 0:
            return sum(nums)
        elif type_id == 1:
            return prod(nums)
        elif type_id == 2:
            return min(nums)
        elif type_id == 3:
            return max(nums)
        elif type_id == 5:
            return 1 if nums[0] > nums[1] else 0
        elif type_id == 6:
            return 1 if nums[0] < nums[1] else 0
        elif type_id == 7:
            return 1 if nums[0] == nums[1] else 0
        assert False, 'wut: ' + str(type_id)


if __name__ == '__main__':
    main(get_input())
