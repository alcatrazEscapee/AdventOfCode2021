import numpy as np

from utils import get_input, ints


def main():
    """
    Goal: develop an analytical solution for the problem.
    First, describe the recurrence relation. Note that:

    a0,n = a1,n-1
    a1,n = a2,n-1
    ...
    a6,n = a7,n-1 + a0,n-1
    a7,n = a8,n-1
    a8,n = a0,n-1

    Where ai,j := the number of fish at time i, with lifetime j
    Then, define yn := a0,n
    The above system can then be reduced down to the following single recurrence relation:

    yn = yn-7 + yn-9

    The solution to this is the equation:

    yn = c1r1^n + c2r2^n + ... + c9r9^n
    where c1, ... c9 are some arbitrary coefficients
          r1, ... rn are the roots of the polynomial x^9 - x^2 - 1 = 0

    These roots are then computed numerically, and then we need to solve for the coefficients.
    In order to do that, we need the first y1, ... y9 values, which can be inferred from the input data
    Then we solve the matrix equation:

    [ r1^1 ... r9^1 ] [ c1 ]   [ y1 ]
    [ ...      ...  ] [ .. ] = [ .. ]
    [ r1^n ... r9^n ] [ c9 ]   [ y9 ]

    This gives us the values of c1, ... c9.
    Now, with an explicit formula for yn, we can define Kn = the sum of a0,n + ... + a8,n
    Using the definition of yn, this becomes:

    Kn = yn+5 + ... + yn + 2*yn-1 + yn-2 + yn-3

    The solution is then rounded to the nearest complex integer, and the real part is now our answer.
    """

    values = ints(get_input())

    ys = np.array([values.count(n % 7) for n in range(9)])
    rs = np.roots((1, 0, 0, 0, 0, 0, 0, -1, 0, -1))  # x^9 - x^2 - 1 = 0
    A = np.array([[pow(r, n) for r in rs] for n in range(9)])
    cs = np.linalg.solve(A, ys)  # Ax = b

    # Note that both rs is constant (across all inputs), and cs is constant (across any specific input).
    # So solving any value of n can be done just with those precomputed values.

    print('Part 1:', solve(80, rs, cs))
    print('Part 2:', solve(256, rs, cs))

def solve(n: int, rs, cs) -> int:
    return round(sum((2 if j == -1 else 1) * sum(c * pow(r, n + j) for r, c in zip(rs, cs)) for j in range(-3, 6)).real)


if __name__ == '__main__':
    main()
