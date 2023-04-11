import sys
from numba import jit


@jit(nopython=True)
def f(x, y):
    return x + y


def main() -> int:
    print("Hello World!")
    print(f(1, 2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
