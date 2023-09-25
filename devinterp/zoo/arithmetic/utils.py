import math


def is_prime(n: int):
    """Checks if a number is prime."""
    return n > 1 and all(n % i for i in range(2, math.floor(math.sqrt(n))))


def modular_exponentiation(base, exponent, modulus):
    """Computes modular exponentiation using the square-and-multiply algorithm."""
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent // 2
        base = (base * base) % modulus
    return result


def modular_division(a, b, p):
    """Computes modular division using Fermat's little theorem."""
    b_inv = modular_exponentiation(b, p - 2, p)
    return (a * b_inv) % p
