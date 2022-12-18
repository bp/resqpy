"""Factorization and functions supporting grid extent determination from corner points."""


def factorize(n):
    """Returns list of prime factors of positive integer n."""
    i = 2
    factors = []
    while True:
        q, r = divmod(n, i)
        if r:
            i += 1
            if i > n:
                return factors
        else:
            factors.append(i)
            if q < i:
                return factors
            n = q


def combinatorial(numbers):
    """Returns a list of all possible product combinations of numbers from list numbers, with some duplicates."""
    if len(numbers) == 0:
        return []
    head = numbers[0]
    c = [head]
    tail = combinatorial(numbers[1:])
    for o in tail:
        if o != head:
            c.append(o)
        c.append(head * o)
    return sorted(c)


def all_factors_from_primes(primes):
    """Returns a sorted list of unique factors from prime factorization."""
    all = list(set(combinatorial(primes)))
    all.append(1)
    return sorted(all)


def all_factors(n):
    """Returns a sorted list of unique factors of n."""
    primes = factorize(n)
    return all_factors_from_primes(primes)


def remove_subset(primary, subset):
    """Remove all elements of subset from primary list."""

    for e in subset:
        primary.remove(e)
