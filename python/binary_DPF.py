import numpy as np
import galois
from polynomial_interpolation_1d import generate_polynomial_lookup


def generate_keys(x, size=256):
    """Generate keys for a binary DPF of size `size`, where P(x) = 1"""

    # Generate a random key for k0 (this will be in binary)
    k0 = np.random.randint(low=0, high=2, size=size)

    # Derive k1 from k0 by flipping the bit that corresponds to x
    k1 = k0.copy()
    k1[x] = not k1[x]
    return (k0, k1)


def evaluate_key(k, x):
    """Evaluate single key `k` at position `x`"""
    return k[x]


def get_next_state(GF, k, current_state, lookup_poly):
    poly = galois.Poly(GF([0]))
    for i in range(len(k)):
        if evaluate_key(k, i) == 1:
            poly += lookup_poly[i]

    return poly(current_state)


if __name__ == "__main__":
    GF = galois.GF(2**16)

    # Interpolation example for hypothetical state machine
    input_token = GF([1, 3, 1, 3, 1, 3])
    curr_state = GF([0, 0, 2, 2, 3, 3])
    next_state = GF([2, 3, 55, 4, 111, 12])

    # Generate lookup table for the state machine
    lookup_poly = generate_polynomial_lookup(GF, input_token, curr_state, next_state)

    # Share token input by selecting correct polynomial from lookup table using DPF and Function Secret Sharing
    chosen_input_token = 3
    k0, k1 = generate_keys(chosen_input_token, size=len(lookup_poly))

    # Evaluate the polynomial at a given input values (checked and seems to work)
    cs0 = get_next_state(GF, k0, 0, lookup_poly)
    cs1 = get_next_state(GF, k1, 0, lookup_poly)
    print("FSS ANSWER:", cs0 + cs1)  # in 2**n prime field + is the same as XOR (^)
    print("REAL ANSWER:", lookup_poly[chosen_input_token](0))

    # Should be at state 3 now...
    # TODO: This is where we need to do the arithmetic sharing I believe

    # Below here does not work yet...
    # k0, k1 = generate_keys(3, size=len(lookup_poly))
    # cs0 = get_next_state(GF, k0, cs0, lookup_poly)
    # cs1 = get_next_state(GF, k1, cs1, lookup_poly)

    # print("FSS ANSWER:", cs0 + cs1)
