import numpy as np
import galois


if __name__ == "__main__":
    # Create a Galois field (this is a finite field used to constrain the values of the polynomial)
    P = 65521  # This prime is less than 2^16, so we can use uint16 (potential efficiency gain)
    GF = galois.GF(P)
    NUM_INPUT_TOKEN = 256  # assume that there are 256 possible input tokens

    # Interpolation example for hypothetical state machine
    input_token = GF([1, 2, 1, 2, 1, 2])
    curr_state = GF([5, 5, 2, 2, 3, 3])
    next_state = GF([1, 2, 55, 4, 111, 12])

    # Instead of interpolating a 2 dimensional polynomial, we can interpolate a 1 dimensional polynomial for each x
    # In the case of our state machine, we can interpolate a 1 dimensional polynomial for each input token
    used_input_tokens = np.unique(input_token)
    used_states = np.unique(curr_state)

    lookup_poly = [-1] * NUM_INPUT_TOKEN
    for token in used_input_tokens:
        # Get the x and y values for this token
        x = curr_state[input_token == token]
        y = next_state[input_token == token]

        # Interpolate the polynomial and place in the lookup table
        lookup_poly[token] = galois.lagrange_poly(x, y)

    # Evaluate the polynomial at a given x value
    print(lookup_poly[2](5))
    print(lookup_poly[2](2))
    print(lookup_poly[2](3))
    #
    print(lookup_poly[1](5))
    print(lookup_poly[1](2))
    print(lookup_poly[1](3))
