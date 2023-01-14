import numpy as np
import galois


def generate_polynomial_lookup(GF, x, y, z):
    """Generate a polynomial lookup table that interpolates the given points. This allows for a 2 dimensional
    polynomial to be placed into a 1 dimensional lookup table."""
    assert len(x) == len(y) == len(z), "Coordinate arrays must be the same length"

    # Instead of interpolating a 2 dimensional polynomial, we can interpolate a 1 dimensional polynomial for each x
    # In the case of our state machine, we can interpolate a 1 dimensional polynomial for each input token
    x_vals = np.unique(x)

    lookup_poly = [galois.Poly(GF([0]))] * (np.max(x_vals) + 1)
    for x_val in x_vals:
        # Get the x and y values for this token
        selected_x = y[x == x_val]
        selected_y = z[x == x_val]

        # Interpolate the polynomial and place in the lookup table
        lookup_poly[x_val] = galois.lagrange_poly(selected_x, selected_y)

    return lookup_poly


if __name__ == "__main__":
    # Create a Galois field (this is a finite field used to constrain the values of the polynomial)
    P = 65521  # This prime is less than 2^16, so we can use uint16 (potential efficiency gain)
    GF = galois.GF(P)

    # Interpolation example for hypothetical state machine
    input_token = GF([1, 3, 1, 3, 1, 3])
    curr_state = GF([5, 5, 2, 2, 3, 3])
    next_state = GF([1, 2, 55, 4, 111, 12])

    lookup_poly = generate_polynomial_lookup(GF, input_token, curr_state, next_state)

    # Evaluate the polynomial at a given input values (checked and seems to work)
    print(lookup_poly[3](5))
    print(lookup_poly[3](2))
    print(lookup_poly[3](3))
    #
    print(lookup_poly[1](5))
    print(lookup_poly[1](2))
    print(lookup_poly[1](3))

    print(lookup_poly)
