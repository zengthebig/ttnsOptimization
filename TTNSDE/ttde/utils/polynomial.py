from jax import numpy as jnp
# 多项式基本操作 
# We represent a polynomial by a 1D array of coefficients in *descending* powers:
#   coeffs = [a0, a1, ..., an]  <=>  p(x) = a0*x^n + a1*x^(n-1) + ... + an
# This matches jnp.polyval / jnp.polyadd / jnp.polymul conventions.
Polynomial = jnp.ndarray


def poly_x() -> Polynomial: # [1,0]
    """
    Return the polynomial p(x) = x.

    In descending-power coefficient format:
        x = 1*x + 0  -> [1, 0]
    """
    return jnp.array([1, 0])


def poly_int(coeffs: Polynomial) -> Polynomial: # 多项式的积分的系数
    """
    Compute coefficients of an antiderivative (indefinite integral) of p(x),
    with integration constant fixed to 0.

    Input:
        coeffs: [a0, a1, ..., an] representing p(x) = a0*x^n + ... + an
    Output:
        integral_coeffs: coefficients of P(x) = ∫ p(x) dx, in descending powers,
                         i.e. P'(x)=p(x), with constant term = 0.

    Example:
        p(x) = 2x^2 + 3x + 4  -> coeffs = [2, 3, 4]
        ∫p = (2/3)x^3 + (3/2)x^2 + 4x + 0  -> [2/3, 3/2, 4, 0]
    """
    # For descending-power coefficients of length L, degrees are:
    #   L-1, L-2, ..., 0
    # After integration, each coefficient is divided by (degree+1):
    #   (L-1)+1 = L, (L-2)+1 = L-1, ..., 0+1 = 1
    denom = jnp.arange(len(coeffs), 0, -1)  # [L, L-1, ..., 1]

    # Divide each coefficient by its (degree+1), then append constant term 0.
    return jnp.concatenate([coeffs / denom, jnp.zeros(1)])


def poly_definite_int(coeffs: Polynomial, l: float, r: float) -> float: # 多项式定积分的值
    """
    Compute the definite integral ∫_{l}^{r} p(x) dx for polynomial p.

    Steps:
        1) Build an antiderivative P(x) with poly_int (constant fixed to 0).
        2) Return P(r) - P(l).

    Note:
        Uses jnp.polyval which expects descending-power coefficients.
    """
    integral = poly_int(coeffs)
    return jnp.polyval(integral, r) - jnp.polyval(integral, l)


def poly_shift(p: Polynomial, h: float) -> Polynomial: 
    """
    Shift the polynomial argument:
        p(x)  ->  p(x - h)

    Input:
        p: coefficients of p(x) in descending powers
        h: shift amount
    Output:
        coefficients of q(x) = p(x - h) (still descending powers)

    Implementation idea:
        Write p(x) in ascending form around (x-h):
            p(x-h) = sum_{i=0}^{deg} c_i * (x-h)^i
        We build powers (x-h)^i iteratively:
            (x-h)^0 = 1
            (x-h)^{i+1} = (x-h)^i * (x-h)
        and accumulate c_i * (x-h)^i into the result.

    Why p[-i-1]?
        Since p is descending-power [a0, ..., an], p[-1] is the constant term,
        p[-2] is the x^1 coefficient, etc. Iterating i from 0 upward matches
        the increasing powers (x-h)^i we build.
    """
    # Result polynomial accumulator, start from 0 (constant polynomial).
    res = jnp.zeros([1])

    # Polynomial (x - h) in descending-power coefficients: [1, -h]
    x_m_h = jnp.array([1, -h])

    # Current power (x - h)^i, initialized at i=0 -> 1
    x_m_h_p = jnp.ones([1])

    # Accumulate p(x-h) = sum_{i>=0} (coeff_of_x^i in p) * (x-h)^i
    for i in range(len(p)):
        # Take coefficient of x^i from p:
        #   i=0 -> constant term p[-1]
        #   i=1 -> x term       p[-2]
        #   ...
        ci = p[-i - 1]

        # Add ci * (x-h)^i to the accumulator.
        res = jnp.polyadd(res, x_m_h_p * ci)

        # Update (x-h)^i -> (x-h)^(i+1)
        x_m_h_p = jnp.polymul(x_m_h_p, x_m_h)

    return res
