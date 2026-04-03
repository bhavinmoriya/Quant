def kelly_fraction(p, b):
    """
    p: probability of winning
    b: odds (profit per unit bet)
    """
    q = 1 - p
    return (b * p - q) / b
