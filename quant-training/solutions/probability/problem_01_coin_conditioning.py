def analytical_solution():
    """
    Sample space:
    HH, HT, TH, TT
    
    Condition: at least one H → remove TT
    
    Remaining: HH, HT, TH → 3 outcomes
    Favorable: HH → 1
    
    Probability = 1/3
    """
    return 1 / 3
