Let's define $E$ as the expected number of additional flips needed to get 'HH' (two consecutive heads), starting from the beginning (no previous heads).

We can define three states:
*   **State 0 (E₀):** No previous heads, or the last flip was a 'T'. This is our starting state.
*   **State 1 (E₁):** The last flip was an 'H', but the flip before that was a 'T' (or it was the first flip).
*   **State 2 (E₂):** We have achieved 'HH'. This is our stopping state, so $E_2 = 0$.

Now, let's set up equations for the expected number of additional flips from each state, assuming the probability of 'H' (p) and 'T' (1-p) are both 0.5 for a fair coin:

**From State 0 (E₀):**
*   Flip 'H' (probability 0.5): We are now in State 1. It took 1 flip, and we need $E_1$ more flips.
*   Flip 'T' (probability 0.5): We are still in State 0. It took 1 flip, and we need $E_0$ more flips.

So, the equation for $E_0$ is:
$E_0 = 1 + 0.5 \cdot E_1 + 0.5 \cdot E_0$

**From State 1 (E₁):**
*   Flip 'H' (probability 0.5): We have achieved 'HH'. It took 1 flip, and we need $E_2$ (which is 0) more flips.
*   Flip 'T' (probability 0.5): We are back to State 0 (because the consecutive 'H' streak is broken). It took 1 flip, and we need $E_0$ more flips.

So, the equation for $E_1$ is:
$E_1 = 1 + 0.5 \cdot E_2 + 0.5 \cdot E_0$

Since $E_2 = 0$, this simplifies to:
$E_1 = 1 + 0.5 \cdot 0 + 0.5 \cdot E_0$
$E_1 = 1 + 0.5 E_0$

Now we have a system of two linear equations:
1.  $E_0 = 1 + 0.5 E_1 + 0.5 E_0$
2.  $E_1 = 1 + 0.5 E_0$

Let's solve these equations.

E0 (Expected flips from start): 6.00000000000000
E1 (Expected flips after one H): 4.00000000000000
