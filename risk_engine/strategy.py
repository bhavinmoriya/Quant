# strategy.py
import numpy as np

def cvar_signal(returns, window=252, alpha=0.95, threshold=-0.03):
    signals = []
    
    for i in range(window, len(returns)):
        window_data = returns[i-window:i]
        tail_loss = window_data[window_data <= np.quantile(window_data, 1-alpha)].mean()
        
        if tail_loss < threshold:
            signals.append(0)  # risk-off
        else:
            signals.append(1)  # risk-on
            
    return np.array(signals)
