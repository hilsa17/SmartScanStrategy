import numpy as np

def estimate_t_scan(toas_sec, min_t=0.2, max_t=5.0, num_steps=10000):
    """
    Finds the radar rotation period from sparse, jittery timestamps.
    Uses Phase Coherence (Point-Process Periodogram) instead of standard Lomb-Scargle,
    because we are analyzing the periodicity of the events, not their amplitudes.
    """
    f_min = 1.0 / max_t
    f_max = 1.0 / min_t
    freqs_hz = np.linspace(f_min, f_max, num_steps)
   
    # Power(f) = | sum( exp(-j * 2 * pi * f * t) ) |^2
    # We test every frequency. The correct frequency will cause all the timestamp
    # phasors to align and point in the exact same direction, maximizing the sum.
   
    best_power = 0
    best_f = f_min
   
    # Vectorized computation for speed
    for f in freqs_hz:
        phasors = np.exp(-1j * 2.0 * np.pi * f * toas_sec)
        power = np.abs(np.sum(phasors))**2
       
        if power > best_power:
            best_power = power
            best_f = f
           
    t_scan = 1.0 / best_f
    return t_scan
