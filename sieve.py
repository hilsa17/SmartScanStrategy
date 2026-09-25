import numpy as np

def is_match(pdw, target_freq, target_aoa, freq_tol=5.0, aoa_tol=2.0):
    # Match pulse against cluster centroid within tolerance limits
    try:
        freq_ok = abs(pdw['centre_freq'] - target_freq) <= freq_tol
        aoa_ok = abs(pdw['aoa'] - target_aoa) <= aoa_tol
        return freq_ok and aoa_ok
    except KeyError:
        return False

def filter_mainbeam_and_convert(pdw_batch, target_freq, target_aoa, freq_tol=5.0, aoa_tol=2.0, min_amp_db=15.0):
    toas_sec = []
    amps_db = []
   
    for pdw in pdw_batch:
        # Ignore weak sidelobes
        if pdw.get('amplitude', 0.0) < min_amp_db:
            continue
           
        # Check frequency and bearing alignment
        if is_match(pdw, target_freq, target_aoa, freq_tol, aoa_tol):
            # Convert us to s for kinematics
            toas_sec.append(pdw['toa'] / 1e6)
            amps_db.append(pdw['amplitude'])
           
    return np.array(toas_sec, dtype=np.float64), np.array(amps_db, dtype=np.float64)