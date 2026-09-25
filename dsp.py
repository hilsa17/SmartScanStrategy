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

if __name__ == "__main__":
    # Smoke test for local verification
    mock_batch = [
        {"toa": 12500000.0, "centre_freq": 4501.0, "pw": 2.5, "aoa": 134.9, "amplitude": 22.0},
        {"toa": 12500010.0, "centre_freq": 4500.0, "pw": 2.5, "aoa": 135.0, "amplitude": 10.0},
        {"toa": 12500020.0, "centre_freq": 9000.0, "pw": 1.0, "aoa": 40.0, "amplitude": 25.0}
    ]
    t, a = filter_mainbeam_and_convert(mock_batch, 4500.0, 135.0)
    print("Filtered TOAs (s):", t)
    print("Filtered Amplitudes (dB):", a)