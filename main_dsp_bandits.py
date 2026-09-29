"""
main_dsp_bandits.py  --  DSP + Bandits
=======================================
Run (from the folder that holds the config_*.h5 files):
    python main_dsp_bandits.py

What happens:
  1. DSP studies the first 10 s of every TEST file (dsp_link.py: sieve -> estimator -> Kalman -> manager).
  2. If DSP confirms a rotating radar, it writes a timetable: "in these slots its beam hits us, listen to band X".
  3. Every bandit algorithm runs twice on the SAME episodes:
        name       -> the bandit chooses every slot
        name+DSP   -> the DSP timetable comes first (DSP dominates); the bandit chooses only the other slots
  4. If DSP confirms nothing, the timetable is empty and name+DSP equals name.

Results: results/bandits_results.csv, results/dsp_bandits_result.json, results/bandits_comparison.png
"""
import sys, glob
sys.path += glob.glob("*/")          # so the other files may sit in sub-folders next to this one

import bandits_all
from dsp_link import dsp_sources

if __name__ == "__main__":
    a = bandits_all.make_parser().parse_args()
    bandits_all.main(a, dsp=dsp_sources(a))
