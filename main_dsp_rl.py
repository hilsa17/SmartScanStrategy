"""
main_dsp_rl.py  --  DSP + Reinforcement Learning
================================================
Run (from the folder that holds the config_*.h5 files):
    python main_dsp_rl.py                      # add --timesteps 20000 --predictor-steps 500 for a quick test

What happens:
  1. DSP studies the first 10 s of every TEST file (dsp_link.py: sieve -> estimator -> Kalman -> manager).
  2. If DSP confirms a rotating radar, it writes a timetable: "in these slots its beam hits us, listen to band X".
  3. The RL agent is trained as usual on the TRAIN files (DSP plays no part in training).
  4. On the TEST episodes the trained agent runs twice:
        DQN       -> the agent chooses every slot
        DQN+DSP   -> the DSP timetable comes first (DSP dominates); the agent chooses only the other slots
  5. If DSP confirms nothing, the timetable is empty and DQN+DSP equals DQN.

Results: results/rl_results.csv, results/final_result.json (includes what DSP concluded), results/rl_training_curve.png
"""
import sys, glob
sys.path += glob.glob("*/")          # so the other files may sit in sub-folders next to this one

import rl_all
from dsp_link import dsp_sources

if __name__ == "__main__":
    a = rl_all.make_parser().parse_args()
    rl_all.run_rl_pipeline(a, dsp=dsp_sources(a))

