Smart Scan Strategy for Electronic Warfare

A software-only scheduler that decides which frequency band an ESM receiver should listen to next. 
The receiver covers 2–18 GHz in 32 bands but can hear only one at a time, so a fixed sweep wastes time on empty bands and misses hopping or briefly illuminating radars.
Our goal is a lower time to first intercept (TTFI) and a higher probability of intercept (P_int).Approach
RL scanner (default): a DQN with an LSTM busy-predictor (PyTorch, Stable-Baselines3). It is rewarded for hits and penalised for misses and retunes, so it learns hop patterns.
DSP tracker: a TOA histogram, Lomb-Scargle periodogram and circular Kalman filter find a rotating radar's period and predict when its beam points at us. 
It takes over only when confident, for a short ±3σ window.
Bandits (D-UCB, SW-UCB, Exp3, Whittle) are the baseline and fail-safe.

If the DSP confirms nothing, the system behaves exactly like the RL scanner alone.Data

radar_data.py turns the Turing Synthetic Radar Dataset (config_*.h5 stare files) into a busy map of time slots × 32 bands. The simulated receiver is imperfect (95% detection, 2% false alarm) and every algorithm is tested on the same episodes.

Files
radar_data.py – data loading, receiver model, metrics
rl_all.py – DQN + LSTM training and evaluation
bandits_all.py – bandit baselines
dsp_link.py – DSP pipeline and timetable
main_dsp_rl.py – run DSP + RL
main_dsp_bandits.py – run DSP + bandits
dqn_spectrum.zip – trained DQN checkpoint

Run
pip install numpy pandas h5py torch gymnasium stable-baselines3 matplotlib
# put the config_*.h5 files in this folder, or pass --data-root
python main_dsp_rl.py          # add --timesteps 20000 --predictor-steps 500 for a quick test
python main_dsp_bandits.py

Results (P_int, TTFI, plots) are written to results/.

Deployment
The model is small enough to run on one plug-in NVIDIA Jetson AGX Orin card in an existing ESM chassis, with only a firmware update to the existing FPGAs.
