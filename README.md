Smart Scan system for electronic warfare:
 
This project has been integrated with Reinforcement Learning, DSP (circular kalman filter, lomb scargle periodogram) to actively predict which frequency band the
transmitted enemy signal might be possibly transmitted in.
Bandit Algorithm is also used for comparison purpose to RL, while the DSP remains an integral part of both. 
RL is superior to Bandit, and definitely the normal standard method.

BANDIT - UCB, D-UCB, WHITTLE, SW -UCB, Exp 3, MYOPIC indices try to predict the same thing as RL but is less efficient
RL MODEL - integrates DQN, LSTM predictor to select which band to look at
DSP - includes Circular Kalman filter and Lomb Scargle periodogram to deterministically select the band to look at, via statistical analysis to calculate the radar time period

There is provision for expansion and further testing, like Multi Radar enemy usage, frequency hopping and Multi armed bandit problem solving, ghost pruner.

DSP has 4 files, estimator, kalman, manager and sieve, all are .py type, and the main.py, outputting DWELL TIME, DURATION & FREQUENCY BAND To look at
Similar thing is done by RL and bandit models
Absolute priority remains with the DSP, as it is the deterministic part of the entire solution. 

rl_all.py is the RL code
bandits_all.py is the bandits algo code

bandits is for comparison, depending on what you want to test, you can run bandits + DSP, or RL + DSP

testing has been done on the dataset provided by SIH 2026, PS 26055
link - JC Wise, Radar emitter Database, 2024 huggingface.co/datasets/alan-turing institute/turing-synthetic radar dataset
