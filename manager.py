import math
import numpy as np

class TrackFile:
    def __init__(self, target_id, kf, t_scan, tau_illum_max):
        self.target_id = target_id
        self.kf = kf
        self.t_scan = t_scan
        self.tau_illum_max = tau_illum_max
        self.consecutive_misses = 0

class TrackManager:
    def __init__(self):
        self.tracks = {}

    def add_track(self, target_id, kf, t_scan, tau_illum_max):
        self.tracks[target_id] = TrackFile(target_id, kf, t_scan, tau_illum_max)
        print(f"[MANAGER] New track spawned: {target_id}")

    def record_hit(self, target_id, measured_theta, measured_tau):
        if target_id in self.tracks:
            track = self.tracks[target_id]
            track.consecutive_misses = 0
            # Running maximum of burst duration for partial-intercept fix
            track.tau_illum_max = max(track.tau_illum_max, measured_tau)
            track.kf.update(measured_theta)

    def record_miss(self, target_id, dt):
        if target_id in self.tracks:
            track = self.tracks[target_id]
            track.consecutive_misses += 1
            track.kf.predict(dt)

    def prune_ghosts(self):
        dead_targets = []
        for target_id, track in self.tracks.items():
            # Hard limit: 4 misses
            if track.consecutive_misses >= 4:
                dead_targets.append(target_id)
                continue
            
            # Mathematical cap: window exceeds 25% of T_scan
            sigma_theta = math.sqrt(track.kf.P[0, 0])
            omega = track.kf.x[1, 0]
            sigma_t = sigma_theta / omega
            total_duration = track.tau_illum_max + 6.0 * sigma_t
            
            if total_duration > (0.25 * track.t_scan):
                dead_targets.append(target_id)
                
        for target_id in dead_targets:
            del self.tracks[target_id]
            print(f"[MANAGER] Pruned ghost track (Covariance Runaway prevented): {target_id}")

    def get_next_interrupts(self, current_time):
        requests = []
        for target_id, track in self.tracks.items():
            theta = track.kf.x[0, 0]
            omega = track.kf.x[1, 0]
            
            # Predict time until phase reaches 2*pi (pointing directly at us)
            dt_to_center = (2.0 * math.pi - theta) / omega
            predicted_center = current_time + dt_to_center
            
            # Dynamic window sizing
            sigma_theta = math.sqrt(track.kf.P[0, 0])
            sigma_t = sigma_theta / omega
            duration = track.tau_illum_max + 6.0 * sigma_t
            start_time = predicted_center - (duration / 2.0)
            
            requests.append({
                "target_id": target_id,
                "start_time": start_time,
                "duration": duration,
                "uncertainty_p00": track.kf.P[0, 0]
            })
            
        # Collision Tie-Breaker: Sort so the highest uncertainty gets priority
        requests.sort(key=lambda x: x["uncertainty_p00"], reverse=True)
        return requests