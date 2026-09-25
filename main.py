import time
import threading
import queue

# When Sujal links the repo, he will import the manager you built
# from manager import TrackManager

def run_daemon_loop(hit_queue, interrupt_queue, track_manager):
    # This loop runs on its own thread, completely decoupled from Domains 2, 3, and 4
    while True:
        try:
            # 1. Pull all pending pulses from Tanisha's receiver (non-blocking)
            raw_hits = []
            while not hit_queue.empty():
                raw_hits.append(hit_queue.get_nowait())
           
            # 2. Feed hits to Track Manager (Runs Sieve, Lomb-Scargle, and Kalman Update)
            if raw_hits:
                track_manager.ingest_hits(raw_hits)
           
            # 3. Advance the physics engine and run Ghost Pruning
            # This expands Covariance (P) and deletes tracks with 4 misses or >25% T_scan widths
            current_time = time.time()
            track_manager.coast_and_prune(current_time)
           
            # 4. Resolve collisions based on Covariance and grab the critical preempt window
            interrupt_token = track_manager.get_winning_interrupt(current_time)
           
            # 5. Seize the receiver if a target is approaching
            if interrupt_token:
                interrupt_queue.put(interrupt_token)
               
        except queue.Empty:
            pass
        except Exception as e:
            # Prevent Domain 5 math errors from crashing the entire hackathon simulation
            print(f"[DOMAIN 5 CRASH PREVENTED] {e}")
           
        # Yield CPU so we don't starve Aditya's data stream or Tusar's RL models
        time.sleep(0.005)