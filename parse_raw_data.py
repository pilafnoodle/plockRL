import csv
import ast
import numpy as np
from ppo_model import ActorCritic
from collections import deque
import torch
import argparse

parser =argparse.ArgumentParser()
parser.add_argument('--infile',type=str,default='raw_states.csv')
parser.add_argument('--outfile',type=str,default='sarsd_buffer.csv')
parser.add_argument('--reward_only',type=bool,default=False)
args=parser.parse_args()


data_name = args.outfile.replace('raw_states_', '').replace('.csv', '')  # 'charlie'
scan_latent, scan_raw, speed, steer = None, None, None, None
raw_scan_prime = None

def compute_straightness(ranges: np.ndarray) -> float:
    beams_per_degree = 3
    forward_idx = 540
    window_size = 20 * beams_per_degree #was 30 for alfie
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    clean = np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0)
    return np.mean(clean) / 20.0 #was 30 for alfie

import numpy as np
def centerline_reward_symmetry(ranges):
    n = len(ranges)
    mid = n // 2
    
    right_side = ranges[:mid]
    left_side = ranges[mid:]
    
    left_min = np.min(left_side)
    right_min = np.min(right_side)
    
    # Normalized asymmetry: 0 = perfectly centered, 1 = hugging one wall
    asymmetry = abs(left_min - right_min) / (left_min + right_min + 1e-6)
    
    # Reward for being centered (positive when centered, decays with asymmetry)
    return 3.0 * (1.0 - asymmetry)


#sterling
def calculate_reward(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -10.0 * speed * is_cornering
    speed_straight_bonus = 2.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    speed_confidence_bonus = 1.5* speed * min(forward_mean, 5.0) #max of 10 reward
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    weaving_penalty = -2.0 * abs(steering) * (straightness **2.5)
    reward_safety =  20* (min(np.min(ranges) / 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=centerline_reward_symmetry(ranges)

    return centerline_reward+weaving_penalty + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus+speed_confidence_bonus

def compute_straightness_plock(ranges: np.ndarray) -> float:
    beams_per_degree = 3
    forward_idx = 540
    window_size = 5 * beams_per_degree
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    clean = np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0)
    forward_mean= np.mean(clean)
    return float(np.clip(np.mean(clean) / 10.0, 0.0, 1.0)),forward_mean #10m forward is completely straight to the car

#dale
def calculate_reward(ranges, speed, steering):
    straightness = compute_straightness(ranges)  
    #speed_on_corner_penalty = -10 * speed * (1.0 - straightness)
    reward_safety = -12 *np.exp(-np.min(ranges)/0.5) #pls dont crash
    useless_turn_penalty = -4 * abs(steering) * straightness
    straightness_bonus = 2.0 * straightness * (1 - abs(steering) / 0.4)
    good_turn_bonus = 0.5* abs(steering) * 1/(max(straightness,0.1))
    speed_reward = 0.04 * speed * (1 + straightness * 3)
    return useless_turn_penalty + straightness_bonus + speed_reward + reward_safety + good_turn_bonus 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = 'ppo_f1tenth_straightness_reward.pth'
model_path='encoder.pth'
model = ActorCritic(lidar_dim=1080).to(device)
model.load(model_path, device)
model.eval()

def compress_lidar(ranges):
    ranges_raw = np.array(ranges, dtype=np.float32)
    ranges_raw = np.where(np.isfinite(ranges_raw), ranges_raw, 30.0)
    with torch.no_grad():
        ranges_t = torch.FloatTensor(ranges_raw).unsqueeze(0).to(device)
        latent_t = model.encode(ranges_t)
        return latent_t.squeeze(0).cpu().numpy()

with open(f'raw_data/{args.infile}', mode='r') as infile, \
     open(f'transitions/{data_name}.csv', mode='w', newline='') as outfile:
    
    print(f'opening {args.infile}, parsing to {data_name}')
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    writer.writerow(['state', 'action', 'reward', 'state_prime', 'done'])
    row_counter=0


    for row in reader:
        try:
            # 1. Basic check: Is the row empty or too short to be real?
            if not row or len(row) < 3:
                scan_latent, scan_raw, speed, steer = None, None, None, None
                continue

            # 2. Syntax check: Does the LIDAR string at least look complete?
            lidar_str = row[0].strip()
            if not lidar_str.startswith('[') or not lidar_str.endswith(']'):
                print(f"Skipping truncated row {row_counter}")
                scan_latent, scan_raw, speed, steer = None, None, None, None
                continue

            scan_prime = ast.literal_eval(lidar_str)
            
            # 3. Shape check (for the 935 vs 1080 issue)
            if len(scan_prime) != 1080:
                print(f"Skipping malformed row {row_counter}: expected 1080, got {len(scan_prime)}")
                scan_latent, scan_raw, speed, steer = None, None, None, None
                continue

            # --- Data is valid, proceed with compression and reward ---
            raw_scan_prime = np.array(scan_prime, dtype=np.float32)
            latent_scan_prime = compress_lidar(scan_prime)
            speed_prime = float(row[1])
            steer_prime = float(row[2])

            if scan_latent is not None:
                reward = calculate_reward(raw_scan_prime, speed_prime, steer_prime)
                done = 1 if raw_scan_prime.min() < 0.15 else 0
                writer.writerow([
                    scan_latent.tolist(),
                    [steer, speed],
                    reward,
                    latent_scan_prime.tolist(),
                    done
                ])
                if done == 1:
                    scan_latent, scan_raw, speed, steer = None, None, None, None
                    row_counter += 1 # Still increment counter
                    continue

            # Shift window for next SARSD transition
            scan_latent = latent_scan_prime
            scan_raw = raw_scan_prime
            speed = speed_prime
            steer = steer_prime

        except (SyntaxError, ValueError) as e:
            print(f"Parsing error at row {row_counter}: {e}")
            scan_latent, scan_raw, speed, steer = None, None, None, None
            continue
        except Exception as e:
            print(f"Unexpected error at row {row_counter}: {e}")
            scan_latent, scan_raw, speed, steer = None, None, None, None
            continue

        if row_counter % 500 == 0:
            print(f'sarsded {row_counter} rows')
        row_counter += 1

#ai
    # for row in reader:
    #     try:

    #         scan_prime = ast.literal_eval(row[0])
    #         if len(scan_prime) != 1080:
    #                 print(f"Skipping malformed row {row_counter}: expected 1080, got {len(scan_prime)}")
    #                 scan_latent, scan_raw, speed, steer = None, None, None, None
    #                 continue
    #         raw_scan_prime = np.array(scan_prime, dtype=np.float32)

    #         latent_scan_prime = compress_lidar(scan_prime)
    #         speed_prime = float(row[1])
    #         steer_prime = float(row[2])

    #         if scan_latent is not None:
    #             reward = calculate_reward(raw_scan_prime, speed_prime, steer_prime)
    #             done = 1 if raw_scan_prime.min() < 0.15 else 0
    #             writer.writerow([
    #                 scan_latent.tolist(),
    #                 [steer, speed],
    #                 reward,
    #                 latent_scan_prime.tolist(),
    #                 done
    #             ])
    #             if done == 1:
    #                 scan_latent, scan_raw, speed, steer = None, None, None, None
    #                 continue
    #         if row_counter%500==0:


    #             print(f'sarsded {row_counter} rows')
    #         row_counter=row_counter+1

    #         scan_latent = latent_scan_prime
    #         scan_raw = raw_scan_prime
    #         speed = speed_prime
    #         steer = steer_prime
    #     except Exception as e:
    #         print(f"Error at row {row_counter}: {e}")
    #         # Reset state on error to prevent bad transitions
    #         scan_latent, scan_raw, speed, steer = None, None, None, None
    #         continue
