import csv
import ast
import numpy as np
from ppo_model import ActorCritic
from collections import deque
import torch
import argparse
import math

parser =argparse.ArgumentParser()
parser.add_argument('--outfile',type=str,default='sarsd_buffer_elijah.csv') #these rewards will be updated
parser.add_argument('--infile',type=str,default='raw_states_charlie.csv') #pull lidar scans from this file

parser.add_argument('--reward_only',type=bool,default=False)
args=parser.parse_args()

data_name = args.outfile.replace('raw_states_', '').replace('.csv', '')  # 'charlie'
scan_latent, scan_raw, speed, steer = None, None, None, None
raw_scan_prime = None
def compute_straightness(ranges: np.ndarray) -> float:
    beams_per_degree = 3
    forward_idx = 540
    window_size = 40 * beams_per_degree
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    clean = np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0)
    raw = np.mean(clean) / 40.0
    return 1 / (1 + np.exp(-raw))  # sigmoid
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
    
    return 3.0 * (1.0 - asymmetry)

def compute_straightness_plock(ranges: np.ndarray) -> float:
    beams_per_degree = 3
    forward_idx = 540
    window_size = 5 * beams_per_degree
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    clean = np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0)
    forward_mean= np.mean(clean)
    return float(np.clip(np.mean(clean) / 10.0, 0.0, 1.0)),forward_mean #10m forward is completely straight to the car

def find_gap_alignment_score(ranges: np.ndarray) -> int:
    gap_window_size = 30
    lo = 100
    hi = 980
    max_start_idx = hi - gap_window_size
    best_start_index = lo
    best_sum = -1.0  
    for x in range(lo, max_start_idx + 1):
        window = ranges[x : x + gap_window_size]
        clean_window = np.nan_to_num(window, nan=30.0, posinf=30.0)
        current_sum = np.sum(clean_window)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_start_index = x
            
    direction_index = best_start_index + (gap_window_size // 2)
    
    error=abs(540-int(direction_index)) #540 is the center scan index

    gap_alignment_score=1-(float(error)/340) #340 is the maximum worst error, if farthest point is at edge of vision

    return gap_alignment_score

def find_gap_direction(ranges: np.ndarray):
    gap_window_size = 30
    lo, hi = 100, 980
    max_start_idx = hi - gap_window_size
    best_start_index, best_sum = lo, -1.0
    for x in range(lo, max_start_idx + 1):
        window = np.nan_to_num(ranges[x:x + gap_window_size], nan=30.0, posinf=30.0)
        s = np.sum(window)
        if s > best_sum:
            best_sum, best_start_index = s, x
    direction_index = best_start_index + (gap_window_size // 2)
    signed_error = (direction_index - 540) / 425.0   # signed, /425 not /340
    return float(np.clip(signed_error, -1.0, 1.0))

    #ultimate_interp2, ultimate interp3,
def calculate_reward(ranges, speed, steering):
    ranges = np.nan_to_num(ranges, nan=30.0, posinf=30.0, neginf=0.0)
 
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
 
    threshold = max(3, speed)
    steer_threshold = max(2.5, speed)
 
    corner_gate = 1.0 - math.tanh(forward_mean / steer_threshold)   # ~1 near walls, ~0 on straights
    straight_gate = math.tanh(forward_mean / steer_threshold)        # complement
    speed_straight_bonus = 1.0 * speed * math.tanh(forward_mean / threshold)
    speed_corner_penalty = -1.0 * speed * corner_gate
 

    gap_dir = find_gap_direction(ranges)
    steer_norm = np.clip(steering / 0.4, -1.0, 1.0)
 
    steer_alignment = steer_norm * gap_dir
    steering_corner_bonus = 6.0 * steer_alignment * corner_gate
 
    steering_straight_penalty = -3.0 * abs(steer_norm) * straight_gate #interp_set1 has no steering straight penalty
 
    reward_safety = 4.0 * (min(min_dist / 0.2, 1.0) - 1.0)
 
    return (reward_safety
            + steering_corner_bonus
            + steering_straight_penalty
            + speed_corner_penalty
            + speed_straight_bonus)
 



#ultimate_interp2, ultimate interp3,
def calculate_reward_ultimate_interp2(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    steer_threshold = max(2.5,speed) 

    speed_straight_bonus =  0.1 * speed * math.tanh(forward_mean/threshold)
    speed_corner_penalty = -1* speed * (1-math.tanh(forward_mean/threshold))
    steering_corner_bonus = 6.0 * abs(steering) * (1-math.tanh(forward_mean/steer_threshold))
    steering_straight_penalty = -4.0 * abs(steering) * (math.tanh(forward_mean/steer_threshold))

    reward_safety =  8.0*(min(forward_mean/ 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

   # centerline_reward=centerline_reward_symmetry(ranges)
    gap_alignment_reward =  find_gap_alignment_score(ranges)
    return  reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus +gap_alignment_reward



#ultimate
def calculate_reward_ultimate(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    speed_straight_bonus = speed * math.tanh(forward_mean/threshold)
    speed_corner_penalty = -speed * (1-math.tanh(forward_mean/threshold))
    steering_corner_bonus = 3.0 * abs(steering) * (1-math.tanh(forward_mean/threshold))
    reward_safety =  (min(min_dist/ 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=centerline_reward_symmetry(ranges)
    gap_alignment_reward = 2.0*find_gap_alignment_score(ranges)
    return centerline_reward + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus +gap_alignment_reward



#veil
def calculate_reward_v(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -14.0 * speed * is_cornering
    speed_straight_bonus = 8.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    reward_safety =  20* (min(np.min(ranges) / 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=centerline_reward_symmetry(ranges)

    return centerline_reward + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus



#van, vega
def calculate_reward_v(ranges,speed,steering):
    min_dist = np.min(ranges)

    #not actually using straightness, just getting forward mean
    dummy, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(2,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -12.0 * speed * is_cornering
    speed_straight_bonus = 1.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    reward_safety =  2* (min(np.min(ranges) / 0.4, 1.0) - 1.0)  #coeff used to be 20 #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=3*centerline_reward_symmetry(ranges)

    return centerline_reward  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus +reward_safety


#ukulele
#less aggressive speed rewards than udon
def calculate_reward_ukulele(ranges,speed,steering):
    min_dist = np.min(ranges)

    #not actually using straightness, just getting forward mean
    dummy, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(2,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -12.0 * speed * is_cornering
    speed_straight_bonus = 20.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    reward_safety =  2* (min(np.min(ranges) / 0.4, 1.0) - 1.0)  #coeff used to be 20 #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=centerline_reward_symmetry(ranges)

    return centerline_reward  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus +reward_safety


#udon
def calculate_reward_udon(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(2,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -14.0 * speed * is_cornering
    speed_straight_bonus = 20.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    reward_safety =  20* (min(np.min(ranges) / 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=2*centerline_reward_symmetry(ranges)

    return centerline_reward + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus




#veggie, universe, umbra,  s
def calculate_reward_universe(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -14.0 * speed * is_cornering
    speed_straight_bonus = 8.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    reward_safety =  10* (min(np.min(ranges) / 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=3*centerline_reward_symmetry(ranges)

    return centerline_reward + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus


#summer,tofu,twig,veggie
def calculate_reward_stt(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -14.0 * speed * is_cornering
    speed_straight_bonus = 8.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    reward_safety =  20* (min(np.min(ranges) / 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=centerline_reward_symmetry(ranges)

    return centerline_reward + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus


#sterling
def calculate_reward_sterling(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -14.0 * speed * is_cornering
    speed_straight_bonus = 8.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    weaving_penalty = -2.0 * abs(steering) * (straightness **2.5)
    reward_safety =  20* (min(np.min(ranges) / 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    centerline_reward=centerline_reward_symmetry(ranges)

    return centerline_reward+weaving_penalty + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus

#stanley, sterling
def calculate_reward_stanley(ranges,speed,steering):
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

#plock
def calculate_reward_plock(ranges,speed,steering):
    min_dist = np.min(ranges)
    straightness, forward_mean = compute_straightness_plock(ranges)
    
    threshold = max(3,speed) 
    is_cornering = max(0, threshold - forward_mean) / threshold

    speed_corner_penalty  = -7.0 * speed * is_cornering
    speed_straight_bonus = 2.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    speed_confidence_bonus = 1.5* speed * min(forward_mean, 5.0) #max of 10 reward
    steering_corner_bonus = 8.0 * abs(steering) * is_cornering
    weaving_penalty = -2.0 * abs(steering) * (straightness **2.5)
    reward_safety =  20* (min(np.min(ranges) / 0.4, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width


    return weaving_penalty + reward_safety  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus+speed_confidence_bonus

#oak
#trying dynamic threshold
def calculate_reward_oak(ranges, speed, steering):
    min_dist = np.min(ranges)
    straightness = compute_straightness(ranges)
    
    beams_per_degree = 3
    forward_idx = 540
    window_size = 40 * beams_per_degree
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    forward_dist = np.mean(np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0))    
    
    threshold = max(1.5,speed*1.3)
    is_cornering = max(0, threshold - forward_dist) / threshold
    
    
    steering_corner_bonus = 6.0 * abs(steering) * is_cornering
    speed_corner_penalty  = -7.0 * speed * is_cornering
    
    speed_straight_bonus = 2.0 * (np.exp(speed / 2.0)) * (1-is_cornering)
    speed_confidence_bonus = 1.5* speed * min(forward_dist, 5.0) #max of 10 reward
    weaving_penalty = -2.0 * abs(steering) * (straightness **2.5)

    reward_safety = 9 * (min(np.min(ranges) / 0.3, 1.0) - 1.0)   #used to be 0.5 #NEED TO ADJUST THIS LINE depending on average track width

    return reward_safety +  weaving_penalty  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus +speed_confidence_bonus



#latte and muffin and nolan
def calculate_reward_latte_muffin_nolan(ranges, speed, steering):
    min_dist = np.min(ranges)
    straightness = compute_straightness(ranges)
    
    beams_per_degree = 3
    forward_idx = 540
    window_size = 40 * beams_per_degree
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    forward_dist = np.mean(np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0))    
    
    threshold = 3.0 #try 1
    is_cornering = max(0, threshold - forward_dist) / threshold
    
    
    steering_corner_bonus = 6.0 * abs(steering) * is_cornering
    speed_corner_penalty  = -5.0 * speed * is_cornering
    
    speed_straight_bonus = 2.0 * (np.exp(speed / 2.0)) * (1-is_cornering)

    weaving_penalty = -2.0 * abs(steering) * (straightness **2.5)
    reward_safety = 9 * (min(np.min(ranges) / 0.5, 1.0) - 1.0)   

    return reward_safety +  weaving_penalty  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus


#kale YES KALE GO 
#why does kale work in sim but not irl at all
def calculate_reward_kale(ranges, speed, steering):
    min_dist = np.min(ranges)
    straightness = compute_straightness(ranges)
    
    beams_per_degree = 3
    forward_idx = 540
    window_size = 50 * beams_per_degree
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    forward_dist = np.mean(np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0))    
    
    threshold = 3.0 #try 1
    is_cornering = max(0, threshold - forward_dist) / threshold
    
    
    steering_corner_bonus = 6.0 * abs(steering) * is_cornering
    speed_corner_penalty  = -5.0 * speed * is_cornering
    
    speed_straight_bonus = 1.0 * speed* (1-is_cornering)


    weaving_penalty = -2.0 * abs(steering) * max(0,1-is_cornering)**2
    reward_safety = 13 * (min(np.min(ranges) / 0.6, 1.0) - 1.0)   #16


    return reward_safety +  weaving_penalty  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus

#june
def calculate_reward_june(ranges, speed, steering):
    min_dist = np.min(ranges)
    straightness = compute_straightness(ranges)
    
    beams_per_degree = 3
    forward_idx = 540
    window_size = 40 * beams_per_degree
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    forward_dist = np.mean(np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0))    
    
    threshold = 3.0 #try 1
    is_cornering = max(0, threshold - forward_dist) / threshold
    
    
    steering_corner_bonus = 6.0 * abs(steering) * is_cornering
    speed_corner_penalty  = -5.0 * speed * is_cornering
    
    speed_straight_bonus = 2.0 * (np.exp(speed / 2.0)) * (1-is_cornering)

    weaving_penalty = -2.0 * abs(steering) * (straightness **2.5)
    reward_safety = 9 * (min(np.min(ranges) / 0.5, 1.0) - 1.0)   

    return reward_safety +  weaving_penalty  +steering_corner_bonus +speed_corner_penalty +speed_straight_bonus

#iris
def calculate_reward_iris(ranges, speed, steering):
    min_dist = np.min(ranges)
    straightness = compute_straightness(ranges)
    
    reward_safety = 8 * (min(np.min(ranges) / 0.5, 1.0) - 1.0)   
    if min_dist < 0.5:
        slow_near_corners = -(speed * (1.0 / max(min_dist, 0.5)))
    else:
        slow_near_corners = 0

    open_bonus = 2.5 * speed * (straightness ** 2)    
    speed_reward = 2.5 * speed * (1 + straightness)  #change back to 3

    weaving_penalty = -2.0 * abs(steering) * (straightness **1.5)
    return reward_safety + slow_near_corners  + open_bonus +weaving_penalty +speed_reward


#golf DO NOT CHANGE BECAUSE THIS WORKS WORKS
def calculate_reward_golf(ranges, speed, steering):
    min_dist = np.min(ranges)
    straightness = compute_straightness(ranges)
    
    reward_safety = 8 * (min(np.min(ranges) / 0.5, 1.0) - 1.0)
       
    if min_dist < 0.4:
        slow_near_corners = -(speed * (1.0 / max(min_dist, 0.4)))
    else:
        slow_near_corners = 0

    speed_reward = 2.0 * speed * (1 + straightness) 
    weaving_penalty = -2.0 * abs(steering) * (straightness **1.5)
    return reward_safety + slow_near_corners  + speed_reward +weaving_penalty 

#felix
def calculate_reward_felix(ranges, speed, steering):
    straightness = compute_straightness(ranges)
    
    reward_safety = -8 * np.exp(-np.min(ranges) / 0.5) #rewards distance away from wall
    #close to wall?

    slow_near_corners = 1/(speed* 1/min(np.min(ranges),0.5 )) #rewards being slow near a wall

    speed_reward = 0.7 * speed * (1 + straightness *3) #rewards going fast
    
    return reward_safety + slow_near_corners   + speed_reward
#ELIJAH DO NOT TOUCH
def calculate_reward_elijah(ranges, speed, steering):
    straightness = compute_straightness(ranges)
    
    reward_safety = -6 * np.exp(-np.min(ranges) / 0.5)
    speed_on_corner_penalty = -3 * speed * max(0.0, 0.65 - straightness)
    straightness_bonus = 3.7 * straightness * max(0.0, 1 - abs(steering) / 0.4)
    useless_turn_penalty = -3 * abs(steering) * max(0.0, straightness - 0.6)
    speed_reward = 0.1 * speed * (1 + straightness)
    good_turn_bonus = 5 * abs(steering) * max(0.0, 0.65 - straightness)
    
    return reward_safety + speed_on_corner_penalty + straightness_bonus + useless_turn_penalty + speed_reward

#charlie
def calculate_reward_charlie(ranges, speed, steering):
    straightness = compute_straightness(ranges)
    wall_penalty = -6.0 if min(ranges) < 0.4 else 0.0
    useless_turn_penalty = -1.1 * abs(steering) * straightness
    straightness_bonus = 3.0 * straightness * (1 - abs(steering) / 0.4)
    speed_reward = 0.02 * speed * (1 + straightness * 3)
    return useless_turn_penalty + straightness_bonus + speed_reward + wall_penalty
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = 'ppo_f1tenth_straightness_reward.pth'
model_path='encoder.pth'
model = ActorCritic(lidar_dim=1080).to(device)
model.load(model_path, device)
model.eval()
#infile has the compressed scans 
#outfile is the file whos rewards we want to edit
import os

with open(f'raw_data/{args.infile}', mode='r') as raw_file, \
     open(f'transitions/{args.outfile}', mode='r') as sarsd_file, \
     open('temp_output.csv', mode='w', newline='') as out_file:

    print(f'opening {args.infile}, recomputing rewards to {args.outfile}')

    raw_reader = csv.reader(raw_file)
    sarsd_reader = csv.DictReader(sarsd_file)
    writer = csv.writer(out_file)
    writer.writerow(['state', 'action', 'reward', 'state_prime', 'done'])

    straightness_vals = []
    min_range_vals=[]
    row_counter = 0
    for raw_row, sarsd_row in zip(raw_reader, sarsd_reader):
        scan = np.array(ast.literal_eval(raw_row[0]), dtype=np.float32)
        speed = float(raw_row[1])
        steer = float(raw_row[2])
        straightness_vals.append(compute_straightness(scan))
        min_range_vals.append(min(scan))
        reward = calculate_reward(scan, speed, steer)
        done = 1 if scan.min() < 0.15 else 0 
        writer.writerow([
            sarsd_row['state'],
            sarsd_row['action'],
            reward,
            sarsd_row['state_prime'],
            done
        ])
        row_counter += 1
        if row_counter % 2000 == 0:
            print(f'processed {row_counter} rows')

straightness_vals = np.array(straightness_vals)
min_range_vals = np.array(min_range_vals)
print(f"straightness: min={straightness_vals.min():.3f} max={straightness_vals.max():.3f} mean={straightness_vals.mean():.3f}")
print(f"min_range: min={min_range_vals.min():.3f} max={min_range_vals.max():.3f} mean={min_range_vals.mean():.3f}")
# replace original with updated version
os.replace('temp_output.csv', f'transitions/{args.outfile}')
print(f'done, updated {args.outfile}')
