# plockRL: Successes, Failures and Thoughts

## Overview
PlockRL is a reinforcement learning pipeline based on the TD3 algorithm for F1Tenth. It tunes an imperfect driving model by training on lidar scan, steering and speed data collected while driving. TD3 is an offline algorithm that calculates the reward of the current state by looking at the actions taken and reward of future states. It requires a transition tuple in the state, action, state_prime, reward, done state format

In the context of PlockRL:
```
State: the current lidar scan
Action: speed, steering
State_prime: the next lidar scan captured
Reward: value of the current action at the current state
Done: 1 if in crash state
```

## Pipeline Architecture

```
                  [ Car Running Inference ]
                     (Naive or Model Base)
                               │
                               ▼
         ┌──────────────────────────────────────────┐
         │          sarsd_data_collect.py           │
         ├──────────────────────────────────────────┤
         │ • Collects csv logs per frame online     │──► raw_states_current.csv
         └─────────────────────┬────────────────────┘
                               │
                               ▼
         ┌──────────────────────────────────────────┐
         │              mirror_data.py              │
         ├──────────────────────────────────────────┤
         │ • Augments data by doubling              │──► raw_states_mirror_current.csv
         │ • Reverses scan list, inverts steering   │
         └─────────────────────┬────────────────────┘
                               │
                               ▼
         ┌──────────────────────────────────────────┐
         │            parse_raw_data.py             │
         ├──────────────────────────────────────────┤
         │ • Formats dataset into TD3 tuples        │ ──► sarsd_buffer_current.csv
         │ • Calculates dummy rewards               │
         │ • Compresses raw 1080-D scans to 128-D   │
         └─────────────────────┬────────────────────┘
                               │
                               ▼
         ┌──────────────────────────────────────────┐
         │           recompute_rewards.py           │
         ├──────────────────────────────────────────┤
         │ • Calculate rewards                      │──► sarsd_buffer_current.csv 
         │ • Overwrites dummy rewards in place      │
         └─────────────────────┬────────────────────┘
                               │
                               ▼
         ┌──────────────────────────────────────────┐
         │               train_td3.py               │
         ├──────────────────────────────────────────┤
         │ • Offline training loop                  │──► td3_current.pth 
         │ • Actor updates delayed by 2 epochs      │
         └──────────────────────────────────────────┘
```
## File structure
```
plockRL/
├── models/
├── raw_data/
└── transitions/
```

## Use

Collect a csv using the sarsd_data_collect.py script while the car is running.

Rename the file to "raw_states_[name]"

Run python3 full_processing.py --name [name]

A model will be output in /models

## Development details
Before data could be collected, we required a driver that could drive the car mostly around a track to fine tune with TD3. For this, we used a farthest point follower on the given f1tenth gym simulator. The first iteration of data collected and trained was simulator only. then, 

The bulk of the work is developing calculate_reward() in recompute_rewards.py. The development process consists of choosing which metrics to penalize and which to optimize and making sure that none conflict or overpower each other. This was tested by writing down possible ranges for every reward and comparing and adding coefficients. This was not efficient and needs to be a better method.

Sometimes the model did not steer enough and external multipliers in the inference script would be used to tune steering and speed with multipliers that scale based on average of forward distance. This worked to some extent, but we were unable to bake the behavior into the model so that it would run without multipliers. However, using multipliers, we were able to collect more perfect data to train on. 

Every reward function is listed in recompute_rewards.py as well as more details in notes.txt

Inference was optimized to around 2ms per input, which is faster than lidar scan speed, so inference speed is not a bottleneck.

## Improvements needed
### Add preprocessing
The type of tubes used for the track can vary. Add 1D gaussian smoothing to eliminate track material specific variation in the lidar scan. In inference, preprocess scan with same smoothing filter. 

### Develop better simulator 
The provided F1tenth gym in RViz did not translate well at all to the real car. Much time was spent testing on a real car which is time consuming and tiring. 

### Create more specific rewards
Rewards were very simple, for example adding a penalty when any lidar scan was less than a threshold. More specific rewards such as a heading reward based on the direction of the farthest point were introduced, but there was not enough time to develop it.

## FUTURE F1TENTH PARTICIPANTS PLEASE READ
### attempt heuristic method first
There are many classical methods for track navigation, such as gap follow, farthest point follower and wall follow. Attempt these first, they are far easier than reinforcement or imitation learning. However these may struggle if competition track intentionally has gaps or holes. 

### streamline pipeline asap
It will save alot of time.

### use rosbags
I forgot these existed, so all data was collected with csvs. Rosbags have synchronization








