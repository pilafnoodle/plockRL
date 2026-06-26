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

The previously trained model would be used to jumpstart the TD3 actor in the next training with a different reward function so it could learn quicker. 

We also used a heuristic-based post-processing step. To speed up the car and encourage turning, we added a interpolated multiplier based on the average forward distance. 

` ` 
However we had trouble tuning the rewards so that the heuristic would be baked into the model

Every reward function is listed in recompute_rewards.py as well as more details in notes.txt

Inference was optimized to around 2ms per input, which is faster than lidar scan speed, so inference speed is not a bottleneck.

## Improvements needed
### Add preprocessing
The type of tubes used for the track can vary. Add 1D gaussian smoothing to eliminate track material specific variation in the lidar scan. In inference, preprocess scan with same smoothing filter. 

### Automate reward tuning
Our rewards were tuned by hand per iteration which was very time consuming. Test many rewards at once and find a way to evaluate models on **software**.

### Develop better simulator 
The provided F1tenth gym in RViz did not translate well at all to the real car. Much time was spent testing on a real car which is time consuming and tiring. 

### Create more specific rewards
Rewards were very simple, for example adding a penalty when any lidar scan was less than a threshold. More specific rewards such as a heading reward based on the direction of the farthest point were introduced, but there was not enough time to develop it.

## FUTURE CISL F1TENTH PARTICIPANTS PLEASE READ
### Attempt heuristic method first
There are many classical methods for track navigation, such as gap follow, farthest point follower and wall follow. Attempt these first, they are far easier than reinforcement or imitation learning. However these may struggle if competition track intentionally has gaps or holes. 

### Streamline pipeline asap
It will save alot of time.

### Use rosbags
I forgot these existed, so all data was collected with csvs. Rosbags have synchronization.

## Isolate network
Originally we had used Tailscale to SSH into the car, but some spots in Winston Chung Hall were spotty and caused the car to disconnect. The solution was using our own router to connect to the car. **Make sure the router's subnet and the Lidar's subnet are different, otherwise they will fight for connection!**

## Roboracer @ IV 2026
We didn't qualify because our car was not able to navigate certain section of the track. The track consisted of bumpy orange tubes and smooth black tubes, and our model was only trained on bumpy orange tubes. We believe the failure to navigate black tubes was because of lack of preprocessing and our model had overfit to the bumpy tubes. The track specifications given to us by Roboracer (overall geometry, min and max width) were also very different from the given track. Roboracer also did not let us know about different types of tubes in the same track. We worked in parallel to test heuristic methods and retrain the model on race day, but it did not work. Our TD3 process was very iterative and required multiple rounds of data collection and training, which we did not have time for. However the process of building the car from scratch, wiring up connections, learning RL trial and error was enough of a learning experience in it of itself.

Gif: Car running on Winston Chung Patio


I feel like the protagonist of a space movie writing this down all their notes about survival and science and placing it in a file to send back to Earth for future astronauts. Good luck future teams ;-)

Team Plock (Riverside Racers) signing off,
Amber and Alex





