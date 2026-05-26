import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
from sensor_msgs.msg import LaserScan       
import csv          
import time
import message_filters
from actor_critic.ppo_model import ActorCritic
import torch
import os
#s a r s d'
#s: current state/scan (incoming lidar scan)
#a: action taken (steering, speed)
#r: reward
#s': future state/scan?
#done flag its just a terminal

def compute_straightness(ranges: np.ndarray) -> float:
    beams_per_degree = 3        # 1080 / 360 = 3
    forward_idx      = 540
    window_size      = 30 * beams_per_degree  # ±15 degrees instead of ±5
    lo = forward_idx - window_size // 2
    hi = forward_idx + window_size // 2
    clean = np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0)
    return np.mean(clean) / 30.0

class sarsd_data_collect(Node):

    def __init__(self):
        super().__init__("sarsd_collector")
            

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = 'ppo_f1tenth_straightness_reward.pth'
        
        self.model = ActorCritic(lidar_dim=1080).to(self.device)
        self.model.load(model_path, self.device)
        self.model.eval()
        self.get_logger().info(f"Loaded encoder from {model_path}")
        self.experience_buffer_csv = "sarsd_buffer_brooke.csv"

        if not os.path.exists(self.experience_buffer_csv):
            with open(self.experience_buffer_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['state', 'action', 'reward', 'state_prime', 'done'])

       

        self.scan_sub = message_filters.Subscriber(self, LaserScan, '/scan')
        self.drive_sub = message_filters.Subscriber(self, AckermannDriveStamped, '/drive')
        self.ts = message_filters.ApproximateTimeSynchronizer(
        [self.scan_sub, self.drive_sub], 10, 0.1)
        self.ts.registerCallback(self.log_data_callback)

        self.steering_previous =None
        self.steering_current = None
        self.state_prime = None
        self.state_current = None


    def calculate_reward(self, ranges, speed, steering):
        wall_penalty = -6.0 if min(ranges) < 0.4 else 0.0 #used to be -3.0 and 0.875
        straightness = compute_straightness(ranges)
        
        useless_turn_penalty = -1.1 * abs(steering) * straightness #used to be -2 *
        straightness_bonus = 3.0 * straightness * (1 - abs(steering) / 0.4)  # high reward when straight path + low steering
        speed_reward = 0.02 * speed * (1 + straightness * 3) #used to be coeff 0.05
        
        return useless_turn_penalty + straightness_bonus + speed_reward + wall_penalty

    
    def log_data_callback(self, scan_msg,drive_msg):

        ranges_raw = np.array(scan_msg.ranges, dtype=np.float32)
        ranges_raw = np.where(np.isfinite(ranges_raw), ranges_raw, 30.0)

        # 2. Use the Model to Encode (Matches self.model.predict logic)
        with torch.no_grad():
            ranges_t = torch.FloatTensor(ranges_raw).unsqueeze(0).to(self.device)
            # .encode() handles clamping and division by 30.0 internally
            latent_t = self.model.encode(ranges_t) 
            self.state_prime = latent_t.squeeze(0).cpu().numpy()

        steering_angle= drive_msg.drive.steering_angle 
        speed = drive_msg.drive.speed
        action=[steering_angle,speed] #make a tuple here 
        done=0

        if self.state_current is not None: #the following two lines are ai
            reward = self.calculate_reward(ranges_raw, speed, steering_angle)

            done = 1 if ranges_raw.min() < 0.1 else 0
            with open(self.experience_buffer_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.state_current.tolist(), 
                    action, 
                    reward, 
                    self.state_prime.tolist(), 
                    done
                ])        
        if done == 1:
            self.state_current = None
        else:
            self.state_current = self.state_prime
        
def main(args=None):
    rclpy.init(args=args)
    node = sarsd_data_collect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally: 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
