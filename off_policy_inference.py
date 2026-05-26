#!/usr/bin/env python3
"""
td3_actor_node.py  —  Inference / Deployment Node

Uses the trained TD3 actor + PPO encoder to drive the F1Tenth car autonomously.
The PPO encoder compresses raw 1080-D LiDAR to a 128-D latent vector,
which is then passed to the TD3 actor to produce [steering, speed].


"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

import numpy as np
import torch
import os
import csv
import time
import math

import argparse

parser =argparse.ArgumentParser()
parser.add_argument('--infile',type=str,default='td3_f1tenth_alfie.pth')
parser.parse_known_args()[0]  # instead of parse_args()
args = parser.parse_known_args()[0]


from ppo_model import ActorCritic
#from td3_train import TD3_Actor
from td3_train import TD3_Actor
LIDAR_DIM  = 1080
LATENT_DIM = 128 #history has 134, lidar only has 128


class TD3ActorNode(Node):

    def __init__(self):
        super().__init__('td3_actor_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('ppo_model_path', 'ppo_f1tenth_straightness_reward.pth')
        self.declare_parameter('td3_model_path', f'models/{args.infile}')     
        self.declare_parameter('max_steering',   0.4)
        self.declare_parameter('max_speed',      3.5)
        self.declare_parameter('min_speed',      0.5)

        ppo_path  = self.get_parameter('ppo_model_path').value
        td3_path  = self.get_parameter('td3_model_path').value
        self.max_steer = self.get_parameter('max_steering').value
        self.max_spd   = self.get_parameter('max_speed').value
        self.min_spd   = self.get_parameter('min_speed').value

        # ── Device ────────────────────────────────────────────────────────────
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ── Load PPO encoder ──────────────────────────────────────────────────
        if not os.path.exists(ppo_path):
            self.get_logger().error(f"PPO model not found: {ppo_path}")
            raise FileNotFoundError(ppo_path)

        self.ppo = ActorCritic(lidar_dim=LIDAR_DIM, latent_dim=LATENT_DIM).to(self.device)
        self.ppo.load(ppo_path, self.device)
        self.ppo.eval()
        self.get_logger().info(f"PPO encoder loaded <- {ppo_path}")

        # ── Load TD3 actor ────────────────────────────────────────────────────
        if not os.path.exists(td3_path):
            self.get_logger().error(f"TD3 model not found: {td3_path}")
            raise FileNotFoundError(td3_path)

        self.actor = TD3_Actor(latent_dim=LATENT_DIM, action_dim=2).to(self.device)
        self.actor.load_state_dict(torch.load(td3_path, map_location=self.device))
        self.actor.eval()
        self.get_logger().info(f"TD3 actor loaded <- {td3_path}")

        # ── ROS pub/sub ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.scan_sub  = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.odom_sub  = self.create_subscription(
            Odometry, '/ego_racecar/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        # ── Logging ───────────────────────────────────────────────────────────
        self.step_count = 0
        self.start_time = time.time()
        self.total_dist = 0.0
        self.prev_x     = None
        self.prev_y     = None

        # self.log_file   = open('td3_run_log.csv', 'w', newline='')
        # self.csv_writer = csv.writer(self.log_file)
        # self.csv_writer.writerow(['time', 'x', 'y', 'steering', 'speed'])

        self.get_logger().info(
            f"TD3 Actor Node ready | device={self.device} | "
            f"ppo={ppo_path} | td3={td3_path}"
        )

    # ── Odometry callback ─────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_x is not None:
            step = math.sqrt((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2)
            if step < 1.0:
                self.total_dist += step

        self.prev_x, self.prev_y = x, y

    # ── LiDAR callback ────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        # Pre-process scan
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)

        # Inference
        with torch.no_grad():
            lidar_t  = torch.FloatTensor(ranges).unsqueeze(0).to(self.device)
            latent   = self.ppo.encode(lidar_t)        # (1, 1080) -> (1, 128)
            action   = self.actor(latent)              # (1, 128)  -> (1, 2)
            steering = float(action[0, 0].cpu())
            speed    = float(action[0, 1].cpu())

        # Safety clamp
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        speed    = float(np.clip(speed,     self.min_spd,   self.max_spd))

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp    = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering 
        drive_msg.drive.speed          = speed 
        self.drive_pub.publish(drive_msg)

        # # Log to CSV
        # self.csv_writer.writerow([
        #     round(time.time() - self.start_time, 3),
        #     round(self.prev_x or 0.0, 3),
        #     round(self.prev_y or 0.0, 3),
        #     round(steering, 4),
        #     round(speed, 4),
        # ])

        # Console logging every 200 steps
        self.step_count += 1
        if self.step_count % 200 == 0:
            self.get_logger().info(
                f"Step {self.step_count} | "
                f"steer={np.degrees(steering):.1f}° | "
                f"speed={speed:.2f} m/s | "
                f"dist={self.total_dist:.2f} m"
            )

    def destroy_node(self):
        # self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TD3ActorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
