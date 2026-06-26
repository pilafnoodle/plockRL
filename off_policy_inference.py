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
from gaussian_filter1d_numpy import gaussian_filter1d_numpy as gaussian_filter1d
import numpy as np
import torch
import os
import time
import math
import random
import argparse
import torch.nn as nn

from ppo_model import ActorCritic
from td3_train import TD3_Actor

LIDAR_DIM = 1080
LATENT_DIM = 128

parser=argparse.ArgumentParser()
parser.add_argument('--model',type=str,default="td3_ultimate_interp_set1.pth")
args=parser.parse_args()

class FusedPolicy(nn.Module):
    def __init__(self, encoder, td3_actor):
        super().__init__()
        self.encoder=encoder
        self.td3_actor=td3_actor

    def forward(self, lidar_scan):
        latent_scan=self.encoder.encode(lidar_scan)
        return self.td3_actor(latent_scan)
    



class TD3ActorNode(Node):

    def __init__(self):
        super().__init__('td3_actor_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('ppo_model_path', 'models/ppo_f1tenth_weave.pth')
        model_path = args.model
        self.declare_parameter('td3_model_path', f'models/{args.model}')
        self.declare_parameter('max_steering', 0.6)
        self.declare_parameter('max_speed', 3.5)
        self.declare_parameter('min_speed', 0.5)
        self.declare_parameter('speed_scale', 1.0)  # safety scaling factor
        self.needs_recov = False

        ppo_path = self.get_parameter('ppo_model_path').value
        td3_path = self.get_parameter('td3_model_path').value
        self.max_steer = self.get_parameter('max_steering').value
        self.max_spd = self.get_parameter('max_speed').value
        self.min_spd = self.get_parameter('min_speed').value
        self.speed_scale = self.get_parameter('speed_scale').value

        self.previous_steering=0

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

        self.fusedPolicy = FusedPolicy(self.ppo, self.actor).to(self.device).eval()
        example = torch.zeros(1, LIDAR_DIM, device=self.device)   # dummy input, same shape as real scan
        with torch.inference_mode():
            self.fusedPolicy = torch.jit.trace(self.fusedPolicy, example, check_trace=False)
            self.fusedPolicy = torch.jit.optimize_for_inference(self.fusedPolicy)



        # ── ROS pub/sub ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )   

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        # ── Logging ───────────────────────────────────────────────────────────
        self.step_count = 0
        self.start_time = time.time()
        self.total_dist = 0.0
        self.prev_x = None
        self.prev_y = None

        # Use a list for O(1) appends; convert to ndarray at shutdown.
        self.inference_time_log = []

        self.get_logger().info(
            f"TD3 Actor Node ready | device={self.device} | "
            f"ppo={ppo_path} | td3={td3_path}"
        )
    
    def get_forward_mean(self,ranges: np.ndarray) -> float:
        beams_per_degree = 3
        forward_idx = 540
        window_size = 5 * beams_per_degree
        lo = forward_idx - window_size // 2
        hi = forward_idx + window_size // 2
        clean = np.nan_to_num(ranges[lo:hi], nan=30.0, posinf=30.0)
        forward_mean= np.mean(clean)
        return forward_mean #10m forward is completely straight to the car

    # ── Odometry callback ─────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_x is not None:
            step = math.sqrt((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2)
            if step < 1.0:  # reject teleport / reset jumps
                self.total_dist += step

        self.prev_x, self.prev_y = x, y

    # ── LiDAR callback ────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        # Pre-process scan
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)


        ranges = gaussian_filter1d(
                    ranges, sigma=2.0, mode="reflect"
                )


        start=0
        end=0
        
        recovery_manuever=False
        if self.needs_recov == False:
           self.needs_recov = np.min(ranges)<0.1

        if self.needs_recov and recovery_manuever: # reverse the car if it is stuck after a crash
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.steering_angle = -1.2*self.previous_steering+0.0
            drive_msg.drive.speed = -1.0 # * self.speed_scale
            self.drive_pub.publish(drive_msg)
            self.get_logger().info("reversing lmao")
        
            if np.min(ranges)>0.15:
                self.needs_recov=False

        else: # o.w. the car is free to act
            # Fix LiDAR dimensions in both directions
            if ranges.shape[0] > LIDAR_DIM:
                ranges = ranges[:LIDAR_DIM]
            elif ranges.shape[0] < LIDAR_DIM:
                # Pad with range_max if scan is short
                pad = np.full(LIDAR_DIM - ranges.shape[0], msg.range_max, dtype=np.float32)
                ranges = np.concatenate([ranges, pad])

            start = time.perf_counter()

            with torch.inference_mode():
                lidar_t = torch.from_numpy(ranges).unsqueeze(0).to(self.device)
                action=self.fusedPolicy(lidar_t)
                steering = float(action[0, 0].cpu())
                speed = float(action[0, 1].cpu())

            end = time.perf_counter()
            self.inference_time_log.append(end - start)

            # ── Scale steering from model range [-0.4, +0.4]
            #    to physical servo range [-0.25, +0.35] ───────────────────
            MODEL_RANGE = 0.4
            SERVO_MIN = -0.26  # physical right limit
            SERVO_MAX =  0.31  # physical left limit

            forward_mean=self.get_forward_mean(ranges)

            # if forward_mean < 1.5:
            #     steering = steering*2
            # elif forward_mean < 2.5:
            #     steering=steering*1.5
            # elif forward_mean <3:
            #     steering=steering*1.5

            #universe.pth
            #steer_mult = np.clip(np.interp(forward_mean, [0, 3], [2, 1.6]), 2, 1.6)
            #speed_mult = np.clip(np.interp(forward_mean, [1.5, 3.2], [0.8, 1.4]), 0.8, 1.4)
            
            #best so far
            #ultimate.pth, collected ultimate_cont, ultimate_interp
            #steer_mult = np.clip(np.interp(forward_mean, [0, 2.5], [2, 1.6]), 2, 1.6)

            #ultimate_interp2
            #steer_mult = np.interp(forward_mean, [1.0, 2.5], [1.8, 1.1]) #utliame interp2
            #speed_mult = np.interp(forward_mean, [0.65 ,1.0 ,2.0 ,3.0], [0.5,1.0,1.2,1.3]) #utliamte interp2

            #ultimate_interp_set2   
            steer_mult=np.interp(forward_mean,[1.5,2,3],[2.0,1.3,1.1])
            speed_mult=np.interp(forward_mean, [0.65,2],[0.8,1.0])


            steering*=steer_mult
            speed*=speed_mult   



            # Clamp model output to its expected range first
            steering = float(np.clip(steering, SERVO_MIN, SERVO_MAX))

            #ADD THIS REMAP TO EVERY PRE-ULTIMATE_INTERP MODEL
            # Asymmetric linear remap, keeping 0 -> 0
            # if steering >= 0:
            #     steering = steering / MODEL_RANGE * SERVO_MAX
            # else:
            #     steering = steering  / MODEL_RANGE * (-SERVO_MIN)
            speed_noise=0
            speed = float(np.clip(speed + speed_noise, self.min_spd, self.max_spd))
            elapsed_ms = (end - start) * 1000.0
            if self.step_count % 50 == 0:
                self.get_logger().info(
                    f"Step {self.step_count} | "
                    f"steer={steering:.2f}° | "
                    f"speed={speed:.2f} m/s | "
                    f"dist={self.total_dist:.2f} m | "
                    f"inference: {elapsed_ms:.2f} ms"
                )
            
            # Publish drive command
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'base_link'
            drive_msg.drive.steering_angle = steering  #get rid of the 1.5
            drive_msg.drive.speed = speed # * self.speed_scale
            self.drive_pub.publish(drive_msg)
            self.previous_steering=steering

        # Console logging every 50 steps
            self.step_count+=1


    def destroy_node(self):
        if len(self.inference_time_log) > 0:
            arr = np.asarray(self.inference_time_log) * 1000.0  # ms
            self.get_logger().info(
                f"Inference latency over {arr.size} steps | "
                f"mean={arr.mean():.2f} ms | "
                f"median={np.median(arr):.2f} ms | "
                f"p95={np.percentile(arr, 95):.2f} ms | "
                f"max={arr.max():.2f} ms"
            )
        else:
            self.get_logger().info("No inference steps recorded.")
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
