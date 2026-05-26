import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ast
import copy
import csv
import numpy as np
from ppo_model import ActorCritic

import argparse

parser =argparse.ArgumentParser()
parser.add_argument('--infile',type=str,default='raw_states_charlie.csv')
parser.add_argument('--name',type=str,default='dumdum')
parser.add_argument('--starter',type=str,default='td3_plock.pth')


args=parser.parse_args()
#do args.data to retrieve name of training file

#alfie.pth trained on td3 to correct weaving from ppo
#brooke.pth is an absolute failure
#charlie.pth trained on real world data

class TD3_Actor(nn.Module):
    def __init__(self, latent_dim=128, action_dim=2):
        super().__init__()

        # minimal change: replace PPO dependency with clean head
        self.actor_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
        )

        self.mean_layer = nn.Linear(64, action_dim)

    def squash(self, raw):
        steering = torch.tanh(raw[..., 0:1]) * 0.4
        speed = torch.sigmoid(raw[..., 1:2]) * (3.5 - 0.5) + 0.5
        return torch.cat([steering, speed], dim=-1)

    def forward(self, latent):
        feat = self.actor_head(latent)
        mean = self.mean_layer(feat)
        return self.squash(mean)


class Critic_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SARSD_dataset(Dataset):
    def __init__(self, csv_path):
        states, actions, rewards, state_primes, dones = [], [], [], [], []

        row_counter = 0
        with open(csv_path, 'r') as oogabooga:
            reader = csv.DictReader(oogabooga)
            for row in reader:
                states.append(ast.literal_eval(row['state']))
                actions.append(ast.literal_eval(row['action']))
                rewards.append(float(row['reward']))
                state_primes.append(ast.literal_eval(row['state_prime']))
                dones.append(int(row['done']))
                row_counter=row_counter+1

                if row_counter%1000==0:
                    print(f"read {row_counter} rows")
                

        self.states = torch.FloatTensor(np.array(states))
        self.actions = torch.FloatTensor(np.array(actions))
        self.rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        self.state_primes = torch.FloatTensor(np.array(state_primes))
        self.dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.state_primes[idx], self.dones[idx]


class TD3_train():
    def __init__(self, state_dim=128, action_dim=2):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize Networks
        self.critic1 = Critic_Net(state_dim, action_dim).to(self.device)
        self.critic2 = Critic_Net(state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Using your ActorCritic class (kept, but NOT used now)
        self.actor = TD3_Actor(state_dim, action_dim).to(self.device)
        
        soft_start_model = f'models/{args.starter}'
        self.actor.load_state_dict(torch.load(soft_start_model))
        self.actor_target = copy.deepcopy(self.actor)

        # Define Optimizers
        self.critic_optimizer1 = torch.optim.Adam(
            list(self.critic1.parameters()), lr=1e-4
        )
        self.critic_optimizer2 = torch.optim.Adam(
            list(self.critic2.parameters()), lr=1e-4
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_warmup = 400
        # Load Buffer
        dataset = SARSD_dataset(f'transitions/{args.infile}')
        print(dataset.dones.min(), dataset.dones.max(), dataset.dones.mean())
        self.dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        print(f"reward min: {dataset.rewards.min():.4f}")
        print(f"reward max: {dataset.rewards.max():.4f}")
        print("positive reward count:", (dataset.rewards > 0).sum())
        print("negative reward count:", (dataset.rewards < 0).sum())
        print("reward mean:", dataset.rewards.mean())
        print("reward std:", dataset.rewards.std())
        print("straight actions reward:", dataset.rewards[dataset.actions[:,0].abs() < 0.05].mean())
        print("weaving actions reward:", dataset.rewards[dataset.actions[:,0].abs() > 0.15].mean())
        print("done=1 count:", dataset.dones.sum())
        print("wall penalty count:", (dataset.rewards < -5).sum())

        self.iterations = 800
        self.noise = 0.2
        self.policy_delay = 2
        self.tau = 0.005
        self.gamma = 0.90

    def train_model(self):

        
        state=None
        action=None
        reward=None
        state_prime=None
        done=None
        actor_loss = torch.tensor(0.0)  # default value before warmup ends
        bc_loss = torch.tensor(0.0)   
        data_iter = iter(self.dataloader)  # add this line before the loop

        for x in range(self.iterations):

            try:
                state, action, reward, state_prime, done = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)  # reset and reshuffle
                state, action, reward, state_prime, done = next(data_iter)

            state, action, reward, state_prime, done = [ t.to(self.device) for t in [state, action, reward, state_prime, done]]

            with torch.no_grad():

                action_prime = self.actor_target(state_prime)

                noise = torch.randn_like(action_prime) * self.noise
                noise = noise.clamp(-0.5, 0.5)

                action_prime = action_prime + noise

                steering = action_prime[:, 0:1].clamp(-0.4, 0.4)
                speed = action_prime[:, 1:2].clamp(0.5, 3.5)

                action_prime = torch.cat([steering, speed], dim=1)

                target_q1 = self.critic1_target(state_prime, action_prime)
                target_q2 = self.critic2_target(state_prime, action_prime)

                target_q = torch.min(target_q1, target_q2)

                y = reward + self.gamma * (1 - done) * target_q


            # delay policys

                        
            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)

            critic_loss1 = nn.functional.mse_loss(current_q1, y)

            self.critic_optimizer1.zero_grad()
            critic_loss1.backward()
            self.critic_optimizer1.step()


            critic_loss2 = nn.functional.mse_loss(current_q2, y) 

            self.critic_optimizer2.zero_grad()
            critic_loss2.backward()
            self.critic_optimizer2.step()

            if x > self.critic_warmup and x % self.policy_delay == 0:

                # in actor update
                action_batch = self.actor(state)
                actor_loss = -self.critic1(state, action_batch).mean()

                # add BC regularization — penalize deviation from buffer actions
                bc_loss = nn.functional.mse_loss(action_batch, action)
                actor_loss = actor_loss + 1.5*  bc_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft Update Target Networks
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if x % 100 == 0:
                print(f"epoch {x}")
                print(f"critic_loss1: {critic_loss1}")
                print(f"critic_loss2: {critic_loss2}")
                print(f"actor_loss: {actor_loss}")
                print(f"mean Q1: {current_q1.mean().item():.4f}")
                print(f"Q1 std: {current_q1.std().item():.4f}")
                print(f"mean target y: {y.mean().item():.4f}")
                print("*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^")


        test_latent = torch.zeros(1, 128).to(self.device)
        test_action = self.actor(test_latent)
        print(f"zero latent action: {test_action}")
        # also check on a real sample from your buffer
        state_sample = next(iter(self.dataloader))[0][0:1].to(self.device)
        print(f"real state action: {self.actor(state_sample)}")
        name = args.infile.replace('sarsd_buffer_', '').replace('.csv', '')
        torch.save(self.actor.state_dict(), f'models/td3_{args.name}.pth')



if __name__ == "__main__":
    print(f"training file:{args.infile}")
    trainer = TD3_train()
    trainer.train_model()
