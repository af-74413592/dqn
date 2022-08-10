import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 10

n_episode = 5000
n_time_step = 500

class Replaymemory:
    def __init__(self,n_s,n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.MEMORY_SIZE = 1000
        self.BATCH_SIZE = 64

        self.all_s = np.empty(shape=(self.MEMORY_SIZE,self.n_s),dtype=np.float32)
        self.all_a = np.random.randint(low=0,high=n_a,size=self.MEMORY_SIZE,dtype=np.uint8)
        self.all_r = np.empty(self.MEMORY_SIZE,dtype=np.float32)
        self.all_done = np.random.randint(low=0,high=2,size=self.MEMORY_SIZE,dtype=np.uint8)
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE,self.n_s),dtype=np.float32)
        self.t_memo = 0
        self.t_max = 0

    def add_memo(self,s,a,r,done,s_):
        self.all_s[self.t_memo] = s
        self.all_a[self.t_memo] = a
        self.all_r[self.t_memo] = r
        self.all_done[self.t_memo] = done
        self.all_s_[self.t_memo] = s_
        self.t_max = max(self.t_max,self.t_memo+1)
        self.t_memo = (self.t_memo +1) % self.MEMORY_SIZE

    def sample(self):
        if self.t_max > self.BATCH_SIZE:
            idxes = random.sample(range(self.t_max),self.BATCH_SIZE)
        else:
            idxes = range(0,self.t_max)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []
        for idx in idxes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s),dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(np.asarray(batch_a), dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(np.asarray(batch_r), dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.int64).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return batch_s_tensor,batch_a_tensor,batch_r_tensor,batch_done_tensor,batch_s__tensor

class DQN(nn.Module):
    def __init__(self,n_input,n_output):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=n_input,out_features=88),
            nn.Tanh(),
            nn.Linear(in_features=88,out_features=n_output)
        )

    def forward(self,x):
        return self.net(x)

    def act(self,obs):
        obs_tensor = torch.as_tensor(obs,dtype=torch.float32)
        q_value = self.forward(obs_tensor.unsqueeze(0))
        max_q_idx = torch.argmax(input=q_value)
        action = max_q_idx.detach().item()
        return action

class Agent:
    def __init__(self,n_input,n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.99
        self.learning_rate = 1e-3

        self.memo = Replaymemory(self.n_input,self.n_output)

        self.online_net = DQN(self.n_input,self.n_output)
        self.target_net = DQN(self.n_input,self.n_output)

        self.optimzer = torch.optim.Adam(self.online_net.parameters(),lr=self.learning_rate)

env = gym.make("CartPole-v1")
s = env.reset()
n_state = len(s)
n_action = env.action_space.n
agent = Agent(n_input=n_state,n_output=n_action)
REWARD_BUFFER = np.empty(shape=n_episode)
for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        #插值衰减
        epsilon = np.interp(episode_i * n_time_step + step_i,[0,EPSILON_DECAY],[EPSILON_START,EPSILON_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)
        s_,r,done,info = env.step(a)
        agent.memo.add_memo(s,a,r,done,s_)
        s = s_
        episode_reward += r

        if done:
            s = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break
        if np.mean(REWARD_BUFFER[:episode_i]) >= 100: #训练完成，状态还原
            while True:
                a = agent.online_net.act(s)
                s, r, done, info = env.step(a)
                env.render()
                if done:
                    env.reset()
        batch_s,batch_a,batch_r,batch_done,batch_s_ = agent.memo.sample()

        #Compute targets
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(dim=1,keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1-batch_done) * max_target_q_values

        #Compute q_values
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input=q_values,dim=1,index=batch_a)

        #compute loss
        loss = F.smooth_l1_loss(targets,a_q_values)

        #Gradient descent
        agent.optimzer.zero_grad()
        loss.backward()
        agent.optimzer.step()

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print("Episode:{}".format(episode_i))
        print("Avg. Reward:{:.2f}".format(np.mean(REWARD_BUFFER[:episode_i])))
