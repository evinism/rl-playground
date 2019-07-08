import gym
import numpy as np
import random


class RandomAgent(object):
    """Has a matrix define the operation"""
    def __init__(self, action_space, mat = np.concatenate((np.random.rand(8, 4), np.zeros((8, 4)), np.random.rand(1, 4)), axis=0)):
        self.action_space = action_space
        self.mat = mat
        self.action_temp = 0.01
        self.is_invalid = False
        self.prev_observation = None

    def act(self, observation, reward, done):
        if self.prev_observation is None:
            self.prev_observation = observation
            return self.action_space.sample()
        composite = np.concatenate((observation, (observation - self.prev_observation), [1]))
        distrib = 0.5 ** (np.matmul(composite, self.mat) / self.action_temp)
        distrib = distrib / sum(distrib)
        if np.isnan(sum(distrib)):
            self.is_invalid = True
            return 1
        action = np.random.choice(4, p=distrib)
        self.prev_observation = observation
        return action
    
    def progenate(self, rep_temp = 0.3):
        actual_temp = random.random() * rep_temp * 2
        add_mat = np.zeros((17, 4))
        for i in range(int(17 * 4 * 0.2)):
            x = random.randint(0, 17 - 1)
            y = random.randint(0, 4 - 1)
            add_mat[x][y] = (random.random() - 0.5) * actual_temp
        new_mat = self.mat + add_mat
        return RandomAgent(self.action_space, mat = new_mat)


def evaluate_agent(agent, should_render = False):
    reward = 0
    total_reward = 0
    done = False
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        total_reward = reward + total_reward * 0.6
        if should_render:
            print(total_reward)
            env.render()
        if agent.is_invalid:
            return -float('inf')
        if done:
            return total_reward 
            break

episode_count = 30
env = gym.make('LunarLander-v2')
num_of_agents = 100
max_to_keep = 5

agents = []
for i in range(num_of_agents):
    agents.append(RandomAgent(env.action_space))

for i in range(episode_count):
    agent_scores = []
    for agent in agents:
        score1 = evaluate_agent(agent)
        score2 = evaluate_agent(agent)
        score3 = 0   
        score = (score1 + score2 + score3) / 3
        agent_scores.append((agent, score))

    def take_second(elem):
        return elem[1]
    agent_scores.sort(key=take_second, reverse=True)

    evaluate_agent(agent_scores[0][0], True)
    agents = [score_record[0] for score_record in agent_scores[:max_to_keep]]
    for i in range(int(num_of_agents / 2)):
        agents.append(agent_scores[i % max_to_keep][0].progenate())

    while len(agents) < num_of_agents:
        agents.append(RandomAgent(env.action_space))



input("Press Enter to continue...")
