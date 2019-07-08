import gym
import numpy as np
import random

class RandomAgent(object):
    """Has a matrix define the operation"""
    def __init__(self, action_space, mat = np.concatenate((np.random.rand(8, 4), np.random.rand(1, 4)), axis=0)):
        self.action_space = action_space
        self.mat = mat
        self.is_invalid = False

    def act(self, observation, reward, done):
        observation = observation
        composite = np.concatenate((observation, [1]))
        action_amts = np.matmul(composite, self.mat)
        return np.argmax(action_amts)
    
    def progenate(self, rep_temp = 2):
        actual_temp = random.random() * rep_temp * 2
        add_mat = np.random.rand(9, 4) * actual_temp
        new_mat = self.mat + add_mat
        return RandomAgent(self.action_space, mat = new_mat)

def combine_agents(agents):
    new_mat = np.mean([agent.mat for agent in agents], axis=0)
    return RandomAgent(env.action_space, mat = new_mat)


def evaluate_agent(agent, should_render = False):
    reward = 0
    total_reward = 0
    done = False
    #env.seed(5)
    ob = env.reset()
    #num_of_steps_to_take = int(200 * random.random())
    prev_ob = None
    p = True
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if p:
            prev_ob = ob
            p = False
        total_reward = total_reward * 1 + reward
        if should_render:
            print(total_reward)
            env.render()
        if agent.is_invalid:
            return -float('inf')
        if done:
            return total_reward
            break
    return total_reward

episode_count = 1000
env = gym.make('LunarLander-v2')
num_of_agents = 100
max_to_keep = 7

print(env.action_space.n)

agents = []
for i in range(num_of_agents):
    agents.append(RandomAgent(env.action_space))

for i in range(episode_count):
    print("generation " + str(i) + " of " + str(episode_count))
    agent_scores = []
    for agent in agents:
        score1 = evaluate_agent(agent)
        score2 = evaluate_agent(agent)
        score3 = evaluate_agent(agent)
        score = (score1 + score2 + score3) / 3
        agent_scores.append((agent, score))


    def take_second(elem):
        return elem[1]
    agent_scores.sort(key=take_second, reverse=True)

    print(agent_scores[0][1])
    print(agent_scores[1][1])
    print(evaluate_agent(agent_scores[0][0], True))
    agents = [score_record[0] for score_record in agent_scores[:max_to_keep]]
    combined_agent = combine_agents(agents)
    agents.append(combined_agent) # recombination
    evaluate_agent(combined_agent, True)
    for i in range(int(num_of_agents * 2 / 3)):
        agents.append(agents[i % max_to_keep + 1].progenate())
    print(str(len(agents)) + ' of ' + str(num_of_agents))

    while len(agents) < num_of_agents:
        agents.append(RandomAgent(env.action_space))



input("Press Enter to continue...")
