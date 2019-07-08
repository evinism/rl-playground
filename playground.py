import gym

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

episode_count = 3
env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        env.render()
        if done:
            break



input("Press Enter to continue...")
