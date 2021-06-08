import gym
import time
start_time = time.time()

env = gym.make('HalfCheetah-v3')
for i_episode in range(1000):
    observation = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
print("--- %s seconds ---" % (time.time() - start_time))