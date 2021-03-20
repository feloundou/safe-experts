import gym
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "recording", force=True)
o = env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        break