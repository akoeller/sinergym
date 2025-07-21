import sinergym
import gymnasium as gym

env = gym.make("Eplus-2room-mild-continuous-v1")

# Execute 3 episodes
for i in range(3):

    # Reset the environment to start a new episode
    obs, info = env.reset()
    print(obs)
