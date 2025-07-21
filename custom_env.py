import numpy as np
from sinergym.utils.logger import TerminalLogger
import logging
import gymnasium as gym
import sinergym
import sys


# This code is to force a re-registration of sinergym environments
# Sinergym registers environments when it is imported. If the configuration files for the environments
# are changed, the environments need to be re-registered for the changes to take effect.
# We remove the sinergym modules from the system's cache, so that the next
# import will re-run the registration code.
modules_to_remove = [
    name for name in sys.modules if name.startswith('sinergym')]
for module_name in modules_to_remove:
    del sys.modules[module_name]


# Logger
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='MAIN',
    level=logging.INFO
)


env = gym.make("Eplus-2room-mild-continuous-v1")

print(env.action_space)
print(env.observation_space)


# Execute episodes
for i in range(1):

    # Reset the environment to start a new episode
    obs, info = env.reset()

    rewards = []
    truncated = terminated = False
    current_month = 0

    while not (terminated or truncated):

        # Random action selection
        # a = env.action_space.sample() # sample() can be used now
        a = np.array([21.0, 21.0], dtype=np.float32)

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

        # print(f"Action: {a}, Reward: {reward}, Obs: {obs}")

        rewards.append(reward)

        # Display results every simulated month
        # if info['month'] != current_month:
        #     current_month = info['month']
        #     logger.info('Reward: {}'.format(sum(rewards)))
        #     logger.info('Info: {}'.format(info))

    logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
                np.mean(rewards), sum(rewards)))
env.close()
