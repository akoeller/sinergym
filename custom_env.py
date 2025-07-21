import sinergym
import gymnasium as gym
import logging
from sinergym.utils.logger import TerminalLogger
import numpy as np

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
        a = env.action_space.sample()

        print("Action: ", a)

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        # Display results every simulated month
        if info['month'] != current_month:
            current_month = info['month']
            logger.info('Reward: {}'.format(sum(rewards)))
            logger.info('Info: {}'.format(info))

    logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
                np.mean(rewards), sum(rewards)))
env.close()
