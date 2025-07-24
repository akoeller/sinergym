import numpy as np
import logging
import gymnasium as gym
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)
from controller import MyRuleBasedController


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


def display_results():
    """Display simulation results from EnergyPlus output files and monitoring data using matplotlib."""

    # Find the most recent results directory
    result_dirs = [d for d in os.listdir('.') if d.startswith(
        'Eplus-2room-mild-continuous-v1-res')]
    if not result_dirs:
        logger.warning("No results directory found")
        return

    latest_dir = max(result_dirs, key=lambda x: int(x.split('res')[-1]))
    monitor_path = Path(latest_dir) / 'episode-1' / 'monitor'
    output_path = Path(latest_dir) / 'episode-1' / 'output'

    if not monitor_path.exists():
        logger.warning(f"Monitor directory not found: {monitor_path}")
        return

    try:
        # Read monitoring data
        # Define file paths
        observations_file = monitor_path / 'observations.csv'
        actions_file = monitor_path / 'agent_actions.csv'
        rewards_file = monitor_path / 'rewards.csv'

        if not observations_file.exists() or not actions_file.exists():
            logger.warning("Monitoring files not found")
            return

        df_obs = pd.read_csv(observations_file)
        df_actions = pd.read_csv(actions_file)
        df_rewards = pd.read_csv(
            rewards_file) if rewards_file.exists() else None

        # Create time steps for plotting - align lengths
        # Actions are one step shorter than observations (no action after final
        # obs)
        obs_timesteps = range(len(df_obs))
        action_timesteps = range(len(df_actions))

        # For aligned plotting, use shorter length
        aligned_length = min(len(df_obs), len(df_actions))
        aligned_timesteps = range(aligned_length)

        # Create subplots with proper spacing (added row for rewards)
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        fig.suptitle(
            'Sinergym Monitoring Results',
            fontsize=16,
            fontweight='bold')

        # Plot 1: Temperature Control - Living Room
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()

        # Temperature observations
        ax1.plot(
            obs_timesteps,
            df_obs['air_temp_lr'],
            'b-',
            label='LR Temperature',
            linewidth=1.5)
        ax1.plot(
            obs_timesteps,
            df_obs['outdoor_temperature'],
            'g--',
            label='Outdoor Temp',
            alpha=0.7)

        # Heating setpoint actions (normalized, need to scale back to
        # meaningful range)
        ax1_twin.plot(
            action_timesteps,
            df_actions['Heating_Setpoint_LR'],
            'r-',
            alpha=0.8,
            label='Heating Setpoint Action',
            linewidth=1)

        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Temperature [°C]', color='b')
        ax1_twin.set_ylabel('Action Value [-1,1]', color='r')
        ax1.set_title('Living Room Temperature Control')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # Plot 2: Temperature Control - Bedroom
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()

        # Temperature observations
        ax2.plot(
            obs_timesteps,
            df_obs['air_temp_br'],
            'b-',
            label='BR Temperature',
            linewidth=1.5)
        ax2.plot(
            obs_timesteps,
            df_obs['outdoor_temperature'],
            'g--',
            label='Outdoor Temp',
            alpha=0.7)

        # Heating setpoint actions
        ax2_twin.plot(
            action_timesteps,
            df_actions['Heating_Setpoint_BR'],
            'r-',
            alpha=0.8,
            label='Heating Setpoint Action',
            linewidth=1)

        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Temperature [°C]', color='b')
        ax2_twin.set_ylabel('Action Value [-1,1]', color='r')
        ax2.set_title('Bedroom Temperature Control')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')

        # Plot 3: Energy Consumption vs Actions - Living Room
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()

        ax3.plot(
            aligned_timesteps,
            df_obs['baseboard_total_heating_energy_lr'][:aligned_length],
            'orange',
            label='LR Heating Energy',
            linewidth=1.5)
        ax3_twin.plot(
            action_timesteps,
            df_actions['Heating_Setpoint_LR'],
            'r-',
            alpha=0.8,
            label='Heating Action',
            linewidth=1)

        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Energy [J]', color='orange')
        ax3_twin.set_ylabel('Action Value [-1,1]', color='r')
        ax3.set_title('Living Room: Energy vs Actions')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')

        # Plot 4: Energy Consumption vs Actions - Bedroom
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        ax4.plot(
            aligned_timesteps,
            df_obs['baseboard_total_heating_energy_br'][:aligned_length],
            'purple',
            label='BR Heating Energy',
            linewidth=1.5)
        ax4_twin.plot(
            action_timesteps,
            df_actions['Heating_Setpoint_BR'],
            'r-',
            alpha=0.8,
            label='Heating Action',
            linewidth=1)

        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Energy [J]', color='purple')
        ax4_twin.set_ylabel('Action Value [-1,1]', color='r')
        ax4.set_title('Bedroom: Energy vs Actions')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')

        # Plot 5: Total Gas Energy vs Combined Actions
        ax5 = axes[2, 0]
        ax5_twin = ax5.twinx()

        ax5.plot(aligned_timesteps,
                 df_obs['boiler_natural_gas_energy'][:aligned_length],
                 'brown',
                 label='Total Gas Energy',
                 linewidth=1.5)

        # Combined action signal
        combined_actions = (
            df_actions['Heating_Setpoint_LR'] + df_actions['Heating_Setpoint_BR']) / 2
        ax5_twin.plot(action_timesteps, combined_actions, 'r-', alpha=0.8,
                      label='Combined Actions', linewidth=1)

        ax5.set_xlabel('Timestep')
        ax5.set_ylabel('Gas Energy [J]', color='brown')
        ax5_twin.set_ylabel('Average Action [-1,1]', color='r')
        ax5.set_title('Total Gas Energy vs Combined Actions')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')

        # Plot 6: Action Correlation
        ax6 = axes[2, 1]
        ax6.scatter(
            df_actions['Heating_Setpoint_LR'],
            df_actions['Heating_Setpoint_BR'],
            alpha=0.5,
            s=1,
            c=action_timesteps,
            cmap='viridis')
        ax6.set_xlabel('Living Room Heating Action')
        ax6.set_ylabel('Bedroom Heating Action')
        ax6.set_title('Action Correlation Over Time')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)

        # Plot 7: Reward Signal (RL-style)
        if df_rewards is not None:
            ax7 = axes[3, 0]
            reward_timesteps = range(len(df_rewards))

            # Plot instantaneous rewards
            ax7.plot(reward_timesteps, df_rewards['reward'], 'r-', alpha=0.7,
                     linewidth=0.8, label='Instantaneous Reward')

            # Calculate and plot moving average (window size 100)
            window_size = min(100, len(df_rewards) // 10)
            if window_size > 1:
                moving_avg = df_rewards['reward'].rolling(
                    window=window_size, center=True).mean()
                ax7.plot(
                    reward_timesteps,
                    moving_avg,
                    'darkred',
                    linewidth=2,
                    label=f'Moving Average ({window_size} steps)')

            ax7.set_xlabel('Timestep')
            ax7.set_ylabel('Reward')
            ax7.set_title('Reward Signal Over Time')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)

            # Plot 8: Cumulative Reward (RL-style)
            ax8 = axes[3, 1]
            cumulative_rewards = df_rewards['reward'].cumsum()

            ax8.plot(reward_timesteps, cumulative_rewards, 'darkgreen',
                     linewidth=2, label='Cumulative Reward')

            # Add episode performance trend line
            if len(cumulative_rewards) > 10:
                z = np.polyfit(reward_timesteps, cumulative_rewards, 1)
                p = np.poly1d(z)
                ax8.plot(
                    reward_timesteps,
                    p(reward_timesteps),
                    'orange',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.8,
                    label='Trend Line')

            ax8.set_xlabel('Timestep')
            ax8.set_ylabel('Cumulative Reward')
            ax8.set_title('Cumulative Reward Progress')
            ax8.grid(True, alpha=0.3)
            ax8.legend()

            # Add performance statistics as text
            final_reward = cumulative_rewards.iloc[-1]
            mean_reward = df_rewards['reward'].mean()
            std_reward = df_rewards['reward'].std()
            ax8.text(
                0.02,
                0.98,
                f'Final: {
                    final_reward:.1f}\nMean: {
                    mean_reward:.3f}\nStd: {
                    std_reward:.3f}',
                transform=ax8.transAxes,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='wheat',
                    alpha=0.8))

        plt.tight_layout()
        plt.savefig('monitoring_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Monitoring plots saved as 'monitoring_results.png'")

        # Display summary statistics
        logger.info("\n=== MONITORING SUMMARY ===")
        logger.info(f"Total timesteps: {len(df_obs)}")
        logger.info(
            f"Average LR temperature: {
                df_obs['air_temp_lr'].mean():.2f}°C")
        logger.info(
            f"Average BR temperature: {
                df_obs['air_temp_br'].mean():.2f}°C")
        logger.info(
            f"Total gas energy consumed: {
                df_obs['boiler_natural_gas_energy'].sum():.0f} J")
        logger.info(
            f"LR action range: [{
                df_actions['Heating_Setpoint_LR'].min():.2f}, {
                df_actions['Heating_Setpoint_LR'].max():.2f}]")
        logger.info(
            f"BR action range: [{
                df_actions['Heating_Setpoint_BR'].min():.2f}, {
                df_actions['Heating_Setpoint_BR'].max():.2f}]")

        # Add reward statistics if available
        if df_rewards is not None:
            logger.info("\n=== REWARD SUMMARY ===")
            logger.info(f"Total reward steps: {len(df_rewards)}")
            logger.info(
                f"Mean reward per step: {
                    df_rewards['reward'].mean():.4f}")
            logger.info(
                f"Std reward per step: {
                    df_rewards['reward'].std():.4f}")
            logger.info(f"Cumulative reward: {df_rewards['reward'].sum():.2f}")
            logger.info(f"Best reward: {df_rewards['reward'].max():.4f}")
            logger.info(f"Worst reward: {df_rewards['reward'].min():.4f}")

            # Episode performance indicators
            positive_rewards = (df_rewards['reward'] > 0).sum()
            negative_rewards = (df_rewards['reward'] < 0).sum()
            logger.info(
                f"Positive reward steps: {positive_rewards} ({
                    positive_rewards /
                    len(df_rewards) *
                    100:.1f}%)")
            logger.info(
                f"Negative reward steps: {negative_rewards} ({
                    negative_rewards /
                    len(df_rewards) *
                    100:.1f}%)")

    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        import traceback
        logger.error(traceback.format_exc())


def run_agent_episode(env, agent_type, agent):
    """
    Run a complete episode with a given agent.

    Args:
        env: The environment to run in
        agent_type: String identifier for the agent type ('random' or 'rbc')
        agent: Agent object with an act() method

    Returns:
        dict: Results containing rewards, observations, actions, and metrics
    """
    obs, info = env.reset()

    total_reward = 0
    rewards = []
    observations = []
    actions = []

    done = False
    truncated = False
    step_count = 0

    try:
        while not (done or truncated):
            if agent_type == 'random':
                a = env.action_space.sample()
            else:
                a = agent.act(obs)

            obs, reward, done, truncated, info = env.step(a)

            rewards.append(reward)
            observations.append(obs.copy())
            actions.append(a.copy() if hasattr(a, 'copy') else a)
            total_reward += reward
            step_count += 1

    except Exception as e:
        print(f"Error during {agent_type} agent execution: {e}")
        raise e

    mean_reward = total_reward / step_count if step_count > 0 else 0

    return {
        'rewards': rewards,
        'observations': observations,
        'actions': actions,
        'cumulative_reward': total_reward,
        'total_reward': total_reward,
        'mean_reward': mean_reward,
        'total_steps': step_count,
        'steps': step_count
    }


def compare_agents():
    """Compare Random Agent vs Rule-Based Controller performance."""

    logger.info("\n" + "=" * 60)
    logger.info("COMPARING RANDOM AGENT vs RULE-BASED CONTROLLER")
    logger.info("=" * 60)

    results = {}
    agent_dirs = {}  # Track directories for each agent

    # Run Random Agent
    logger.info("\n--- Running Random Agent ---")
    env = gym.make("Eplus-2room-mild-continuous-v1")
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)

    # Get the current working directory before running
    before_dirs = set(os.listdir('.'))

    random_results = run_agent_episode(env, 'random', None)
    results['random'] = random_results

    # Find the new directory created for random agent
    after_dirs = set(os.listdir('.'))
    new_dirs = after_dirs - before_dirs
    random_dir = next((d for d in new_dirs if d.startswith(
        'Eplus-2room-mild-continuous-v1-res')), None)
    agent_dirs['random'] = random_dir

    logger.info(
        f"Random Agent - Mean Reward: {random_results['mean_reward']:.4f}")
    logger.info(
        f"Random Agent - Cumulative Reward: {random_results['cumulative_reward']:.2f}")
    logger.info(f"Random Agent - Total Steps: {random_results['total_steps']}")

    env.close()

    # Run Rule-Based Controller
    logger.info("\n--- Running Rule-Based Controller ---")
    # RBC needs raw observations, not normalized ones
    env = gym.make("Eplus-2room-mild-continuous-v1")
    env = LoggerWrapper(env)
    env = CSVLogger(env)

    # Create RBC agent (no need to pass env)
    # temp_env = gym.make("Eplus-2room-mild-continuous-v1")
    rbc_agent = MyRuleBasedController(env)
    # temp_env.close()

    # Get directories before RBC run
    before_dirs = set(os.listdir('.'))

    rbc_results = run_agent_episode(env, 'rbc', rbc_agent)
    results['rbc'] = rbc_results

    # Find the new directory created for RBC agent
    after_dirs = set(os.listdir('.'))
    new_dirs = after_dirs - before_dirs
    rbc_dir = next((d for d in new_dirs if d.startswith(
        'Eplus-2room-mild-continuous-v1-res')), None)
    agent_dirs['rbc'] = rbc_dir

    logger.info(f"RBC Agent - Mean Reward: {rbc_results['mean_reward']:.4f}")
    logger.info(
        f"RBC Agent - Cumulative Reward: {rbc_results['cumulative_reward']:.2f}")
    logger.info(f"RBC Agent - Total Steps: {rbc_results['total_steps']}")

    env.close()

    # Performance Comparison
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 60)

    improvement_mean = ((rbc_results['mean_reward'] -
                         random_results['mean_reward']) /
                        abs(random_results['mean_reward']) *
                        100)
    improvement_cumulative = (
        (rbc_results['cumulative_reward'] - random_results['cumulative_reward']) / abs(
            random_results['cumulative_reward']) * 100)

    logger.info(f"Mean Reward Improvement: {improvement_mean:+.2f}%")
    logger.info(
        f"Cumulative Reward Improvement: {
            improvement_cumulative:+.2f}%")

    # Determine winner
    if rbc_results['cumulative_reward'] > random_results['cumulative_reward']:
        winner = "Rule-Based Controller"
        winner_advantage = rbc_results['cumulative_reward'] - \
            random_results['cumulative_reward']
    else:
        winner = "Random Agent"
        winner_advantage = random_results['cumulative_reward'] - \
            rbc_results['cumulative_reward']

    logger.info(
        f"Winner: {winner} (advantage: {
            winner_advantage:+.2f} total reward)")

    # Action Analysis
    logger.info(f"\n--- Action Analysis ---")
    random_actions = np.array(random_results['actions'])
    rbc_actions = np.array(rbc_results['actions'])

    logger.info(
        f"Random Agent - LR Action Range: [{random_actions[:, 0].min():.2f}, {random_actions[:, 0].max():.2f}]")
    logger.info(
        f"Random Agent - BR Action Range: [{random_actions[:, 1].min():.2f}, {random_actions[:, 1].max():.2f}]")
    logger.info(
        f"Random Agent - Action Std: LR={random_actions[:, 0].std():.2f}, BR={random_actions[:, 1].std():.2f}")

    logger.info(
        f"RBC Agent - LR Action Range: [{rbc_actions[:, 0].min():.2f}, {rbc_actions[:, 0].max():.2f}]")
    logger.info(
        f"RBC Agent - BR Action Range: [{rbc_actions[:, 1].min():.2f}, {rbc_actions[:, 1].max():.2f}]")
    logger.info(
        f"RBC Agent - Action Std: LR={rbc_actions[:, 0].std():.2f}, BR={rbc_actions[:, 1].std():.2f}")

    return results, agent_dirs


def create_comparison_plot(results):
    """Create agent comparison plots and individual monitoring plots."""
    """Create visualization comparing both agents."""

    plt.figure(figsize=(16, 12))

    # Extract data
    random_rewards = results['random']['rewards']
    rbc_rewards = results['rbc']['rewards']

    # Ensure same length for comparison
    min_length = min(len(random_rewards), len(rbc_rewards))
    random_rewards = random_rewards[:min_length]
    rbc_rewards = rbc_rewards[:min_length]
    timesteps = range(min_length)

    # Plot 1: Reward Comparison
    plt.subplot(2, 3, 1)
    plt.plot(
        timesteps,
        random_rewards,
        'r-',
        alpha=0.7,
        linewidth=0.8,
        label='Random Agent')
    plt.plot(
        timesteps,
        rbc_rewards,
        'b-',
        alpha=0.7,
        linewidth=0.8,
        label='RBC Agent')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Instantaneous Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Cumulative Rewards
    plt.subplot(2, 3, 2)
    random_cumulative = np.cumsum(random_rewards)
    rbc_cumulative = np.cumsum(rbc_rewards)
    plt.plot(
        timesteps,
        random_cumulative,
        'r-',
        linewidth=2,
        label='Random Agent')
    plt.plot(timesteps, rbc_cumulative, 'b-', linewidth=2, label='RBC Agent')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Moving Average Rewards
    plt.subplot(2, 3, 3)
    window_size = min(100, len(random_rewards) // 10)
    if window_size > 1:
        random_ma = pd.Series(random_rewards).rolling(
            window=window_size, center=True).mean()
        rbc_ma = pd.Series(rbc_rewards).rolling(
            window=window_size, center=True).mean()
        plt.plot(timesteps, random_ma, 'r-', linewidth=2,
                 label=f'Random Agent (MA-{window_size})')
        plt.plot(timesteps, rbc_ma, 'b-', linewidth=2,
                 label=f'RBC Agent (MA-{window_size})')
    plt.xlabel('Timestep')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Action Comparison - Living Room
    plt.subplot(2, 3, 4)
    random_actions = np.array(results['random']['actions'])
    rbc_actions = np.array(results['rbc']['actions'])
    plt.plot(timesteps,
             random_actions[:min_length,
                            0],
             'r-',
             alpha=0.7,
             label='Random Agent')
    plt.plot(timesteps, rbc_actions[:min_length, 0],
             'b-', alpha=0.7, label='RBC Agent')
    plt.xlabel('Timestep')
    plt.ylabel('Heating Setpoint')
    plt.title('Living Room Actions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Action Comparison - Bedroom
    plt.subplot(2, 3, 5)
    plt.plot(timesteps,
             random_actions[:min_length,
                            1],
             'r-',
             alpha=0.7,
             label='Random Agent')
    plt.plot(timesteps, rbc_actions[:min_length, 1],
             'b-', alpha=0.7, label='RBC Agent')
    plt.xlabel('Timestep')
    plt.ylabel('Heating Setpoint')
    plt.title('Bedroom Actions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Performance Statistics
    plt.subplot(2, 3, 6)
    agents = ['Random', 'RBC']
    mean_rewards = [
        results['random']['mean_reward'],
        results['rbc']['mean_reward']]
    cumulative_rewards = [
        results['random']['cumulative_reward'],
        results['rbc']['cumulative_reward']]

    x = np.arange(len(agents))
    width = 0.35

    plt.bar(x - width / 2, mean_rewards, width, label='Mean Reward', alpha=0.8)
    plt.bar(x + width / 2,
            [r / 1000 for r in cumulative_rewards],
            width,
            label='Cumulative Reward (÷1000)',
            alpha=0.8)

    plt.xlabel('Agent Type')
    plt.ylabel('Performance')
    plt.title('Performance Summary')
    plt.xticks(x, agents)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('agents_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    logger.info("Agent comparison plots saved as 'agents_comparison.png'")


# Run the comparison
comparison_results, agent_directories = compare_agents()

# Create comparison visualization
create_comparison_plot(comparison_results)

# Still display the latest results from monitoring
display_results()
