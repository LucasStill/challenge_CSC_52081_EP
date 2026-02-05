"""
Simple Plotting Functions for Student Gym Environment

Basic plotting functions for visualizing observations and rewards.
Students can use these as inspiration to create their own visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_observations(
    observations: List[np.ndarray],
    actions: Optional[List[int]] = None,
    sensor_names: Optional[List[str]] = None,
    figsize: tuple = (15, 10),
    title: str = "Observation Dimensions Over Time"
) -> None:
    """
    Simple function to plot observation dimensions over time.
    
    This function displays each observation dimension in a separate plot (one per line)
    for easier analysis. Only Repair (1) and Sell (2) actions are shown as markers;
    Do Nothing (0) actions are hidden to keep plots clean and focused.
    
    Args:
        observations: List of observation arrays (each of shape (9,))
        actions: Optional list of actions taken at each step
        sensor_names: Optional list of names for the 9 observation dimensions
        figsize: Figure size (width, height) - note: this is now per plot
        title: Plot title - note: this is now per plot
        
    Example:
        >>> from student_client import create_student_gym_env, plot_observations
        >>> import numpy as np
        >>> 
        >>> # Collect observations manually
        >>> env = create_student_gym_env()
        >>> obs, info = env.reset()
        >>> observations = [obs]
        >>> actions = []
        >>> 
        >>> for step in range(50):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     observations.append(obs)
        ...     actions.append(action)
        ...     if terminated or truncated:
        ...         break
        >>> 
        >>> env.close()
        >>> 
        >>> # Plot the observations (shows 9 separate plots, one per dimension)
        >>> # Only Repair and Sell actions will be shown as markers
        >>> plot_observations(observations, actions)
    """
    if not observations:
        print("⚠️ No observations provided.")
        return
    
    # Default sensor names if not provided
    if sensor_names is None:
        sensor_names = [
            'HPC_Tout',      # High Pressure Compressor Temperature Outlet
            'HP_Nmech',      # High Pressure Shaft Mechanical Speed
            'HPC_Tin',       # High Pressure Compressor Temperature Inlet
            'LPT_Tin',       # Low Pressure Turbine Temperature Inlet
            'Fuel_flow',     # Fuel Flow Rate
            'HPC_Pout_st',   # High Pressure Compressor Pressure Outlet (static)
            'LP_Nmech',      # Low Pressure Shaft Mechanical Speed
            'phase_type',    # Flight Phase Type
            'DTAMB'
        ]
    
    # Convert to numpy array for easier manipulation
    observations_array = np.array(observations)
    num_steps = len(observations)
    steps = np.arange(num_steps)
    
    # Plot each observation dimension in separate figures (one per line)
    for i, sensor_name in enumerate(sensor_names):
        plt.figure(figsize=(12, 4))  # Wider, shorter figure for each dimension
        plt.plot(steps, observations_array[:, i], 'b-', linewidth=2, label='Observation')
        plt.title(f'{sensor_name} (Dimension {i})', fontsize=14, fontweight='bold')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add action markers if actions are provided (only show Repair and Sell, not Do Nothing)
        if actions is not None and len(actions) == num_steps:
            for step, action in enumerate(actions):
                # Only plot Repair (1) and Sell (2) actions, skip Do Nothing (0)
                if action in [1, 2]:
                    action_color = 'r' if action == 1 else 'g'
                    action_marker = 'o' if action == 1 else 's'
                    action_label = 'Repair' if action == 1 else 'Sell'
                    plt.scatter(step, observations_array[step, i], 
                              color=action_color, marker=action_marker, s=80, 
                              label=action_label, alpha=0.8)
        
        # Add legend (only for meaningful actions)
        if actions is not None:
            # Create custom legend handles (only for Repair and Sell)
            handles = []
            if any(a == 1 for a in actions):
                handles.append(plt.scatter([], [], color='red', marker='o', s=80, label='Repair (1)', alpha=0.8))
            if any(a == 2 for a in actions):
                handles.append(plt.scatter([], [], color='green', marker='s', s=80, label='Sell (2)', alpha=0.8))
            if handles:
                plt.legend(handles=handles, loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.show()


def plot_rewards(
    rewards: List[float],
    actions: Optional[List[int]] = None,
    figsize: tuple = (12, 6),
    title: str = "Step Rewards Over Time"
) -> None:
    """
    Simple function to plot step rewards over time.
    
    This function creates a clean visualization of individual step rewards,
    with cumulative reward shown in the legend.
    
    Args:
        rewards: List of reward values obtained at each step
        actions: Optional list of actions taken at each step
        figsize: Figure size (width, height)
        title: Plot title
        
    Example:
        >>> from student_client import create_student_gym_env, plot_rewards
        >>> 
        >>> # Collect rewards
        >>> env = create_student_gym_env()
        >>> obs, info = env.reset()
        >>> rewards = []
        >>> actions = []
        >>> 
        >>> for step in range(50):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     rewards.append(reward)
        ...     actions.append(action)
        ...     if terminated or truncated:
        ...         break
        >>> 
        >>> env.close()
        >>> 
        >>> # Plot step rewards
        >>> plot_rewards(rewards, actions)
    """
    if not rewards:
        print("⚠️ No rewards provided.")
        return
    
    steps = np.arange(len(rewards))
    cumulative_reward = np.sum(rewards)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot individual rewards as a line with markers
    plt.plot(steps, rewards, 'b-', linewidth=2, 
            marker='o', markersize=8, label=f'Step Reward')
    
    # Add action markers if provided (only Repair and Sell)
    if actions is not None and len(actions) == len(rewards):
        for step, action in enumerate(actions):
            if action == 1:  # Repair
                plt.scatter(step, rewards[step], color='red', 
                          marker='o', s=120, label='Repair',
                          edgecolor='black', linewidth=1, alpha=0.8, zorder=5)
            elif action == 2:  # Sell
                plt.scatter(step, rewards[step], color='green', 
                          marker='s', s=120, label='Sell',
                          edgecolor='black', linewidth=1, alpha=0.8, zorder=5)
    
    # Customize plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend with cumulative reward
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Remove duplicate labels and add cumulative reward info
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Add cumulative reward to legend
    cumulative_handle = plt.Line2D([], [], color='black', linestyle='none', 
                                   marker='', markersize=0, label=f'Cumulative: {cumulative_reward:.1f}')
    unique_handles.append(cumulative_handle)
    unique_labels.append(f'Cumulative: {cumulative_reward:.1f}')
    
    if unique_handles:
        plt.legend(handles=unique_handles, labels=unique_labels, 
                  loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"📊 Reward Statistics:")
    print(f"   Total Steps: {len(rewards)}")
    print(f"   Total Reward: {cumulative_reward:.2f}")
    print(f"   Average Reward: {np.mean(rewards):.2f}")
    print(f"   Max Reward: {np.max(rewards):.2f}")
    print(f"   Min Reward: {np.min(rewards):.2f}")