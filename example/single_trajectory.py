from student_client import create_student_gym_env


def main():

    step_size = 10
    env = create_student_gym_env(step_size=step_size)

    # Reset environment to get initial observation
    obs, info = env.reset()
    print(f"📋 Starting episode {info.get('episode_id', 'unknown')}")

    # Initialize data collection arrays
    observations = []
    actions = []
    rewards = []
    total_timesteps = 0

    for step in range(50):

        # Choose a random action (0=do nothing, 1=repair, 2=sell)
        action = env.action_space.sample()
        print(action)

        # Take step in environment
        obs_result, reward, terminated, truncated, info = env.step(
            action=action
        )

        observations.append(obs_result)
        actions.append(action)

        rewards.append(reward)

        # Update total timesteps - server advances by step_size and returns all observations
        total_timesteps += step_size

        # Print progress every step
        if step % 1 == 0:
            print(f" Step {total_timesteps}: Reward={reward:.2f}, Total={sum(rewards):.2f}")

        # Check if episode ended
        if terminated or truncated:
            print(f"🏁 Episode ended at step {total_timesteps} with reward={reward:.2f}")
            break

    # Print summary statistics
    total_reward = sum(rewards)
    print(f"\n Episode Summary:")
    print(f"   Total Steps: {len(actions)}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(
        f"   Actions Taken: {len([a for a in actions if a == 1])} repairs, {len([a for a in actions if a == 2])} sell")

    # Finish episode
    env.close()

if __name__ == "__main__":
    main()