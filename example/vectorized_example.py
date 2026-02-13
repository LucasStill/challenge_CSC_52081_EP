import numpy as np
from student_client.student_gym_env_vectorized import create_student_gym_env_vectorized

def main():

    env = create_student_gym_env_vectorized(
        num_envs=4,
    )

    print(f"Environment created with {env.num_envs} parallel environments")
    print(f"   Episode IDs: {env.episode_ids}")

    # Reset all environments
    print(f"\n🔄 Resetting all environments...")
    observations, infos = env.reset()

    print(f"   Observations shape: {observations.shape}")
    print(f"   First observation: {observations[0]}")

    for step in range(40):

        # A) Check if any environments terminated
        terminated_envs = env.get_terminated_env_indices()
        if terminated_envs:
            print(f"   ⚠️  Environments {terminated_envs} terminated")
            reset_obs, reset_infos = env.reset_specific_envs(terminated_envs)
            for i, env_id in enumerate(terminated_envs):
                infos[env_id] = reset_infos[i]  # reset previous info dict

        # Generate random actions for all environments
        actions = np.random.randint(0, 3, size=env.num_envs)

        print(f"\n   Step {step + 1}:")
        print(f"      Actions: {actions}")

        # Take step
        observations, rewards, terminateds, truncateds, infos = env.step(actions)

        print(f"      Rewards: {rewards}")
        print(f"      Terminated: {terminateds}")
        print(f"      Active environments: {env.get_active_count()}/{env.num_envs}")

        # Show filtered info (production mode)
        for i, info in enumerate(infos):
            print(f"      Env {i} info: {info}")

    env.close()

if __name__ == "__main__":
    main()