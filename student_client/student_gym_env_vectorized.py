"""
Student Gym Environment - Vectorized Version

A vectorized gym environment for educational purposes that provides
standard gym vectorized interface while hiding internal implementation details.

This environment allows students to work with multiple parallel environments
through a single interface, similar to how they would use gymnasium.vector
environments.
"""

import logging
import os
from typing import Dict, Tuple, Optional, Any, List, Union
import numpy as np
import gymnasium as gym
from dotenv import load_dotenv
from gymnasium import spaces
import httpx
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StudentGymEnvVectorizedConfig(BaseModel):
    """Configuration for student gym vectorized environment"""
    server_url: str = "http://localhost:8001"
    user_token: str
    env_type: str = "DegradationEnv"
    num_envs: int = 4  # Number of parallel environments
    max_steps_per_episode: int = 1000
    auto_reset: bool = True
    timeout: float = 30.0
    prod: bool = True  # Production mode: hide internal information
    step_size: int = 10  # Number of simulation steps to compute per environment step
    return_all_states: bool = True  # Return observations for all steps in step_size

CLIENT_VERSION = "0.1"

class StudentGymEnvVectorized(gym.Env):
    """
    Student Gym Environment - Vectorized Version

    A vectorized gym environment that provides standard gym vectorized interface
    while hiding internal implementation details.

    This environment is designed for educational purposes and provides:
    - Standard gym vectorized interface (reset, step, close)
    - Multiple parallel environments managed through a single interface
    - Observation space: 9 dimensions (7 sensors + 1 phase + 1 timestep)
    - Action space: 3 actions (0=do nothing, 1=repair, 2=sell)
    - Basic episode information without internal details

    Students can use this environment for reinforcement learning without
    needing to understand the underlying simulation complexity.

    The environment follows the gymnasium vectorized environment specification:
    - reset() returns: (observations, infos)
    - step() returns: (observations, rewards, terminateds, truncateds, infos)
    - When return_all_states=False: observations is a numpy array of shape (num_envs, 9)
    - When return_all_states=True: observations is a list of numpy arrays, where each array
      has shape (num_steps, 9) and num_steps may vary between environments
    - All actions are numpy arrays of integers (0, 1, or 2)
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
            self,
            config: StudentGymEnvVectorizedConfig,
            episode_ids: Optional[List[str]] = None,
            session_id: Optional[str] = None
    ):
        """
        Initialize the student gym vectorized environment.

        Args:
            config: Configuration for the environment
            episode_ids: Optional list of existing episode IDs to restore
            session_id: Optional existing session ID
        """
        super().__init__()

        self.config = config
        self.server_url = config.server_url.rstrip('/')
        self.user_token = config.user_token
        self.num_envs = config.num_envs
        self.auto_reset = config.auto_reset
        self.prod = config.prod  # Store production mode setting
        self.return_all_states = config.return_all_states

        # HTTP client
        self.client = httpx.Client(
            base_url=self.server_url,
            timeout=config.timeout,
            headers={
                'User-Token': self.user_token,
                'Content-Type': 'application/json'
            }
        )

        # Check for client updates
        self._check_for_updates()

        # Initialize session
        self.session_id = session_id
        self._initialize_session()

        # Initialize episodes
        self.episode_ids = episode_ids or []
        if not self.episode_ids:
            # Create new episodes
            self._initialize_episodes()
        else:
            # Restore existing episodes
            self._restore_episodes()

        # Set up observation and action spaces
        self._setup_spaces()

        # Track episode state
        self.current_step = 0
        self.total_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.terminateds = np.zeros(self.num_envs, dtype=bool)
        self.truncateds = np.zeros(self.num_envs, dtype=bool)
        self.active_envs = np.ones(self.num_envs, dtype=bool)

        logger.info(f"StudentGymEnvVectorized initialized with {self.num_envs} environments")
        logger.info(f"Episode IDs: {self.episode_ids}")

    def _check_for_updates(self):
        """Check if a newer version of the client is available"""
        try:
            response = self.client.get("/api/v1/version")
            response.raise_for_status()
            version_data = response.json()
            latest_version = version_data.get('latest_version', '0.0')

            # Compare versions
            client_major, client_minor = self._parse_version(CLIENT_VERSION)
            latest_major, latest_minor = self._parse_version(latest_version)

            # Check if update is available
            if (latest_major > client_major) or (latest_major == client_major and latest_minor > client_minor):
                update_message = (
                    f"🚀 A new version ({latest_version}) of the student client is available! "
                    f"You are currently using version {CLIENT_VERSION}. "
                    f"Please update for the best experience."
                )
                print(f"\n🔵 {update_message}\n")
                logger.info(update_message)
            elif latest_major == client_major and latest_minor == client_minor:
                logger.info(f"Client is up to date (version {CLIENT_VERSION})")
            else:
                logger.info(f"Client version {CLIENT_VERSION} is newer than latest {latest_version}")

        except httpx.HTTPStatusError as e:
            logger.debug(f"Could not check for updates: {e}")
        except Exception as e:
            logger.debug(f"Version check failed: {e}")

    def _parse_version(self, version: str) -> Tuple[int, int]:
        """Parse version string into major and minor components"""
        try:
            parts = version.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            return major, minor
        except Exception:
            return 0, 0

    def _initialize_session(self):
        """Initialize or reuse a user session"""
        if not self.session_id:
            # Create new session
            try:
                response = self.client.post("/api/v1/session/create")
                response.raise_for_status()
                session_data = response.json()
                self.session_id = session_data['session_id']
                logger.info(f"Created new session: {self.session_id}")
            except httpx.HTTPStatusError as e:
                # Extract server error message for better user feedback
                try:
                    error_detail = e.response.json().get('detail', str(e))
                    user_message = f"Session creation failed: {error_detail}"
                    logger.error(f"Failed to create session: {error_detail}")
                    raise RuntimeError(user_message)
                except Exception:
                    # Fallback if we can't parse JSON response
                    logger.error(f"Failed to create session: {e}")
                    raise RuntimeError(f"Could not create session: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                raise RuntimeError(f"Could not create session: {str(e)}")
        else:
            logger.info(f"Using existing session: {self.session_id}")

    def _initialize_episodes(self):
        """Initialize new episodes"""
        # Create episodes one by one
        for i in range(self.num_envs):
            env_config = {
                'env_type': self.config.env_type,
                'max_steps_per_episode': self.config.max_steps_per_episode,
                'auto_reset': self.config.auto_reset,
                'step_size': self.config.step_size
            }

            try:
                response = self.client.post(
                    "/api/v1/episode/create",
                    json=env_config,
                    headers={'Session-ID': self.session_id}
                )
                response.raise_for_status()
                episode_data = response.json()

                self.episode_ids.append(episode_data['episode_id'])
                logger.info(f"Created new episode {i + 1}/{self.num_envs}: {episode_data['episode_id']}")

            except Exception as e:
                logger.error(f"Failed to create episode {i + 1}/{self.num_envs}: {e}")
                raise RuntimeError(f"Could not create episode: {str(e)}")

    def _restore_episodes(self):
        """Restore existing episodes"""
        if len(self.episode_ids) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} episode IDs, got {len(self.episode_ids)}")

        for i, episode_id in enumerate(self.episode_ids):
            try:
                # Verify episode exists
                response = self.client.get(f"/api/v1/episode/{episode_id}")
                response.raise_for_status()

                # Get the latest state
                response = self.client.get(f"/api/v1/episode/{episode_id}/state/latest")
                response.raise_for_status()
                state_data = response.json()

                logger.info(f"Restored episode {i + 1}/{self.num_envs}: {episode_id} at step {state_data['step']}")

            except Exception as e:
                logger.error(f"Failed to restore episode {episode_id}: {e}")
                raise RuntimeError(f"Could not restore episode: {str(e)}")

    def _filter_info_dict(self, info: Dict) -> Dict:
        """
        Filter info dictionary based on production mode.
        In production mode, removes internal information like degradation levels.

        Args:
            info: Original info dictionary from server

        Returns:
            Filtered info dictionary safe for student use
        """
        if not self.prod:
            # Development mode: return all information
            return info

        # Production mode: filter out internal information
        filtered_info = {}

        # Always include these safe fields
        safe_fields = ['step', 'episode_id', 'total_reward', 'message', 'error']
        for field in safe_fields:
            if field in info:
                filtered_info[field] = info[field]

        # Remove sensitive internal fields
        internal_fields = ['degradation', 'max_degradation']  # , 'terminated', 'truncated']
        for field in internal_fields:
            if field in info:
                del info[field]

        # Add back essential status fields with safe defaults
        filtered_info['terminated'] = info.get('terminated', False)
        filtered_info['truncated'] = info.get('truncated', False)

        return filtered_info

    def _setup_spaces(self):
        """Set up observation and action spaces"""
        # Observation space: 9 dimensions (7 sensors + 1 phase + 1 timestep)
        # This matches the standard observation format from the admin environment
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )

        # Action space: 3 discrete actions (0=do nothing, 1=repair, 2=sell)
        self.action_space = spaces.Discrete(3)

        # For vectorized environments, we also define the single spaces
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Reset all environments to initial state.

        Returns:
            observations: Array of current observations for all environments
            infos: List of info dictionaries for each environment
        """
        try:
            # Prepare seeds for each environment
            if seed is not None:
                seeds = [seed + i for i in range(self.num_envs)]
            else:
                seeds = [None] * self.num_envs

            # Use vectorized reset endpoint
            reset_data = {
                'episode_ids': self.episode_ids,
                'seeds': seeds
            }

            response = self.client.post("/api/v1/episode/vectorized_reset", json=reset_data)
            response.raise_for_status()

            reset_info = response.json()

            # Process results
            observations = []
            infos = []

            for i, (obs, info) in enumerate(zip(reset_info['observations'], reset_info['infos'])):
                if obs:  # Check if observation is not empty
                    observations.append(np.array(obs, dtype=np.float32))
                    self.active_envs[i] = True
                    self.terminateds[i] = False
                    self.truncateds[i] = False
                    self.total_rewards[i] = 0.0
                else:
                    # Failed to reset this environment
                    observations.append(np.zeros(self.observation_space.shape[0], dtype=np.float32))
                    self.active_envs[i] = False
                    self.terminateds[i] = True
                    self.truncateds[i] = False

                # Filter info based on production mode
                filtered_info = self._filter_info_dict(info)
                infos.append(filtered_info)

            # Convert to numpy array
            observations_array = np.array(observations, dtype=np.float32)

            logger.info(f"All {self.num_envs} environments reset successfully")

            return observations_array, infos

        except Exception as e:
            logger.error(f"Failed to reset environments: {e}")
            raise RuntimeError(f"Could not reset environments: {str(e)}")

    def step(self, actions: np.ndarray, step_size: Optional[int] = None, return_all_states: Optional[bool] = True) -> \
    Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Take a step in all environments.

        Args:
            actions: Array of actions to take for each environment
            step_size: Optional override for step_size (uses configured step_size if None)
            return_all_states: Optional override for return_all_states

        Returns:
            observations: Array of new observations for all environments (or list of arrays if return_all_states=True)
            rewards: Array of rewards for each environment
            terminateds: Array of terminated flags for each environment
            truncateds: Array of truncated flags for each environment
            infos: List of info dictionaries for each environment

        Note:
            When return_all_states=True, observations is a list where each element is a 2D array
            of shape (num_steps, 9) for that environment. Different environments may have
            different numbers of steps due to early termination.
        """
        try:
            # Validate actions
            if len(actions) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")

            # Prepare step data for each environment
            effective_step_size = step_size if step_size is not None else self.config.step_size
            effective_return_all_states = return_all_states if return_all_states is not None else self.return_all_states

            episode_actions = []
            for i, (episode_id, action) in enumerate(zip(self.episode_ids, actions)):
                # Skip terminated environments if not auto-resetting
                if self.terminateds[i] and not self.auto_reset:
                    continue

                episode_actions.append({
                    'episode_id': episode_id,
                    'action': int(action),
                    'step_size': effective_step_size,
                    'return_all_states': effective_return_all_states
                })

            # Use vectorized step endpoint
            step_data = {'episode_actions': episode_actions}

            response = self.client.post("/api/v1/episode/vectorized_step", json=step_data)
            response.raise_for_status()

            response_payload = response.json()

            # Process results
            observations = []
            rewards = []
            terminateds = []
            truncateds = []
            infos = []

            # Create mapping from episode_id to index
            episode_to_index = {episode_id: i for i, episode_id in enumerate(self.episode_ids)}

            for i, (obs, reward, terminated, truncated, info) in enumerate(zip(
                    response_payload['observations'],
                    response_payload['rewards'],
                    response_payload['terminateds'],
                    response_payload['truncateds'],
                    response_payload['infos']
            )):
                episode_id = episode_actions[i]['episode_id']
                env_idx = episode_to_index[episode_id]

                if obs:  # Check if observation is not empty
                    if effective_return_all_states and isinstance(obs[0], list):
                        # Multiple observations per environment - convert to 2D array
                        obs_array = np.array(obs, dtype=np.float32)
                        observations.append(obs_array)
                    elif effective_return_all_states and isinstance(obs, list) and len(obs) > 0 and isinstance(obs[0],
                                                                                                               (int,
                                                                                                                float)):
                        # Single observation but we expected multiple - convert to 2D array with single row
                        obs_array = np.array([obs], dtype=np.float32)
                        observations.append(obs_array)
                    else:
                        # Single observation per environment
                        observations.append(np.array(obs, dtype=np.float32))

                    self.active_envs[env_idx] = not (terminated or truncated)
                else:
                    # Failed or terminated environment
                    if effective_return_all_states:
                        # Return empty array for terminated environments when return_all_states=True
                        observations.append(np.array([], dtype=np.float32).reshape(0, self.observation_space.shape[0]))
                    else:
                        observations.append(np.zeros(self.observation_space.shape[0], dtype=np.float32))
                    self.active_envs[env_idx] = False

                rewards.append(float(reward))
                terminateds.append(terminated)
                truncateds.append(truncated)

                # Update local state
                self.terminateds[env_idx] = terminated
                self.truncateds[env_idx] = truncated
                self.total_rewards[env_idx] += float(reward)

                # Filter info based on production mode
                filtered_info = self._filter_info_dict(info)
                infos.append(filtered_info)

            # Convert to numpy arrays
            # Handle mixed observation shapes when return_all_states=True
            if effective_return_all_states:
                # For return_all_states=True, we need to handle variable-length observations
                # Create a list to hold all observations in consistent format
                final_observations = []
                for obs in observations:
                    if obs.ndim == 1:
                        # Single observation - wrap in array
                        final_observations.append(obs.reshape(1, -1))
                    elif obs.ndim == 2 and obs.shape[0] == 0:
                        # Empty observation (terminated env) - keep as empty 2D array
                        final_observations.append(obs)
                    elif obs.ndim == 2:
                        # Multiple observations - use as is
                        final_observations.append(obs)
                    else:
                        # Unexpected shape - convert to 2D
                        final_observations.append(obs.reshape(-1, self.observation_space.shape[0]))

                # observations_array will be a list of arrays with potentially different lengths
                observations_array = final_observations
            else:
                # For return_all_states=False, all observations should be 1D
                observations_array = np.array(observations, dtype=np.float32)

            rewards_array = np.array(rewards, dtype=np.float32)
            terminateds_array = np.array(terminateds, dtype=bool)
            truncateds_array = np.array(truncateds, dtype=bool)

            logger.debug(f"Step completed: rewards={rewards_array}, terminateds={terminateds_array}")

            return (
                observations_array,
                rewards_array,
                terminateds_array,
                truncateds_array,
                infos
            )

        except Exception as e:
            logger.error(f"Failed to step environments: {e}")
            # Return terminated state for all environments on error
            observations_array = np.zeros((self.num_envs, self.observation_space.shape[0]), dtype=np.float32)
            rewards_array = np.zeros(self.num_envs, dtype=np.float32)
            terminateds_array = np.ones(self.num_envs, dtype=bool)
            truncateds_array = np.zeros(self.num_envs, dtype=bool)
            infos = [{'error': str(e)} for _ in range(self.num_envs)]

            return (
                observations_array,
                rewards_array,
                terminateds_array,
                truncateds_array,
                infos
            )

    def close(self):
        """Clean up the environment"""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
            logger.info(f"Closed student vectorized environment with {self.num_envs} environments")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    def render(self, mode: str = 'human'):
        """
        Render the environment (limited functionality for remote env)

        Args:
            mode: Render mode ('human' for text display)
        """
        if mode == 'human':
            print(f"Student Vectorized Environment - {self.num_envs} environments")
            print(f"Active environments: {np.sum(self.active_envs)}/{self.num_envs}")
            print(f"Total rewards: {self.total_rewards}")
            print(f"Status: {np.sum(self.terminateds)} terminated, {np.sum(self.truncateds)} truncated")
        return None

    def get_episode_info(self) -> List[Dict]:
        """
        Get information about all episodes

        Returns:
            List of dictionaries with episode information
        """
        try:
            episode_infos = []
            for episode_id in self.episode_ids:
                response = self.client.get(f"/api/v1/episode/{episode_id}")
                response.raise_for_status()
                episode_infos.append(response.json())
            return episode_infos
        except Exception as e:
            logger.error(f"Failed to get episode info: {e}")
            return [{'error': str(e)} for _ in range(self.num_envs)]

    def get_active_count(self) -> int:
        """Get number of active (non-terminated) environments."""
        return np.sum(self.active_envs)

    def get_terminated_env_indices(self) -> List[int]:
        """
        Get indices of all currently terminated environments.

        Returns:
            List of indices for terminated environments
        """
        return [i for i, terminated in enumerate(self.terminateds) if terminated]

    def reset_specific_envs(self, env_indices: List[int], seeds: Optional[List[Optional[int]]] = None) -> Tuple[
        np.ndarray, List[Dict]]:
        """
        Manually reset specific environments.

        Args:
            env_indices: List of environment indices to reset
            seeds: Optional list of seeds for each environment (None for random)

        Returns:
            Tuple of (observations, infos) for the reset environments
        """
        if seeds is None:
            seeds = [None] * len(env_indices)

        # Validate inputs
        if len(env_indices) != len(seeds):
            raise ValueError("env_indices and seeds must have the same length")

        # Get episode IDs for the specified environments
        episode_ids = [self.episode_ids[i] for i in env_indices]

        # Use vectorized reset endpoint
        reset_data = {
            'episode_ids': episode_ids,
            'seeds': seeds
        }

        try:
            response = self.client.post("/api/v1/episode/vectorized_reset", json=reset_data)
            response.raise_for_status()

            reset_info = response.json()

            # Process results
            observations = []
            infos = []

            for i, (obs, info) in enumerate(zip(reset_info['observations'], reset_info['infos'])):
                env_idx = env_indices[i]

                if obs:  # Check if observation is not empty
                    observations.append(np.array(obs, dtype=np.float32))
                    self.active_envs[env_idx] = True
                    self.terminateds[env_idx] = False
                    self.truncateds[env_idx] = False
                    self.total_rewards[env_idx] = 0.0
                else:
                    # Failed to reset this environment
                    observations.append(np.zeros(self.observation_space.shape[0], dtype=np.float32))
                    self.active_envs[env_idx] = False
                    self.terminateds[env_idx] = True
                    self.truncateds[env_idx] = False

                # Filter info based on production mode
                filtered_info = self._filter_info_dict(info)
                infos.append(filtered_info)

            # Convert to numpy array
            observations_array = np.array(observations, dtype=np.float32)

            return observations_array, infos

        except Exception as e:
            logger.error(f"Failed to reset specific environments: {e}")
            raise RuntimeError(f"Could not reset environments: {str(e)}")


def create_student_gym_env_vectorized(
        server_url: Optional[str] = None,
        user_token: Optional[str] = None,
        env_type: Optional[str] = None,
        num_envs: int = 4,
        max_steps_per_episode: int = 1000,
        auto_reset: bool = True,
        timeout: float = 30.0,
        prod: bool = True,
        step_size: int = 1,
        return_all_states: bool = True,
        episode_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None
) -> StudentGymEnvVectorized:
    """
    Factory function to create a student gym vectorized environment.

    This is the main entry point for students to create vectorized environments.

    Args:
        server_url: URL of the gym server
        user_token: User authentication token
        env_type: Type of environment to create
        num_envs: Number of parallel environments
        max_steps_per_episode: Maximum steps before truncation
        auto_reset: Whether to auto-reset terminated environments
        timeout: HTTP timeout in seconds
        prod: Production mode (True to hide internal information like degradation)
        step_size: Number of simulation steps to compute per environment step (default: 1)
        return_all_states: Return observations for all steps in step_size (default: True)
        episode_ids: Optional list of existing episode IDs to restore
        session_id: Optional existing session ID

    Returns:
        StudentGymEnvVectorized instance

    Example:
        >>> from student_client import create_student_gym_env_vectorized
        >>> env = create_student_gym_env_vectorized(
        ...     server_url='http://rlchallenge.orailix.com',
        ...     user_token='student_user',
        ...     num_envs=4
        ... )
        >>> obs, infos = env.reset()
        >>> actions = np.array([0, 1, 0, 2])  # One action per environment
        >>> obs, rewards, terminateds, truncateds, infos = env.step(actions)

    Production Mode Example:
        >>> # Development mode - show all information
        >>> env = create_student_gym_env_vectorized(..., prod=False)
        >>>
        >>> # Production mode - hide internal information (default)
        >>> env = create_student_gym_env_vectorized(..., prod=True)

    Step Size Example:
        >>> # Use step_size=5 for better performance (5 simulation steps per environment step)
        >>> env = create_student_gym_env_vectorized(..., step_size=5)
        >>>
        >>> # Default step_size=1 (original behavior)
        >>> env = create_student_gym_env_vectorized(...)  # step_size defaults to 1
    """

    # Load environment variables from .env file
    load_dotenv()

    # Determine configuration - prioritize function parameters, then .env, then defaults
    def get_config_value(param_value, env_var_name, default_value, param_type=str):
        """Helper to get configuration value with priority: param > .env > default"""
        if param_value is not None:
            return param_value

        env_value = os.getenv(env_var_name)
        if env_value is not None:
            try:
                if param_type == bool:
                    return env_value.lower() == 'true'
                elif param_type == int:
                    return int(env_value)
                elif param_type == float:
                    return float(env_value)
                else:
                    return env_value
            except ValueError:
                logger.warning(f"Invalid {param_type.__name__} value for {env_var_name} in .env: '{env_value}'")
                return default_value

        return default_value

    # Get configuration values with proper priority
    config_server_url = get_config_value(server_url, 'SERVER_URL', 'http://rlchallenge.orailix.com', str)
    config_user_token = get_config_value(user_token, 'USER_TOKEN', 'student_user', str)
    config_env_type = get_config_value(env_type, 'ENV_TYPE', 'DegradationEnv', str)
    config_max_steps = get_config_value(max_steps_per_episode, 'MAX_STEPS_PER_EPISODE', 1000, int)
    config_auto_reset = get_config_value(auto_reset, 'AUTO_RESET', True, bool)
    config_timeout = get_config_value(timeout, 'TIMEOUT', 30.0, float)

    # Check if we're using .env values and provide guidance if missing
    if (server_url is None or user_token is None) and not os.path.exists('.env'):
        logger.warning(
            "No .env file found and no explicit parameters provided. "
            "Using default values. For better setup, create a .env file with:"
            "\nSERVER_URL=http://rlchallenge.orailix.com"
            "\nUSER_TOKEN=student_user"
            "\nENV_TYPE=DegradationEnv"
            "\nMAX_STEPS_PER_EPISODE=1000"
            "\nAUTO_RESET=True"
            "\nTIMEOUT=30.0"
        )

    max_num_envs = 4
    if num_envs > max_num_envs:
        num_envs = max_num_envs
        print(f'Warning: max number of environments is 8, setting num_envs to this value ({max_num_envs})')

    config = StudentGymEnvVectorizedConfig(
        server_url=config_server_url,
        user_token=config_user_token,
        env_type=config_env_type,
        num_envs=num_envs,
        max_steps_per_episode=config_max_steps,
        auto_reset=config_auto_reset,
        timeout=config_timeout,
        prod=prod,
        step_size=step_size,
        return_all_states=return_all_states
    )

    return StudentGymEnvVectorized(
        config=config,
        episode_ids=episode_ids,
        session_id=session_id
    )