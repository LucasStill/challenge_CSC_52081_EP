"""
Student Gym Environment

A simplified gym environment for educational purposes.
Provides standard gym interface without exposing internal implementation details.
"""

import logging
import os
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StudentGymEnvConfig(BaseModel):
    """Configuration for student gym environment"""
    server_url: str = "http://localhost:8001"
    user_token: str
    env_type: str = "DegradationEnv"
    max_steps_per_episode: int = 1000
    auto_reset: bool = True
    timeout: float = 30.0
    prod: bool = True  # Production mode: hide internal information
    step_size: int = 10 # Number of simulation steps to compute per environment step

# Client version
CLIENT_VERSION = "0.1"

class StudentGymEnv(gym.Env):
    """
    Student Gym Environment
    
    A simplified gym environment that provides standard gym interface
    while hiding internal implementation details.
    
    This environment is designed for educational purposes and provides:
    - Standard gym interface (reset, step, close, render)
    - Observation space: 9 dimensions (7 sensors + 1 phase + 1 timestep)
    - Action space: 3 actions (0=do nothing, 1=repair, 2=sell)
    - Basic episode information without internal details
    
    Students can use this environment for reinforcement learning without
    needing to understand the underlying simulation complexity.
    
    The environment follows the classic gym specification:
    - reset() returns: (observation, info)
    - step() returns: (observation, reward, terminated, truncated, info)
    - All observations are numpy arrays of shape (9,)
    - All actions are integers (0, 1, or 2)
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(
        self,
        config: StudentGymEnvConfig,
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize the student gym environment.
        
        Args:
            config: Configuration for the environment
            episode_id: Optional existing episode ID to restore
            session_id: Optional existing session ID
        """
        super().__init__()
        
        self.config = config
        self.server_url = config.server_url.rstrip('/')
        self.user_token = config.user_token
        self.episode_id = episode_id
        self.session_id = session_id
        self.auto_reset = config.auto_reset
        self.prod = config.prod  # Store production mode setting
        
        # Configure httpx logging to suppress verbose HTTP request logs
        import logging as httpx_logging
        httpx_logger = httpx_logging.getLogger("httpx")
        httpx_logger.setLevel(httpx_logging.WARNING)
        
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
        
        # Initialize session and episode
        self._initialize_session()
        self._initialize_episode()
        
        # Set up observation and action spaces
        self._setup_spaces()
        
        # Track episode state
        self.current_step = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        
        logger.info(f"StudentGymEnv initialized with episode {self.episode_id}")

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

    def _initialize_episode(self):
        """Initialize or restore an episode"""
        if not self.episode_id:
            # Create new episode
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

                self.episode_id = episode_data['episode_id']
                self.current_observation = np.array(episode_data['initial_observation'], dtype=np.float32)
                self.current_step = 0
                self.terminated = False
                self.truncated = False

                logger.info(f"Created new episode: {self.episode_id}")
            except Exception as e:
                logger.error(f"Failed to create episode: {e}")
                raise RuntimeError(f"Could not create episode: {str(e)}")
        else:
            # Restore existing episode
            try:
                response = self.client.get(f"/api/v1/episode/{self.episode_id}")
                response.raise_for_status()
                episode_info = response.json()

                # Get the latest state
                response = self.client.get(f"/api/v1/episode/{self.episode_id}/state/latest")
                response.raise_for_status()
                state_data = response.json()

                self.current_observation = np.array(state_data['observation'], dtype=np.float32)
                self.current_step = state_data['step']
                self.terminated = state_data['terminated']
                self.truncated = state_data['truncated']
                self.total_reward = episode_info.get('total_reward', 0.0)

                logger.info(f"Restored episode {self.episode_id} at step {self.current_step}")
            except Exception as e:
                logger.error(f"Failed to restore episode {self.episode_id}: {e}")
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
        internal_fields = [ 'terminated', 'truncated']
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

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Current observation (sensor measurements)
            info: Additional information dictionary
        """
        try:
            reset_data = {'episode_id': self.episode_id}
            if seed is not None:
                reset_data['seed'] = seed

            response = self.client.post("/api/v1/episode/reset", json=reset_data)
            response.raise_for_status()

            reset_info = response.json()

            # Update local state
            self.current_observation = np.array(reset_info['observation'], dtype=np.float32)
            self.current_step = 0
            self.terminated = False
            self.truncated = False
            self.total_reward = 0.0

            logger.info(f"Episode {self.episode_id} reset successfully")

            # Filter info based on production mode
            reset_info = {
                'step': self.current_step,
                'episode_id': self.episode_id
            }

            return self.current_observation, self._filter_info_dict(reset_info)

        except Exception as e:
            logger.error(f"Failed to reset episode {self.episode_id}: {e}")
            raise RuntimeError(f"Could not reset episode: {str(e)}")

    def step(self, action: int, step_size: Optional[int] = 10, return_all_states: bool = True) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (0=do nothing, 1=repair, 2=sell)
            step_size: Number of steps to execute (default: config.step_size, max: 50)
            return_all_states: If True, returns list of all observations for the steps

        Returns:
            observation: 
                - Single observation (np.ndarray) if return_all_states=False
                - List of observations if return_all_states=True
            reward: Total reward for all steps executed
            terminated: Whether episode is terminated
            truncated: Whether episode was truncated
            info: Additional information including step_size and return_all_states

        Note:
            - step_size is guaranteed to be <= 50 for performance
            - When return_all_states=True, observations are unrolled into a list
            - Each observation in the list has shape (9,)
        """
        try:
            if self.terminated or self.truncated:
                if self.auto_reset:
                    return self.reset()
                else:
                    return (
                        self.current_observation,
                        0.0,
                        True,
                        False,
                        {'message': 'Episode already terminated'}
                    )

             # Send step request with effective step_size and return_all_states
            effective_step_size = step_size if step_size is not None else self.config.step_size
            # Guarantee: step_size cannot be larger than 50
            effective_step_size = min(effective_step_size, 50)
            step_data = {
                'episode_id': self.episode_id,
                'action': int(action),
                'step_size': effective_step_size,
                'return_all_states': return_all_states
            }

            response = self.client.post("/api/v1/episode/step", json=step_data)
            response.raise_for_status()

            response_payload = response.json()
            observation = response_payload['observation']
            step_info = response_payload['info']

            # Handle observation unrolling when return_all_states is True
            if return_all_states and isinstance(observation, list):
                # Check if observation is a flat list of sensor values or a list of observation arrays
                if len(observation) > 0 and isinstance(observation[0], (int, float)):
                    # Server returned flat list of sensor values - need to reshape
                    # Each observation should have 9 sensor values
                    num_observations = len(observation) // 9
                    observations_list = []
                    for i in range(num_observations):
                        start_idx = i * 9
                        end_idx = start_idx + 9
                        obs_array = np.array(observation[start_idx:end_idx], dtype=np.float32)
                        observations_list.append(obs_array)
                else:
                    # Server returned list of observation arrays
                    observations_list = [np.array(obs, dtype=np.float32) for obs in observation]
                
                final_observation = observations_list
                # Update current observation to the last state
                self.current_observation = observations_list[-1] if observations_list else self.current_observation
            else:
                # Single observation - convert to numpy array
                observations_list = None
                self.current_observation = np.array(observation, dtype=np.float32)
                final_observation = self.current_observation

            reward = float(response_payload['reward'])
            self.terminated = response_payload['terminated']
            self.truncated = response_payload['truncated']
            self.current_step = response_payload['step']
            self.total_reward += reward

            if self.terminated:
                print(f'Episode {self.episode_id} reached termination state, reason: {step_info["reason"]}')

            logger.debug(f"Step {self.current_step}: reward={reward}, terminated={self.terminated}")

            # Combine step info with our local info
            combined_info = {
                'step': self.current_step,
                'episode_id': self.episode_id,
                'total_reward': self.total_reward,
                'return_all_states': return_all_states,
                'step_size': effective_step_size,
                **step_info.get('info', {})
            }

            return (
                np.array(final_observation),
                reward,
                self.terminated,
                self.truncated,
                self._filter_info_dict(combined_info)
            )

        except Exception as e:
            logger.error(f"Failed to step episode {self.episode_id}: {e}")
            # Return a terminated state on error
            return (
                self.current_observation,
                0.0,
                True,
                False,
                {'error': str(e), 'step': self.current_step}
            )

    def close(self):
        """Clean up the environment"""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
            logger.info(f"Closed environment {self.episode_id}")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    def render(self, mode: str = 'human'):
        """
        Render the environment (limited functionality for remote env)
        
        Args:
            mode: Render mode ('human' for text display)
        """
        if mode == 'human':
            print(f"Environment {self.episode_id} - Step {self.current_step}")
            print(f"Current observation: {self.current_observation}")
            print(f"Total reward: {self.total_reward}")
            print(f"Status: {'Terminated' if self.terminated else 'Active'}")
        return None

    def get_episode_info(self) -> Dict:
        """
        Get information about the current episode
        
        Returns:
            Dictionary with episode information
        """
        try:
            response = self.client.get(f"/api/v1/episode/{self.episode_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get episode info: {e}")
            return {'error': str(e)}





def create_student_gym_env(
    server_url: Optional[str] = None,
    user_token: Optional[str] = None,
    env_type: Optional[str] = None,
    max_steps_per_episode: Optional[int] = None,
    auto_reset: Optional[bool] = None,
    timeout: Optional[float] = None,
    prod: bool = True,
    step_size: int = 10,
    episode_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> StudentGymEnv:
    """
    Factory function to create a student gym environment.
    
    This is the main entry point for students to create environments.
    By default, it loads configuration from .env file if no parameters are provided.
    
    Args:
        server_url: URL of the gym server (defaults to .env SERVER_URL or 'http://localhost:8001')
        user_token: User authentication token (defaults to .env USER_TOKEN or 'student_user')
        env_type: Type of environment to create (defaults to .env ENV_TYPE or 'DegradationEnv')
        max_steps_per_episode: Maximum steps before truncation (defaults to .env MAX_STEPS_PER_EPISODE or 1000)
        auto_reset: Whether to auto-reset terminated environments (defaults to .env AUTO_RESET or True)
        timeout: HTTP timeout in seconds (defaults to .env TIMEOUT or 30.0)
        prod: Production mode (True to hide internal information like degradation)
        episode_id: Optional existing episode ID to restore
        session_id: Optional existing session ID
        
    Returns:
        StudentGymEnv instance
        
    Example (Simplest - uses .env file):
        >>> from student_client import create_student_gym_env
        >>> env = create_student_gym_env()
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        
    Example (Manual configuration):
        >>> env = create_student_gym_env(
        ...     server_url='http://localhost:8001',
        ...     user_token='student_user'
        ... )
        
    Production Mode Example:
        >>> # Development mode - show all information
        >>> env = create_student_gym_env(prod=False)
        >>> 
        >>> # Production mode - hide internal information (default)
        >>> env = create_student_gym_env(prod=True)
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
    
    config = StudentGymEnvConfig(
        server_url=config_server_url,
        user_token=config_user_token,
        env_type=config_env_type,
        max_steps_per_episode=config_max_steps,
        auto_reset=config_auto_reset,
        timeout=config_timeout,
        step_size=step_size
    )
    
    return StudentGymEnv(
        config=config,
        episode_id=episode_id,
        session_id=session_id
    )