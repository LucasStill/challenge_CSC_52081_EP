# Student Gym Environment Challenge

A simplified gym environment for educational reinforcement learning challenges. This package provides students with a standard gym interface while hiding internal implementation details.

## Quick Start

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd challenge_CSC_52081_EP

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the student client package
pip install -e .
```

### Basic Usage

```python
from student_client import create_student_gym_env

# SIMPLEST USAGE: Just call with no parameters!
# It automatically loads from .env file or uses sensible defaults
env = create_student_gym_env(user_token='student_token')

# Use standard gym interface
obs, info = env.reset()

for step in range(100):
    # Choose action (0=do nothing, 1=repair, 2=sell)
    action = env.action_space.sample()
    
    # Take step in environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print(f"Episode terminated at step {step}")
        break

# Clean up
env.close()
```


## Features

### Standard Gym Interface

The student client provides a familiar gym interface:

- `env.reset()` - Reset environment to initial state
- `env.step(action)` - Take a step in the environment
- `env.close()` - Clean up the environment

### Observation Space

- **Type**: Continuous
- **Shape**: `(9,)`
- **Content**: 7 sensor measurements + 1 flight phase + 1 timestep
- **Sensors**: HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin, Fuel_flow, HPC_Pout_st, LP_Nmech, phase_type, timestep

### Action Space

- **Type**: Discrete(3)
- **Actions**:
  - `0`: Do nothing (continue operation)
  - `1`: Repair (reduce degradation, with cost)
  - `2`: Sell (terminate episode, get sale reward)

## Using .env File (Automatic)

The environment automatically loads from `.env` file if present. Just create a `.env` file in your project root:

```env
# Server configuration
SERVER_URL=http://rlchallenge.orailix.com
USER_TOKEN=student_token

# Environment settings
ENV_TYPE=DegradationEnv
MAX_STEPS_PER_EPISODE=700
AUTO_RESET=True
TIMEOUT=30.0
PROD=True
```

Then simply call:

```python
from student_client import create_student_gym_env

# This automatically loads from .env file
env = create_student_gym_env()
```

If no `.env` file is found, it uses sensible defaults and shows a helpful warning message.

## Requirements

### Dependencies

The project requires the following Python packages:

```
numpy>=1.21.0
gymnasium>=0.26.0
httpx>=0.23.0
pydantic>=1.9.0
python-dotenv>=0.19.0
```

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy gymnasium httpx pydantic python-dotenv
```

## Examples

### Simple Random Policy

```python
from student_client import create_student_gym_env

def run_random_policy():
    """Run a simple random policy for demonstration"""
    
    env = create_student_gym_env(
        server_url='http://localhost:8001',
        user_token='student_user'
    )
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: Reward={reward:.2f}, Total={total_reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    return total_reward

if __name__ == "__main__":
    run_random_policy()
```

### Training Loop

```python
from student_client import create_student_gym_env

def train_agent(num_episodes=10):
    """Simple training loop example"""
    
    for episode in range(num_episodes):
        env = create_student_gym_env(
            server_url='http://rlchallenge.orailix.com',
            user_token='student_user'
        )
        
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(100):
            # Random action for demonstration
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: Total reward = {total_reward:.2f}")
        env.close()

train_agent()
```



## License

This project is for educational purposes only.

## Support

For questions or issues, please contact your instructor or teaching assistant.

---

**Version**: 1.0.0
**Last Updated**: 2026
