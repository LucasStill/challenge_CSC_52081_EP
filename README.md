# 🎓 Student Gym Environment Challenge

A simplified gym environment for educational reinforcement learning challenges. This package provides students with a standard gym interface while hiding internal implementation details.

## 🚀 Quick Start

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

# 🚀 SIMPLEST USAGE: Just call with no parameters!
# It automatically loads from .env file or uses sensible defaults
env = create_student_gym_env()

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

### Manual Configuration (if needed)

```python
from student_client import create_student_gym_env

# Create environment with explicit parameters
env = create_student_gym_env(
    server_url='http://localhost:8001',
    user_token='student_user',
    max_steps_per_episode=200
)
```

## 📋 Features

### Standard Gym Interface

The student client provides a familiar gym interface:

- `env.reset()` - Reset environment to initial state
- `env.step(action)` - Take a step in the environment
- `env.close()` - Clean up the environment
- `env.render()` - Display environment state

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

## 📖 Configuration

### Environment Configuration

```python
create_student_gym_env(
    server_url='http://localhost:8001',  # Gym server URL
    user_token='student_user',          # Authentication token
    env_type='DegradationEnv',         # Environment type
    max_steps_per_episode=1000,        # Maximum steps per episode
    auto_reset=True,                   # Auto-reset on termination
    timeout=30.0                       # HTTP timeout in seconds
)
```

### Using .env File (Automatic)

The environment automatically loads from `.env` file if present. Just create a `.env` file in your project root:

```env
# Server configuration
SERVER_URL=http://localhost:8001
USER_TOKEN=student_user

# Environment settings
ENV_TYPE=DegradationEnv
MAX_STEPS_PER_EPISODE=1000
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

## 🔧 Requirements

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

## 📊 Examples

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
            server_url='http://localhost:8001',
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

## 🎯 Project Structure

```
challenge_CSC_52081_EP/
├── .env                    # Environment configuration
├── .gitignore              # Git ignore patterns
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── student_client/         # Main package
│   ├── __init__.py         # Package initialization
│   ├── student_gym_env.py  # Gym environment implementation
│   └── README.md           # Package-specific documentation
├── example/                # Example scripts and notebooks
│   └── single_trajectory.ipynb
└── main.py                 # Main entry point
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Connection refused | Check if gym server is running and URL is correct |
| Authentication failed | Verify user token matches server expectations |
| Module not found | Install dependencies with `pip install -r requirements.txt` |
| Action invalid | Use actions 0, 1, or 2 only |

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('student_client').setLevel(logging.DEBUG)

# Check environment variables
import os
print(f"SERVER_URL: {os.getenv('SERVER_URL')}")
print(f"USER_TOKEN: {os.getenv('USER_TOKEN')}")
```

## 📚 Additional Resources

- **Gym Documentation**: https://gymnasium.farama.org/
- **Reinforcement Learning**: https://spinningup.openai.com/
- **Python Documentation**: https://docs.python.org/3/

## 🎯 Best Practices

### Environment Management

```python
# Always close environments
env.close()

# Use context managers
with create_student_gym_env(...) as env:
    obs, info = env.reset()
    # ... use environment ...
```

### Error Handling

```python
try:
    env = create_student_gym_env(...)
    obs, info = env.reset()
    # ... use environment ...
except Exception as e:
    print(f"Error: {e}")
    # Handle error gracefully
finally:
    env.close()
```

## 📝 License

This project is for educational purposes only.

## 🤝 Support

For questions or issues, please contact your instructor or teaching assistant.

---

**Version**: 1.0.0
**Last Updated**: 2024
**Maintainer**: RL Challenge Team