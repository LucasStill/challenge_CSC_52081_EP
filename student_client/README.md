# 🎓 Student Client for RL Simulator

A simplified gym environment interface designed for educational purposes. This package provides students with a standard gym interface without exposing internal implementation details.

## 🚀 Quick Start

```python
from student_client import create_student_gym_env

# Create environment
env = create_student_gym_env(
    user_token='student_user'
)

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

## 📋 Features

### Standard Gym Interface

The student client provides a familiar gym interface:

```python
# Create environment
env = create_student_gym_env(...)

# Reset environment
obs, info = env.reset()

# Step through environment
obs, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

### Observation Space

- **Type**: Continuous
- **Shape**: `(9,)`
- **Content**: 7 sensor measurements + 1 flight phase + 1 timestep
- **Sensors**: HPC_Tout, HP_Nmech, HPC_Tin, LPT_Tin, Fuel_flow, HPC_Pout_st, LP_Nmech, phase_type, timestep
- **Format**: Standard numpy array matching admin environment observations

### Action Space

- **Type**: Discrete(3)
- **Actions**:
  - `0`: Do nothing (continue operation)
  - `1`: Repair (reduce degradation, with cost)
  - `2`: Sell (terminate episode, get sale reward)

### Info Dictionary

The `info` dictionary contains useful information:

```python
{
    'step': int,                # Current step number
    'episode_id': str,          # Unique episode identifier
    'total_reward': float,      # Cumulative reward
    # Additional environment-specific info...
}
```


## 🔧 Requirements

### Prerequisites

- Python 3.8+
- Gym server running (`python -m backend.gym_server`)
- Redis server running (`redis-server`)

### Installation

```bash
# Install the student client package
pip install -e .

# Or install from source
python -m pip install -e "student_client"
```

## 📊 Understanding the Environment

### Observation Space

The observation space contains 9 sensor measurements:

| Index | Sensor      | Description                                       |
|-------|-------------|---------------------------------------------------|
| 0 | HPC_Tout    | High Pressure Compressor Temperature Outlet       |
| 1 | HP_Nmech    | High Pressure Shaft Mechanical Speed              |
| 2 | HPC_Tin     | High Pressure Compressor Temperature Inlet        |
| 3 | LPT_Tin     | Low Pressure Turbine Temperature Inlet            |
| 4 | Fuel_flow   | Fuel Flow Rate                                    |
| 5 | HPC_Pout_st | High Pressure Compressor Pressure Outlet (static) |
| 6 | LP_Nmech    | Low Pressure Shaft Mechanical Speed               |
| 7 | phase_type  | Flight Phase Type                                 |
| 8 | DTAMB       | Ambiant Temperature Deviation                     |

### Action Space

Three actions are available:

| Action | Description | Effect |
|--------|-------------|--------|
| 0 | Do Nothing | Continue normal operation |
| 1 | Repair | Reduce degradation (with cost) |
| 2 | Sell | Terminate episode, get sale reward |

### Reward System

The reward system encourages:
- ✅ Long operation with low degradation
- ✅ Strategic maintenance actions
- ❌ Avoiding failures and high degradation

### Contact

For any question, email: thil (at) lix dot polytechnique.fr