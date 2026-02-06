# Turbofan Engine Maintenance Challenge

A reinforcement learning environment for optimizing turbofan engine maintenance. Students learn to balance engine longevity with maintenance costs using a standard gym interface.

## Key Concepts

### Step Size and Observations

The environment uses a `step_size` parameter that determines how many flight observations are returned per `env.step()` call:

- **step_size=10**: Each `env.step()` advances the simulation by 10 flights. Recommended to stay fixed for speed.  
- **Returned observations**: A list of 'step_size' observation arrays, each representing sensor data from one flight
- **Goal**: Learn optimal maintenance timing based on degradation patterns across multiple flights

### Actions and Objectives

**Action Space (Discrete):**
- `0`: Do nothing (continue normal operation)
- `1`: Repair (reduce engine degradation, incurs maintenance cost)
- `2`: Sell (terminate episode, receive sale reward based on remaining engine life)

**Objective:** Maximize total reward by:
- Prolonging engine lifetime through strategic repairs
- Avoiding catastrophic failures
- Selling at optimal time for maximum profit

## Configuration

### .env File

Create a `.env` file in your project root for automatic configuration:

```env
# Server configuration
SERVER_URL=http://rlchallenge.orailix.com
USER_TOKEN=your_student_token

# Environment settings
STEP_SIZE=10 # advised to keep it fixed for speed
MAX_STEPS_PER_EPISODE=700
```

### Basic Usage

```python
from student_client import create_student_gym_env

# Automatically loads from .env file
env = create_student_gym_env()

# Reset environment
obs, info = env.reset()

# Run episode
for step in range(100):
    # Choose action based on your policy
    action = env.action_space.sample()
    
    # Get next 10 flight observations
    obs_list, reward, terminated, truncated, info = env.step(
        action=action,
        step_size=10,
        return_all_states=True
    )
    
    if terminated or truncated:
        break

# Clean up
env.close()
```

## Examples

### Simple Example

See `example/simple_example.py` for a complete working example with:
- Environment setup
- Action policy implementation
- Data collection and analysis
- Visualization of results

### Single Trajectory Analysis

For detailed trajectory analysis, see `example/single_trajectory.ipynb` which includes:
- Step-by-step environment interaction
- Observation analysis
- Action timing visualization
- Reward optimization strategies


## Installation

```bash
# Clone repository
git clone <repository-url>
cd challenge_CSC_52081_EP

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Observation Space

Each observation contains 9 sensor measurements:
- HPC_Tout: High Pressure Compressor Temperature Outlet
- HP_Nmech: High Pressure Shaft Mechanical Speed
- HPC_Tin: High Pressure Compressor Temperature Inlet
- LPT_Tin: Low Pressure Turbine Temperature Inlet
- Fuel_flow: Fuel Flow Rate
- HPC_Pout_st: High Pressure Compressor Pressure Outlet
- LP_Nmech: Low Pressure Shaft Mechanical Speed
- phase_type: Flight Phase Type
- DTAMB: Temperature Difference

## Learning Resources

- Start with `simple_example.py` to understand the basic interaction pattern
- Use `single_trajectory.ipynb` for in-depth analysis of engine degradation
- Implement your own policies in the same structure
- Focus on action timing relative to degradation patterns

## Support

For technical issues, contact thil (at) lix (dot) polytechnique (dot) fr

---

**Version**: 1.0.0
**Focus**: Turbofan engine maintenance optimization