"""
Student Client Package

A simplified gym environment interface for students.
Hides internal implementation details while providing standard gym functionality.
"""

from .student_gym_env import create_student_gym_env, StudentGymEnv
from .plotting import plot_observations, plot_rewards

__version__ = "1.0.0"
__all__ = ['create_student_gym_env', 'StudentGymEnv', 'plot_observations', 'plot_rewards']