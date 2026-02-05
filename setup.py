#!/usr/bin/env python3
"""
Setup script for Student Gym Environment package
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='student_gym_env',
    version='1.0.0',
    description='Student Gym Environment for Reinforcement Learning Challenges',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='RL Challenge Team',
    author_email='',
    url='',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='reinforcement-learning gym environment education',
    project_urls={
        'Source': '',
        'Documentation': '',
    },
    include_package_data=True,
    package_data={
        '': ['*.env', '*.md', '*.txt'],
    },
    entry_points={},
)