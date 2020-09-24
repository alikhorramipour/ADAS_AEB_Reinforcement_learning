# AEB implementation using reinforcement learning

A simple implementation of AEB braking using OpenAI Gym

**ADAS:** Advanced driver-assistance systems

**AEB:** Autonomous emergency braking

## Agent and Environment
Starting with 20m/s, ego car's radar detects an object 150-200m in front of it and has to decide when is the best time to activate the braking system.

## Rewarding
The agent gets -0.5 if it stops too far from the object, -1.0 if collision occurs and if it stops close to the object, it get's a reward proportional to the distance (![equation](https://latex.codecogs.com/svg.latex?e^{-x})).
