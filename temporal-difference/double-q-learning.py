import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.environments import GridworldWithObstacles

# TODO: Implement the double q learning agent

env = GridworldWithObstacles()
env.render()

