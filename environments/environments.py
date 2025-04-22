from enum import Enum

class GridworldAction(Enum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3


class DynamicProgrammingGridworld:
    """
    A simple gridworld environment for Dynamic Programming experiments.

    The grid is a 5x5 matrix where each cell represents a state and contains a reward value when moving
    to this state.
    The agent can move in one of four directions: NORTH, SOUTH, EAST, WEST.
    Special rewards are located at specific positions in the grid:
    - (0, 1): reward 10
    - (0, 3): reward 5

    It is much less complex than the other environments, as in dynamic programming, the agent is fully
    aware of the environments dynamics and hold all information necessary.

    Attributes:
        grid (list[list[int]]): The reward matrix for each state.
        height (int): The number of rows in the grid.
        width (int): The number of columns in the grid.
        actions (list[GridworldAction]): Available actions the agent can take.
    """
    def __init__(self):
        # Grid holds the reward to move to this field
        self.grid = [
            [0, 10, 0, 5, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        self.height = len(self.grid)
        self.width = len(self.grid)
        self.actions = list(GridworldAction)



class GridworldWithObstacles:
    """
        A 10x10 gridworld environment with static obstacles, a start position, and a goal position.

        This environment simulates a simple navigation task where an agent must reach a goal while avoiding obstacles.
        The agent can move in four directions (up, right, down, left), cannot move into obstacles or outside the grid,
        and starts at the top-left corner (0, 0) with the goal located at the bottom-right corner (9, 9).

        Attributes:
            width (int): Width of the grid.
            height (int): Height of the grid.
            start_pos (tuple[int, int]): Starting position of the agent.
            goal_pos (tuple[int, int]): Goal position to reach.
            obstacles (set[tuple[int, int]]): Set of grid positions occupied by obstacles.
            agent_pos (tuple[int, int]): Current position of the agent in the environment.

        Methods:
            reset(): Resets the agent to the start position.
            step(action): Applies an action and returns the result (next_state, reward, done, info).
            get_valid_actions(): Returns a list of valid actions from the current state.
            is_terminal(state): Returns whether a state is terminal (i.e., goal reached).
            state(): Returns the current agent position.
            render(agent_pos=None): Prints a visual representation of the grid.
        """
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.start_pos = (0, 0)
        self.goal_pos = (9, 9)

        # Define the obstacle positions (spanning multiple fields)
        self.obstacles = {(1, 3), (2, 3), (3, 3), (4, 3), (6, 1), (6, 2), (6, 3), (6, 5), (6, 6),
                          (6, 7), (6, 8), (4, 6), (5, 6), (6, 6), (8, 2), (9, 2)}

        # Current agent position
        self.agent_pos = self.start_pos

    def reset(self):
        """Resets the environment to the starting state."""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: GridworldAction):
        """
        Takes an action and returns (next_state, reward, done, info).
        Actions: 0=Up, 1=Right, 2=Down, 3=Left
        """
        y, x = self.agent_pos
        next_pos = {
            GridworldAction.NORTH: (y - 1, x),
            GridworldAction.SOUTH: (y + 1, x),
            GridworldAction.EAST: (y, x + 1), GridworldAction.WEST: (y, x - 1),
        }.get(action, (y, x))

        ny,  nx = next_pos

        valid_move = False

        # Check boundaries and obstacles
        if 0 <= ny < self.height and 0 <= nx < self.width and (ny, nx) not in self.obstacles:
            valid_move = True
            self.agent_pos = (ny, nx)

        done = self.is_terminal(self.agent_pos)
        reward = 10 if done else -0.1
        reward = reward - 2 if not valid_move else reward

        # New state is the position of the agent.
        # Reward: Only target field yields positive reward.
        # Done: If we arrive in the terminal state.
        # Information: Empty for now.
        return self.agent_pos, reward, done, {}

    def get_valid_actions(self):
        """Returns a list of valid actions for the current state. Returns all actions that will
        not result in a collision from the current position."""
        valid_actions = []
        y, x = self.agent_pos

        for action in GridworldAction:
            ny, nx = {GridworldAction.NORTH: (y - 1, x), GridworldAction.SOUTH: (y + 1, x),
                GridworldAction.WEST: (y, x - 1), GridworldAction.EAST: (y, x + 1), }[action]

            if 0 <= ny < self.height and 0 <= nx < self.width and (ny, nx) not in self.obstacles:
                valid_actions.append(action)

        return valid_actions

    def is_terminal(self, state):
        """Returns True if the given state is terminal."""
        return state == self.goal_pos

    def state(self):
        """Returns the current state (e.g., agent's position)."""
        return self.agent_pos

    def render(self, agent_pos=None):
        """Renders the gridworld with the agent and obstacles."""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # Place obstacles
        for (y, x) in self.obstacles:
            grid[y][x] = '#'

        # Place start and goal
        sy, sx = self.start_pos
        gy, gx = self.goal_pos
        grid[sy][sx] = 'S'
        grid[gy][gx] = 'G'

        # Place agent (optional)
        pos = agent_pos if agent_pos is not None else self.agent_pos
        ay, ax = pos
        if grid[ay][ax] not in ('S', 'G'):  # don't overwrite S or G
            grid[ay][ax] = 'A'

        # Print grid
        print()
        for row in grid:
            print(' '.join(row))
        print()