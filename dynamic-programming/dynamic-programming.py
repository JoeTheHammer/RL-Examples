import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.environments import GridworldAction, DynamicProgrammingGridworld

action_symbols = {GridworldAction.NORTH: '↑', GridworldAction.SOUTH: '↓', GridworldAction.WEST: '←', GridworldAction.EAST: '→'}

class DynamicProgrammingAgent:
    def __init__(self, env, start_state):
        self.env = env
        self.position = start_state
        self.actions = list(GridworldAction)
        self.transition_probabilities = self.initialize_transitions()
        self.V = {state: 0.0 for state in self.transition_probabilities}
        self.policy = {}

    def render_position(self):
        for y in range(self.env.height):
            row = ""
            for x in range(self.env.width):
                if (x, y) == self.position:
                    row += " A "
                else:
                    reward = self.env.grid[y][x]
                    row += f"{reward:2d} "  # pad reward nicely
            print(row)
        print()  # blank line for spacing

    def initialize_transitions(self):
        transitions = {}

        for y in range(self.env.height):
            for x in range(self.env.width):
                state = (x, y)
                transitions[state] = {}

                for action in self.actions:
                    next_state = self._move(state, action)
                    reward = self.env.grid[y][x]
                    # Set of probabilities that we assume to be given: p(s0 , r |s, a)
                    # Needed to solve DP problems (see Sutton & Barton Book page 73)
                    # In DP, the agents knows this stuff in advance!
                    transitions[state][action] = [(next_state, 0.25, reward)]

        return transitions

    def _move(self, state, action):
        x, y = state
        if action == GridworldAction.NORTH:
            y = max(y - 1, 0)
        elif action == GridworldAction.SOUTH:
            y = min(y + 1, self.env.height - 1)
        elif action == GridworldAction.WEST:
            x = max(x - 1, 0)
        elif action == GridworldAction.EAST:
            x = min(x + 1, self.env.width - 1)
        return x, y

    def train(self):
        self.value_iteration()
        self.extract_policy()

    # Implements the value iteration algorithm that computes the optimal state-value function V.
    # For each state, it applies the Bellman optimality update by taking the maximum expected return
    # over all possible actions, based on the current estimate of V. The process repeats until the
    # value function converges (i.e., the maximum change across all states is below a small threshold φ).
    # Once V converges, an optimal policy can be derived by choosing the action that maximizes expected return in each state.
    def value_iteration(self, gamma=0.9, phi=0.0001):
        while True:
            delta = 0
            new_V = {}

            for state in self.transition_probabilities:
                old_value = self.V[state]
                # Bellman optimality update (value iteration step)
                new_value = max(sum(
                    prob * (reward + gamma * self.V[next_state]) for next_state, prob, reward in
                    self.transition_probabilities[state][action]) for action in self.actions)
                new_V[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            self.V = new_V

            if delta < phi:
                break

    # In DP, we need the model (there transition probabilities) to receive the policy.
    # For example in monte carlo methods, we don't need this
    def extract_policy(self, gamma=0.9):
        for state in self.transition_probabilities:
            best_action = None
            best_value = float('-inf')

            for action in self.actions:
                expected_return = sum(
                    prob * (reward + gamma * self.V[next_state]) for next_state, prob, reward in
                    self.transition_probabilities[state][action])

                if expected_return > best_value:
                    best_value = expected_return
                    best_action = action

            self.policy[state] = best_action

    def render_values(self):
        print("Approximated values of each state")
        for y in range(self.env.height):
            row = ""
            for x in range(self.env.width):
                value = self.V[(x, y)]
                row += f"{value:6.2f} "
            print(row)
        print()

    def render_policy(self):
        print("Learned policy")

        for y in range(self.env.height):
            row = ""
            for x in range(self.env.width):
                if (x, y) == (1, 0) or (x,y) == (3, 0):
                    row += f" T "
                    continue

                action = self.policy.get((x, y), None)
                symbol = action_symbols.get(action, '·')  # Use '·' or ' ' if action is None
                row += f" {symbol} "
            print(row)
        print()

    def solve_env(self):
        # We don't get a reward from the environments since in DP we already know the best
        # policy. Therefore, we don't update anything during the solving - we solved the
        # whole DP problem beforehand.
        self.render_position()
        steps = 0
        while True:
            action = self.policy[self.position]
            steps += 1
            self.position = self._move(self.position, action)
            symbol = action_symbols.get(action, '·')
            print(f"Action: {symbol}")
            print()

            self.render_position()
            if self.position == (1, 0) or self.position == (3, 0):
                print(f"Solved in {steps} steps")
                break


start_position = (2, 4)
env = DynamicProgrammingGridworld()
agent = DynamicProgrammingAgent(env, start_position)
print("Begin training")
agent.train()
print("Finished training")
agent.render_values()
agent.render_policy()


print("Begin simulation ...")
agent.solve_env()



