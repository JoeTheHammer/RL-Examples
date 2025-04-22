import sys
import os
import time
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.environments import GridworldAction
from environments.environments import GridworldWithObstacles

def clear_console():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the Q-learning agent.

        Args:
            env: The gridworld environment
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {}  # Q[state][action] = value
        self.actions = list(GridworldAction)

    def initialize_state(self, state):
        """
        Initializes Q-values for a new state to 0.0 for all actions.
        """
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}

    def choose_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.
        """
        self.initialize_state(state)
        if random.random() < self.epsilon:
            # Explore: Do a random action
            return random.choice(self.actions)
        else:
            # Exploit: Choose next action greedily
            return max(self.Q[state], key=self.Q[state].get)

    def q_learning_update(self, state, action, reward, next_state):
        """
        Performs the Q-learning update (Sutton book page 131):
        Q(s, a) ← Q(s, a) + α [R + γ max_a' Q(s', a') - Q(s, a)]
        This is off-policy because it uses the best possible next action, not the one actually taken.
        """
        self.initialize_state(next_state)

        # Get value of best next action (not the one actually taken)
        current_estimate = self.Q[state][action]
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
        target = reward + self.gamma * self.Q[next_state][best_next_action]
        self.Q[state][action] += self.alpha * (target - current_estimate)

    def train(self, num_episodes=1000, max_steps=500):
        """
        Trains the agent using Q-learning for a number of episodes.
        """
        for _ in tqdm(range(num_episodes), desc="Episodes"):
            state = self.env.reset()

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.q_learning_update(state, action, reward, next_state)

                state = next_state
                if done:
                    break

    def get_policy(self):
        """
        Returns the greedy policy derived from Q.
        """
        return { s: max(a_map, key=a_map.get) for s, a_map in self.Q.items() }

    def get_q_values(self):
        """
        Returns the current Q-values.
        """
        return self.Q


hard_env = True

best_solution = 18
if hard_env:
    best_solution = 40

env = GridworldWithObstacles(hard_env)
env.render()
agent = QLearningAgent(env)
agent.train(50000)

clear_console()

print("---- START THE GAME ----")
done = False
state = env.reset()
counter = 0
env.render()

while not done:
    clear_console()              # Clear terminal
    env.render()                 # Draw updated grid
    time.sleep(0.5)              # Pause for animation effect
    action = agent.get_policy().get(state)
    next_state, reward, done, _ = env.step(action)
    counter += 1
    state = next_state

clear_console()
env.render()
print()
print(f"The Q Learning agent solved the puzzle in {counter} steps. Theoretical best solution is {best_solution}.")
print()