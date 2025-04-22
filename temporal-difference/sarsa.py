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

class SarsaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initializes the SARSA agent.

        Args:
            env: The gridworld environment
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate (for epsilon-greedy policy)
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {}  # Q[state][action] = value
        self.actions = list(GridworldAction)

    def initialize_state(self, state):
        """
        Initializes Q-values for a new state with 0
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

    def sarsa_update(self, state, action, reward, next_state, next_action):
        """
        Performs the SARSA update (Sutton book page 130):
        Q(s, a) ← Q(s, a) + α [R + γ Q(s', a') - Q(s, a)]
        Update is based on the action we actually took, not on the best possible aciton
        (as it would be for q learning)
        """
        self.initialize_state(next_state)

        current_estimate = self.Q[state][action] # Action we actually took used for estimate - on policy learning
        target = reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] += self.alpha * (target - current_estimate)


    def train(self, num_episodes=1000, max_steps=500):
        """
        Trains the agent using SARSA for a number of episodes.
        """
        for _ in tqdm(range(num_episodes), desc="Episodes"):
            state = self.env.reset()
            action = self.choose_action(state)

            for _ in range(max_steps):
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)

                self.sarsa_update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action

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


env = GridworldWithObstacles()
env.render()
agent = SarsaAgent(env)
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
print(f"The SARSA agent solved the puzzle in {counter} steps. Theoretical best solution is 18.")
print()