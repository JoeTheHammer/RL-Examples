import sys
import random
from tqdm import tqdm
import os
import time

def clear_console():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.environments import GridworldWithObstacles, GridworldAction

# Example for off policy monte carlo control without exploring starts
# Example for off-policy Monte Carlo control without exploring starts
class MonteCarloAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        """Initialize the agent with policies, Q-values, and returns."""
        self.env = env
        self.gamma = gamma  # Discount factor: how much future rewards are worth
        self.epsilon = epsilon  # Exploration rate for behavior policy (ε-soft)

        self.Q = {}  # Q[s][a]: estimated action-value function
        self.C = {}  # C[s][a]: cumulative sum of importance sampling weights
        self.pi = {}  # π[s]: greedy policy derived from Q (target policy)

        self.actions = list(GridworldAction)  # Available actions

    def initialize_state(self, state):
        """Ensure that state is initialized in Q, C, and pi dictionaries."""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}  # Q-values start at 0
            self.C[state] = {a: 0.0 for a in self.actions}  # Importance weights start at 0
            self.pi[state] = self.actions[0]  # Arbitrary initialization of greedy policy

    def behavior_policy(self, state):
        """Epsilon-soft policy used to generate behavior (training episodes)."""
        self.initialize_state(state)

        # With probability ε choose a random action (exploration),
        # otherwise follow the current greedy policy π (exploitation)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.pi[state]

    def generate_episode(self, max_steps = 5000):
        """Generates one full episode using the behavior policy."""
        episode = []
        state = self.env.reset()  # Reset environment to initial state
        done = False
        steps = 0

        # Generate a sequence of (state, action, reward), stop after 'max_steps' to avoid beeing stucked an episode.
        while not done and steps < max_steps:
            self.initialize_state(state)
            action = self.behavior_policy(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            steps += 1

        return episode

    def update(self, episode):
        """
        Update Q and π using the episode and weighted importance sampling.
        This implements the algorithm from Sutton & Barto, page 111:
        'Off-policy Monte Carlo control for estimating π*'
        """
        G = 0  # Return
        W = 1  # Importance sampling ratio (starts at 1 for full episode weight)

        # Iterate over the episode backwards: from last step to first
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            self.initialize_state(state)

            # Calculate return incrementally: G_t = R_{t+1} + γ * G_{t+1}
            G = self.gamma * G + reward

            # Accumulate importance weights
            self.C[state][action] += W
            q = self.Q[state][action]

            # Update Q using weighted incremental mean
            self.Q[state][action] += (W / self.C[state][action]) * (G - q)

            # Improve policy greedily with respect to Q
            self.pi[state] = max(self.Q[state], key=self.Q[state].get)

            # Early stopping: stop update if behavior policy diverges from target policy
            # Because importance sampling assumes episodes are *possible* under the target policy.
            if action != self.pi[state]:
                break

            # Update importance sampling weight
            prob = self.get_behavior_policy_prob(state, action)
            W = W / prob  # Adjust W to account for likelihood under behavior policy

    def get_behavior_policy_prob(self, state, action):
        """
        Returns the probability that the ε-soft behavior policy chooses `action` in `state`,
        assigning higher probability to the greedy action and equal smaller probability to others.
        """
        self.initialize_state(state)

        n = len(self.actions)
        if action == self.pi[state]:
            # Probability of choosing the greedy action: mostly 1-ε, plus small ε/n
            return 1 - self.epsilon + self.epsilon / n
        else:
            # Probability of choosing a non-greedy action: only from exploration
            return self.epsilon / n

    def train(self, num_episodes=15000):
        print("Training Monte Carlo Agent")
        returns = []

        for _ in tqdm(range(num_episodes), desc="Episodes"):
            episode = self.generate_episode()
            total_reward = sum(r for _, _, r in episode)
            returns.append(total_reward)
            self.update(episode)

        print(f"Average return: {sum(returns) / len(returns):.2f}")

    def get_policy(self):
        """Returns the current target policy π."""
        return self.pi

    def get_q_values(self):
        """Returns the current action-value function Q."""
        return self.Q


env = GridworldWithObstacles()
agent = MonteCarloAgent(env, epsilon=0.2) # Relative high epsilon ensures enough exploration
agent.train()

print(agent.get_policy())
print(agent.get_q_values())

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
print(f"The agent solved the puzzle in {counter} steps. Theoretical best solution is 18.")


