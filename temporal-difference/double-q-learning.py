import sys
import os
import time
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.environments import GridworldAction, GridworldWithObstacles

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


class DoubleQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Double Q‑Learning agent: maintains two Q‑tables (Q1 & Q2).
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
        """
        self.env = env
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        # **CRUCIAL DIFFERENCE**: two separate Q‑tables
        self.Q1 = {}  # Q1[state][action]
        self.Q2 = {}  # Q2[state][action]
        self.actions = list(GridworldAction)

    def initialize_state(self, state):
        """Make sure both Q1 & Q2 have entries for this state."""
        if state not in self.Q1:
            self.Q1[state] = {a: 0.0 for a in self.actions}
            self.Q2[state] = {a: 0.0 for a in self.actions}

    def choose_action(self, state):
        """
        Epsilon‑greedy based on the **sum** of Q1+Q2:
        a* = argmax_a [Q1(s,a) + Q2(s,a)]
        """
        self.initialize_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Exploit: greedy w.r.t. combined estimate (what Q1 thinks is the best action in that state
        # plus what Q2 thinks is the best action in that state)
        q_sum = {a: self.Q1[state][a] + self.Q2[state][a] for a in self.actions}
        return max(q_sum, key=q_sum.get)

    def double_q_update(self, state, action, reward, next_state):
        """
            Performs one Double Q‑Learning update:
            The core idea of Double Q-Learning is to reduce overestimation bias in action-value estimates
            that happens in regular Q-Learning.

            It does this by *decoupling* action selection and action evaluation:
            - One Q-table is used to select the best action at the next state.
            - The *other* Q-table is used to evaluate how good that action actually is.

            This avoids overestimation because we don't use the same values for both selection and evaluation.
            """

        # Ensure the next state is initialized in both Q1 and Q2
        self.initialize_state(next_state)

        # Use "coin flip" to evaluate which Q is which
        if random.random() < 0.5:
            # ➤ Select the best next action according to Q1
            best_next = max(self.Q1[next_state], key=self.Q1[next_state].get)

            # ➤ But use Q2 to estimate its value → avoids bias
            estimate_of_other_q = self.Q2[next_state][best_next]

            # Standard TD update for Q1
            target = reward + self.gamma * estimate_of_other_q
            self.Q1[state][action] += self.alpha * (target - self.Q1[state][action])
        else:
            # Update Q2, use Q1 to evaluate
            # ➤ Select the best next action according to Q2
            best_next = max(self.Q2[next_state], key=self.Q2[next_state].get)

            # ➤ Use Q1 to evaluate it
            estimate_of_other_q = self.Q1[next_state][best_next]

            # Standard TD update for Q2
            target = reward + self.gamma * estimate_of_other_q
            self.Q2[state][action] += self.alpha * (target - self.Q2[state][action])

    def train(self, num_episodes=1000, max_steps=500):
        """Main training loop, identical structure to Q‑Learning."""
        for _ in tqdm(range(num_episodes), desc="Episodes"):
            state = self.env.reset()
            for _ in range(max_steps):
                action     = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.double_q_update(state, action, reward, next_state)
                state = next_state
                if done:
                    break

    def get_policy(self):
        """
        Greedy policy from the **sum** of Q1 + Q2:
        π(s) = argmax_a [Q1(s,a) + Q2(s,a)]
        """
        policy = {}
        for s in self.Q1:
            q_sum = {a: self.Q1[s][a] + self.Q2[s][a] for a in self.actions}
            policy[s] = max(q_sum, key=q_sum.get)
        return policy

    def get_q_values(self):
        """Returns both Q1 and Q2 tables (for analysis)."""
        return self.Q1, self.Q2


# ——— Usage Example ———

if __name__ == "__main__":

    hard_env = False

    best_solution = 18
    if hard_env:
        best_solution = 40

    env   = GridworldWithObstacles(hard_env)
    env.render()

    agent = DoubleQLearningAgent(env)
    agent.train(50000)

    clear_console()
    print("---- START THE GAME ----")
    done  = False
    state = env.reset()
    counter = 0

    while not done:
        clear_console()
        env.render()
        time.sleep(0.5)

        action = agent.get_policy().get(state)
        state, _, done, _ = env.step(action)
        counter += 1

    clear_console()
    env.render()
    print(f"The Double Q Learning agent solved the puzzle in {counter} steps. Theoretical best solution is {best_solution}.")
