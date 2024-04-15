import gym
import numpy as np
import matplotlib.pyplot as plt

# Definition einer einfachen Umgebung für das autonome Fahrzeug
class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # Aktionen: oben, unten, links, rechts
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(2,), dtype=int)
        self.state = None
        self.goal = (4, 4)  # Zielposition

    def reset(self):
        self.state = (0, 0)  # Startposition
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0: x = max(x - 1, 0)
        elif action == 1: x = min(x + 1, 5)
        elif action == 2: y = max(y - 1, 0)
        elif action == 3: y = min(y + 1, 5)
        
        self.state = (x, y)
        
        done = self.state == self.goal
        reward = 1 if done else -1 if self.state == (0, 0) else 0
        
        return self.state, reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((6, 6), dtype=int)
        grid[self.goal] = 2  # Ziel mit 2 markieren
        grid[self.state] = 1  # Aktuelle Position des Agenten mit 1 markieren
        print(grid)

# Training des Agenten mit Q-Learning
def train_q_learning(env, episodes=1000):
    q_table = np.zeros((6, 6, env.action_space.n))
    alpha = 0.1  # Lernrate
    gamma = 0.99  # Diskontierungsfaktor
    epsilon = 1.0  # Anfangswert für Exploration

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:  # Exploration
                action = env.action_space.sample()
            else:  # Exploitation
                x, y = state
                action = np.argmax(q_table[x, y])

            next_state, reward, done, _ = env.step(action)
            x, y = state
            nx, ny = next_state

            old_value = q_table[x, y, action]
            next_max = np.max(q_table[nx, ny])

            # Update der Q-Tabelle
            q_table[x, y, action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state

        epsilon *= 0.99  # Reduzierung der Explorationsrate

    return q_table

# Bewertung des Agenten
def evaluate_agent(env, q_table=None, episodes=100, use_random=False):
    metrics = {
        'average_steps_to_goal': 0,
        'average_rewards': 0,
        'collision_count': 0,
        'success_rate': 0
    }

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        steps = 0

        while not done:
            if use_random or q_table is None:  # Ohne Entscheidungsalgorithmus
                action = env.action_space.sample()
            else:  # Mit Entscheidungsalgorithmus
                x, y = state
                action = np.argmax(q_table[x, y])

            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            steps += 1
            
            if reward == -1:
                metrics['collision_count'] += 1
            
            if done and reward > 0:
                metrics['success_rate'] += 1

            state = next_state
        
        metrics['average_steps_to_goal'] += steps
        metrics['average_rewards'] += total_rewards

    metrics['average_steps_to_goal'] /= episodes
    metrics['average_rewards'] /= episodes
    metrics['success_rate'] /= episodes

    return metrics

# Visualisierung der Agentenbewegung
def visualize_agent_path(env, q_table, use_random, episode):
    state = env.reset()
    done = False
    states = [state]
    while not done:
        if use_random:
            action = env.action_space.sample()
        else:
            x, y = state
            action = np.argmax(q_table[x, y])
        state, _, done, _ = env.step(action)
        states.append(state)

    xs, ys = zip(*states)
    plt.plot(ys, xs, 'o-', label='Path')
    plt.plot(ys[0], xs[0], 'go', markersize=10, label='Start')  # Start
    plt.plot(ys[-1], xs[-1], 'ro', markersize=10, label='Goal')  # Ziel
    plt.grid()
    plt.legend()
    plt.title('Agent Path' + (' with ' if not use_random else ' without ') + 'Decision Algorithm')
    plt.show()

# Hauptausführung
env = SimpleDrivingEnv()
q_table = train_q_learning(env)

# Bewertung des Agenten ohne Entscheidungsalgorithmen
metrics_without_algo = evaluate_agent(env, use_random=True)
print("Leistung ohne Entscheidungsalgorithmen:", metrics_without_algo)

# Bewertung des Agenten mit Entscheidungsalgorithmen
metrics_with_algo = evaluate_agent(env, q_table)
print("Leistung mit Entscheidungsalgorithmen:", metrics_with_algo)

# Visualisierung ohne Entscheidungsalgorithmen
print("Visualisierung ohne Entscheidungsalgorithmen:")
visualize_agent_path(env, q_table, use_random=True, episode=0)

# Visualisierung mit Entscheidungsalgorithmen
print("Visualisierung mit Entscheidungsalgorithmen:")
visualize_agent_path(env, q_table, use_random=False, episode=0)

env.close()