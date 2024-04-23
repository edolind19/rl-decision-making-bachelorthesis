import time
import gym
import numpy as np
import matplotlib.pyplot as plt

# Definition einer einfachen Umgebung für das autonome Fahrzeug
class SimpleDrivingEnv(gym.Env):
    def __init__(self):
        super(SimpleDrivingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(2,), dtype=int)
        self.state = None
        self.goal = (4, 4)  
        self.obstacles = [(2, 2), (1, 3)]  

    def reset(self):
        self.state = (0, 0)  
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0: x = max(x - 1, 0)
        elif action == 1: x = min(x + 1, 5)
        elif action == 2: y = max(y - 1, 0)
        elif action == 3: y = min(y + 1, 5)
        
        self.state = (x, y)
        
        done = self.state == self.goal
        collision = self.state in self.obstacles
        reward = 10 if done else -10 if collision else -1
        
        return self.state, reward, done or collision, {}

    def render(self, mode='human'):
        grid = np.zeros((6, 6), dtype=int)
        grid[self.goal] = 2  
        for obs in self.obstacles:
            grid[obs] = -1 
        grid[self.state] = 1
        print(grid)

# Training des Agenten mit Q-Learning in der erweiterten Umgebung
def train_q_learning(env, episodes=1000):
    q_table = np.zeros((6, 6, env.action_space.n))
    alpha = 0.1  
    gamma = 0.99  
    epsilon = 1.0  

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:  
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

        epsilon *= 0.99  

    return q_table


# Bewertung des Agenten
def evaluate_agent(env, q_table=None, episodes=100, use_random=False):
    metrics = {
        'average_steps_to_goal': 0,
        'average_rewards': 0,
        'collision_count': 0,
        'success_rate': 0,
        'min_rewards': float('inf'),
        'max_rewards': float('-inf'),
        'avg_negative_rewards': 0,
        'avg_successful_steps': 0,
        'max_penalty_episodes': 0,
        'average_time_to_goal': 0
    }
    total_negative_rewards = 0
    successful_steps = []
    time_to_goal = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        steps = 0
        episode_rewards = []
        episode_start_time = time.time()  

        while not done:
            if use_random or q_table is None:
                action = env.action_space.sample()
            else:
                x, y = state
                action = np.argmax(q_table[x, y])

            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            episode_rewards.append(reward)
            steps += 1

            if reward == -1:
                metrics['collision_count'] += 1
            
            if done and reward > 0:
                metrics['success_rate'] += 1
                successful_steps.append(steps)
                time_to_goal.append(time.time() - episode_start_time)

            state = next_state

        metrics['average_steps_to_goal'] += steps
        metrics['average_rewards'] += total_rewards
        metrics['min_rewards'] = min(metrics['min_rewards'], min(episode_rewards, default=0))
        metrics['max_rewards'] = max(metrics['max_rewards'], max(episode_rewards, default=0))
        total_negative_rewards += sum(r for r in episode_rewards if r < 0)
        if min(episode_rewards, default=0) == -10:
            metrics['max_penalty_episodes'] += 1

    metrics['average_steps_to_goal'] /= episodes
    metrics['average_rewards'] /= episodes
    metrics['avg_negative_rewards'] = total_negative_rewards / episodes
    metrics['success_rate'] /= episodes
    if successful_steps:
        metrics['avg_successful_steps'] = sum(successful_steps) / len(successful_steps)
    if time_to_goal:
        metrics['average_time_to_goal'] = sum(time_to_goal) / len(time_to_goal)

    return metrics

# Visualisierung der Agentenbewegung in der erweiterten Umgebung
def visualize_agent_path(env, q_table, use_random, episode, show=False):
    state = env.reset()
    done = False
    path = [state]

    # Starte Visualisierung
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # Zeichne den Start- und den Zielpunkt sowie Hindernisse
    ax.plot(0, 0, 'go', markersize=10, label='Start')  
    ax.plot(env.goal[1], env.goal[0], 'ro', markersize=10, label='Goal') 
    for obs in env.obstacles:
        ax.plot(obs[1], obs[0], 'xk', markersize=10, label='Obstacle' if 'Obstacle' not in ax.get_legend_handles_labels()[1] else "")

    # Zeichnet den Pfad des AF
    for _ in range(25):  
        if use_random:
            action = env.action_space.sample()
        else:
            x, y = state
            action = np.argmax(q_table[x, y])
        state, _, done, _ = env.step(action)
        path.append(state)
        if done:
            break

    xs, ys = zip(*path)
    ax.plot(ys, xs, 'o-', label='Path', markersize=5, linewidth=2)

    # Zeichne ein "Auto" als Punkt am Ende des Pfades
    ax.plot(ys[-1], xs[-1], 'bs', markersize=10, label='Car')

    # Legende
    ax.legend()

    # Zeigt die Visualisierung
    if show:
        plt.title('Agent Path with' + ('out ' if use_random else ' ') + 'Decision Algorithm')
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


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
fig, ax = visualize_agent_path(env, q_table, use_random=True, episode=0, show=True)

# Visualisierung mit Entscheidungsalgorithmen
print("Visualisierung mit Entscheidungsalgorithmen:")
fig, ax = visualize_agent_path(env, q_table, use_random=False, episode=0, show=True)

env.close()