import gym
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import math

class SimpleDrivingEnv(gym.Env):
    def __init__(self, sensor_data):
        super(SimpleDrivingEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)  
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, -np.pi, -3, -3]), high=np.array([5, 5, np.pi, 3, 3]), dtype=np.float32)
        self.state = None
        self.goal = (4, 4)
        self.obstacles = [(2, 2), (1, 3)]
        self.sensor_data = sensor_data
        self.current_step = 0

    def reset(self):
        self.state = [0, 0, 0, 0, 0]  
        self.current_step = 0
        return np.array(self.state)

    def step(self, action):
        x, y, theta, vx, vy = self.state
        acc, steering = action
        theta += steering * 0.05 
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  
        vx += acc * math.cos(theta)  
        vy += acc * math.sin(theta)  
        x += vx
        y += vy
        x = np.clip(x, 0, 5)
        y = np.clip(y, 0, 5)

        self.state = [x, y, theta, vx, vy]
        self.current_step += 1
        
        done = np.linalg.norm([x - self.goal[0], y - self.goal[1]]) < 0.5
        collision = any(np.linalg.norm([x - ox, y - oy]) < 0.5 for ox, oy in self.obstacles)
        reward = 100 if done else -100 if collision else -1 - 0.1 * (vx**2 + vy**2)

        return np.array(self.state), reward, done or collision, {}

    def render(self, mode='human'):
        grid = np.zeros((6, 6), dtype=int)
        grid[self.goal] = 2
        for obs in self.obstacles:
            grid[obs] = -1
        grid[int(self.state[0]), int(self.state[1])] = 1
        plt.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.title("Simple Driving Environment")
        plt.show()
    
    def get_sensor_value(self):
        # Rückgabe des Gierwinkels (Yaw) als Orientierungsmaß
        if self.current_step < len(self.sensor_data):
            return self.sensor_data.iloc[self.current_step]['yaw']
        else:
            return 0  

# Training des Agenten mit Q-Learning in der erweiterten Umgebung
def train_q_learning(env, episodes=1000):
    q_table = np.zeros((6, 6, 3, 3, env.action_space.n)) 
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            x, y, vx, vy = state
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[x, y, vx + 1, vy + 1])

            next_state, reward, done, _ = env.step(action)
            nx, ny, nvx, nvy = next_state

            old_value = q_table[x, y, vx + 1, vy + 1, action]
            next_max = np.max(q_table[nx, ny, nvx + 1, nvy + 1])

            # Update der Q-Tabelle
            q_table[x, y, vx + 1, vy + 1, action] = old_value + alpha * (reward + gamma * next_max - old_value)

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

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        steps = 0

        while not done:
            x, y, vx, vy = state
            if use_random or q_table is None:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[x, y, vx + 1, vy + 1])

            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            steps += 1

            if reward == -100:
                metrics['collision_count'] += 1
            
            if done and reward == 100:
                metrics['success_rate'] += 1
                successful_steps.append(steps)

            state = next_state

        metrics['average_steps_to_goal'] += steps
        metrics['average_rewards'] += total_rewards
        metrics['min_rewards'] = min(metrics['min_rewards'], total_rewards)
        metrics['max_rewards'] = max(metrics['max_rewards'], total_rewards)
        if total_rewards < 0:
            total_negative_rewards += total_rewards
        if total_rewards == -100:
            metrics['max_penalty_episodes'] += 1

    metrics['average_steps_to_goal'] /= episodes
    metrics['average_rewards'] /= episodes
    metrics['avg_negative_rewards'] = total_negative_rewards / episodes
    metrics['success_rate'] /= episodes
    if successful_steps:
        metrics['avg_successful_steps'] = sum(successful_steps) / len(successful_steps)

    return metrics

# Hauptausführung
if __name__ == "__main__":
    sensor_data = pd.read_csv('Orientation.csv')
    env = SimpleDrivingEnv(sensor_data)
    q_table = train_q_learning(env)

    # Anzahl der Durchläufe
    num_runs = 30

    # Liste der Metriken
    metric_names = ['average_steps_to_goal', 'average_rewards', 'collision_count', 'success_rate', 
                    'min_rewards', 'max_rewards', 'avg_negative_rewards', 'avg_successful_steps', 
                    'max_penalty_episodes', 'average_time_to_goal']

    # Öffne die CSV-Datei im Schreibmodus
    with open('evaluation_results.csv', 'w', newline='') as csvfile:
        # Erstelle einen CSV-Writer
        writer = csv.writer(csvfile)

        # Schreibe die Headerzeile
        writer.writerow(['Run'] + metric_names)

        # Schleife für mehrere Durchläufe
        for i in range(num_runs):
            env = SimpleDrivingEnv()
            q_table = train_q_learning(env)
            print(f"Durchlauf {i+1}/{num_runs}")

            # Bewertet Agenten ohne Entscheidungsalgorithmen
            metrics_without_algo = evaluate_agent(env, use_random=True)
            print("Leistung ohne Entscheidungsalgorithmen:", metrics_without_algo)

            # Bewertet Agenten mit Entscheidungsalgorithmen
            metrics_with_algo = evaluate_agent(env, q_table)
            print("Leistung mit Entscheidungsalgorithmen:", metrics_with_algo)

            # Schreibt die Ergebnisse in die CSV-Datei
            writer.writerow([i+1] + [metrics_without_algo[metric] for metric in metric_names])
            writer.writerow([i+1] + [metrics_with_algo[metric] for metric in metric_names])

    env.close()
