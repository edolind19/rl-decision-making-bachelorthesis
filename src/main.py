import gym
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

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
        
        sensor_value = self.get_sensor_value()  # Hier wird der Sensorwert abgerufen
        return self.state, reward, done or collision, sensor_value

    def render(self, mode='human'):
        grid = np.zeros((6, 6), dtype=int)
        grid[self.goal] = 2  
        for obs in self.obstacles:
            grid[obs] = -1 
        grid[self.state] = 1
        plt.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.title("Simple Driving Environment")
        plt.show()
    
    def get_sensor_value(self):
        # Dummy-Sensorwert, der eine zufällige Zahl zwischen 0 und 100 zurückgibt
        return np.random.randint(0, 101)

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
            else:  
                x, y = state
                action = np.argmax(q_table[x, y])
                next_max = np.max(q_table[nx, ny])

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

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        steps = 0

        while not done:
            if use_random or q_table is None:
                action = env.action_space.sample()
            else:
                x, y = state
                action = np.argmax(q_table[x, y])

            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            steps += 1

            if reward == -1:
                metrics['collision_count'] += 1
            
            if done and reward > 0:
                metrics['success_rate'] += 1
                successful_steps.append(steps)

            state = next_state

        metrics['average_steps_to_goal'] += steps
        metrics['average_rewards'] += total_rewards
        metrics['min_rewards'] = min(metrics['min_rewards'], total_rewards)
        metrics['max_rewards'] = max(metrics['max_rewards'], total_rewards)
        total_negative_rewards += total_rewards if total_rewards < 0 else 0
        if total_rewards == -10:
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
    env = SimpleDrivingEnv()
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
