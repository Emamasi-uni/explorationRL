import os
import sys
from collections import defaultdict

import torch
from stable_baselines3 import DQN

from base_model import netCounterBase
from callback import RewardLoggerCallback
from constants import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES_IG, NUM_CLASSES_BASE
from helper import save_dict
from ig_model import NetIG
from custum_map import GridMappingEnv


def load_models():
    base_model = netCounterBase(NUM_CLASSES_BASE, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    ig_model = NetIG(NUM_CLASSES_IG, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    base_model_path = os.path.join('data', 'fold_1_saved_base_model.pth')
    ig_model_path = os.path.join('data', 'fold_1_saved_ig_model_global_MSE_entropy_loss_head.pth')
    if torch.cuda.is_available():
        base_model.load_state_dict(torch.load(base_model_path))
        ig_model.load_state_dict(torch.load(ig_model_path))
    else:
        base_model.load_state_dict(torch.load(base_model_path, map_location=torch.device('cpu')))
        ig_model.load_state_dict(torch.load(ig_model_path, map_location=torch.device('cpu')))

    return base_model, ig_model


def create_env(size, step, base_model, ig_model, strategy, render=False):
    env = GridMappingEnv(n=size, max_steps=step,
                         ig_model=ig_model,
                         base_model=base_model,
                         strategy=strategy,
                         render_mode='human' if render else None)
    return env


def train(episodes, render, strategy):
    train_data = defaultdict(list)
    callback = RewardLoggerCallback()
    base_model, ig_model = load_models()
    env = create_env(size=10, step=500, base_model=base_model, ig_model=ig_model, render=render, strategy=strategy)

    model_dqn = DQN("MlpPolicy", env, verbose=1)
    model_dqn.learn(total_timesteps=episodes, callback=callback)

    train_data["episode_rewards"] = callback.episode_rewards
    train_data["episode_cells_marker_pred_1"] = callback.episode_cells_marker_pred_1
    train_data["episode_cells_seen_pov"] = callback.episode_cells_seen_pov
    train_data["episode_steps"] = callback.episode_steps

    save_dict(train_data, f"./data/{strategy}/train_data_{strategy}.json")

    model_dqn.save(f"./data/{strategy}/dqn_exploration_{strategy}")
    del model_dqn


def test(render, strategy, initial_seed=42, num_runs=10):
    test_data = defaultdict(list)
    base_model, ig_model = load_models()
    env = create_env(size=10, step=500, base_model=base_model, ig_model=ig_model, render=render, strategy=strategy)

    model_dqn = DQN.load(f"./data/{strategy}/dqn_exploration_{strategy}")

    # Lista per tenere traccia delle metriche per ogni run
    cumulative_rewards_per_run = []
    cells_marker_pred_1_per_run = []
    cells_seen_pov_per_run = []
    total_steps_per_run = []
    cells_marker_pred_1_each_step = []

    for run in range(num_runs):
        seed = initial_seed + run
        obs, info = env.reset(seed=seed)
        cumulative_reward = 0.0
        steps = 0
        cells_marker_pred_1_run = []
        while True:
            action, _states = model_dqn.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            cumulative_reward += reward
            steps += 1
            cells_marker_pred_1_run.append(sum(
                1 for row in env.state[1:env.n + 1, 1:env.n + 1]
                for cell in row if cell['marker_pred'] == 1
            ))

            if terminated or truncated:
                break

        cells_marker_pred_1_each_step.append(cells_marker_pred_1_run)

        # Aggiorna il conteggio delle celle viste da 9 punti di vista e delle celle con marker predetto corretto
        cells_marker_pred_1 = sum(
            1 for row in env.state[1:env.n + 1, 1:env.n + 1]
            for cell in row if cell['marker_pred'] == 1
        )

        cells_seen_pov = sum(
            1 for row in env.state[1:env.n + 1, 1:env.n + 1]
            for cell in row if sum(cell['pov']) == 0
        )

        # Salva le metriche per il run corrente
        cumulative_rewards_per_run.append(cumulative_reward)
        cells_marker_pred_1_per_run.append(cells_marker_pred_1)
        cells_seen_pov_per_run.append(cells_seen_pov)
        total_steps_per_run.append(steps)

    max_length = max(len(sotto_lista) for sotto_lista in cells_marker_pred_1_each_step)

    somme = [0] * max_length
    conteggi = [0] * max_length

    # Sommare gli elementi corrispondenti di ogni lista
    for sotto_lista in cells_marker_pred_1_each_step:
        for i, valore in enumerate(sotto_lista):
            somme[i] += valore
            conteggi[i] += 1

    # Calcolare la media per ogni posizione
    cells_marker_pred_1_mean = [somme[i] / conteggi[i] for i in range(max_length)]
    # Stampa le metriche
    print("Cumulative Rewards per Run:", cumulative_rewards_per_run)
    print("Cells with Correct Marker Prediction per Run:", cells_marker_pred_1_per_run)
    print("Cells Seen from 0 POVs per Run:", cells_seen_pov_per_run)

    test_data["cells_marker_pred_1_mean"] = cells_marker_pred_1_mean
    test_data["cumulative_rewards_per_run"] = cumulative_rewards_per_run
    test_data["cells_marker_pred_1_per_run"] = cells_marker_pred_1_per_run
    test_data["cells_seen_pov_per_run"] = cells_seen_pov_per_run
    test_data["total_steps_per_run"] = total_steps_per_run

    save_dict(test_data, f"./data/{strategy}/test_data_{strategy}.json")


strategy = sys.argv[1]
# 50000
train(episodes=15000, render=False, strategy=strategy)
test(render=True, strategy=strategy, initial_seed=42, num_runs=10)