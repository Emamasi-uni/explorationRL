import sys
from collections import defaultdict
from stable_baselines3 import DQN
from callback import RewardLoggerCallback
from custum_map import GridMappingEnv
from helper import save_dict, load_models


def create_env(size, step, base_model, ig_model, strategy, render=False):
    env = GridMappingEnv(n=size, max_steps=step,
                         ig_model=ig_model,
                         base_model=base_model,
                         strategy=strategy,
                         render_mode='human' if render else None)
    return env


def train(render, strategy):
    if strategy == "random_agent":
        return

    base_model, ig_model = load_models()
    curriculum = [
        {"size": 5, "steps": 250, "episodes": 5000},
        {"size": 10, "steps": 500, "episodes": 15000},
        {"size": 20, "steps": 1000, "episodes": 40000},
        # {"size": 30, "steps": 1200, "episodes": 40000},
    ]

    train_data = defaultdict(list)
    model_dqn = None

    for level in curriculum:
        size = level["size"]
        max_steps = level["steps"]
        episodes = level["episodes"]
        print(f"Training on environment {size}x{size} for {episodes} time steps.")

        # Creazione dell'ambiente per il livello corrente
        env = create_env(size=size, step=max_steps, base_model=base_model, ig_model=ig_model, render=render,
                         strategy=strategy)
        callback = RewardLoggerCallback()

        # Se il modello è già stato addestrato a un livello precedente, usalo come punto di partenza
        if model_dqn is None:
            model_dqn = DQN("MlpPolicy", env, verbose=1, buffer_size=800000)
        else:
            model_dqn.set_env(env)  # Imposta l'ambiente aggiornato sul modello esistente

        # Addestra il modello per il numero di episodi specificato per il livello corrente
        model_dqn.learn(total_timesteps=episodes, callback=callback)

        # Salvataggio dei dati di addestramento
        train_data["episode_rewards"].extend(callback.episode_rewards)
        train_data["episode_cells_marker_pred_1"].extend(callback.episode_cells_marker_pred_1)
        train_data["episode_cells_seen_pov"].extend(callback.episode_cells_seen_pov)
        train_data["episode_steps"].extend(callback.episode_steps)

        # Salvataggio del modello dopo ogni livello di addestramento
        model_dqn.save(f"./data/{strategy}_curriculum/dqn_exploration_{strategy}_level_{size}")

    # Salva tutti i dati di addestramento cumulativi
    save_dict(train_data, f"./data/{strategy}_curriculum/train_data_{strategy}_curriculum.json")


def test(render, strategy, initial_seed=42, num_runs=10):
    test_data = defaultdict(list)
    base_model, ig_model = load_models()
    env = create_env(size=20, step=1000, base_model=base_model, ig_model=ig_model, render=render, strategy=strategy)
    if strategy != "random_agent":
        model_dqn = DQN.load(f"./data/{strategy}_curriculum/dqn_exploration_{strategy}_level_20")

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
            if strategy != "random_agent":
                action, _states = model_dqn.predict(obs)
            else:
                action = env.action_space.sample()
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
            for cell in row if sum(cell['pov']) == 9
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

    save_dict(test_data, f"./data/{strategy}_curriculum/test_data_{strategy}_curriculum.json")


strategy = 'policy2_ig_reward'
train(render=False, strategy=strategy)
test(render=False, strategy=strategy, initial_seed=42, num_runs=20)
