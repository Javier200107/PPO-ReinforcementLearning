import gymnasium as gym
import numpy as np
from src.PPOAgent import PPOAgentDiscrete
from src.utils import plot_learning_curve
import os
import datetime
from sklearn.model_selection import ParameterGrid

config_grid = {
    'N': [20],
    'n_games': [5000],
    'batch_size': [5],
    'actor_lr': [0.001], 
    'critic_lr': [0.001],
    'gamma': [0.99], # discount factor
    'n_epochs': [4],
    'epsilon': [0.1, 0.2],  # clip ratio
    'gae_lambda': [0.95],
    'checkpoint_dir': ['./models/'],
    'video_dir': ['./video/'],
    'seed': [0],
    'c1': [0.5, 1], # Value coefficient
    'c2': [0.01, 0.001], # Entropy coefficient
}

env = gym.make("CartPole-v1", render_mode='rgb_array')
for i, conf in enumerate(list(ParameterGrid(config_grid))):

    """if not os.path.exists(conf['video_dir']):
        os.makedirs(conf['video_dir'])  # Create the directory if it does not exist
    new_dir = conf['checkpoint_dir'] + '/' + "discrete" + "/" + env.spec.id + "/" + 'gamma_' + str(conf['gamma']) + '_epsilon_' + str(conf['epsilon']) + '_gae_lambda_' + str(conf['gae_lambda']) + '/' + str(k)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir) """
    print(env.action_space.n)
    print(env.observation_space.shape)

    agent = PPOAgentDiscrete(
        n_actions=env.action_space.n, input_dims=env.observation_space.shape,
        checkpoint_directory=None, batch_size=conf['batch_size'], actorn_lr=conf['actor_lr'], critic_lr=conf['critic_lr'],
        gamma=conf['gamma'], clip_ratio=conf['epsilon'], c1=conf['c1'], c2=conf['c2'], gae_lambda=conf['gae_lambda'], n_epochs=conf['n_epochs'], 
        env_name=env.spec.id, seed=conf['seed']
    )
    #timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    results_dir = './results/' + agent.env_type + '/' + env.spec.id + '/' + 'epsilon_' + str(conf['epsilon']) + '_c1_' + str(conf['c1']) + '_c2_' + str(conf['c2']) + '/'
    plots_dir = results_dir + '/plots/'

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters_per_episode = 0
    learn_iters_total = 0
    avg_score = 0
    avg_score_last_20 = 0
    std_score = 0
    std_score_last_20 = 0
    time_steps_per_episode = 0
    total_time_steps = 0
    total_terminations = 0
    total_truncations = 0

    f_episode_logs = open(results_dir + '/episode_logs.txt', 'w')

    #for i in range(n_games):
    i = 0
    while avg_score_last_20 < 500 and i < conf['n_games']:
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, value, log_probs = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(
                action)
            if terminated:
                total_terminations += 1
            if truncated:
                total_truncations += 1
            done = terminated or truncated
            time_steps_per_episode += 1
            score += reward
            agent.store_transition(
                observation, action, log_probs, value, reward, done
            )
            if time_steps_per_episode % conf['N'] == 0:
                agent.learn()
                learn_iters_per_episode += 1
            observation = observation_
        total_time_steps += time_steps_per_episode
        learn_iters_total += learn_iters_per_episode
        score_history.append(score)
        avg_score = np.mean(score_history)
        avg_score_last_20 = np.mean(score_history[-20:])
        std_score = np.std(score_history)
        std_score_last_20 = np.std(score_history[-20:])

        if avg_score_last_20 > best_score:
            best_score = score
            #agent.save_models()

        logs_e = f"episode {i}, score {score}, avg_score_total {avg_score:.3f}, avg_score_last_20 {avg_score_last_20:.3f}, time_steps_per_episode {time_steps_per_episode}, learning_steps {learn_iters_per_episode}, std_score {std_score:.3f}, std_score_last_20 {std_score_last_20:.3f}"
        print(logs_e)
        logs_to_write = f"episode {i}, score {score}, avg_score_total {avg_score:.3f}, avg_score_last_20 {avg_score_last_20:.3f}, time_steps_per_episode {time_steps_per_episode}, learning_steps {learn_iters_per_episode}, std_score {std_score:.3f}, std_score_last_20 {std_score_last_20:.3f}, total_time_steps {total_time_steps}, total_learning_steps {learn_iters_total}, total_terminations {total_terminations}, total_truncations {total_truncations}"
        f_episode_logs.write(logs_to_write + '\n')

        learn_iters_per_episode = 0
        time_steps_per_episode = 0
        i += 1

    env.close()
    with open(results_dir + '/conf.txt', 'w') as f:
        f.write(str(conf))
    with open(results_dir + '/losses.txt', 'w') as f:
        f.write('actor_loss\n')
        for loss in agent.actor_loss_history:
            f.write(str(loss) + '\n')
        f.write('critic_loss\n')
        for loss in agent.critic_loss_history:
            f.write(str(loss) + '\n')
    x = [j + 1 for j in range(i)]
    plot_learning_curve(x, score_history, plots_dir, combo= f'conf: epsilon {conf["epsilon"]} c1 {conf["c1"]} c2 {conf["c2"]}')
