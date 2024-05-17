import gymnasium as gym
import numpy as np
from src.PPOAgent import PPOAgentContinuous
from src.utils import plot_learning_curve
import os
import datetime
from gymnasium.wrappers import RecordVideo

conf = {
    'N': 20,
    'n_games': 6000,
    'batch_size': 64,
    'actor_lr': 0.0003, 
    'critic_lr': 0.0003,
    'gamma': 0.99, # discount factor
    'n_epochs': 16,
    'epsilon': 0.2,  # clip ratio
    'gae_lambda': 0.95,
    'checkpoint_dir': './models/',
    'video_dir': './video/',
    'seed': 0,
    'c1': 0.5,
    'c2': 0.0,
}

env = gym.make('Pendulum-v1', g=9.81, render_mode='rgb_array')
env = RecordVideo(env, video_folder=conf['video_dir'], episode_trigger=lambda x: x == conf['n_games'] - 1)

# Check if the checkpoint directory exists
if not os.path.exists(conf['video_dir']):
    os.makedirs(conf['video_dir'])  # Create the directory if it does not exist

if not os.path.exists(conf['checkpoint_dir']):
    os.makedirs(conf['checkpoint_dir']) 

# input_dims = env.observation_space.shape???
agent = PPOAgentContinuous(
    n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape,
    checkpoint_directory=conf['checkpoint_dir'], batch_size=conf['batch_size'], actorn_lr=conf['actor_lr'], critic_lr=conf['critic_lr'],
    gamma=conf['gamma'], clip_ratio=conf['epsilon'], c1=conf['c1'], c2=conf['c2'], gae_lambda=conf['gae_lambda'], n_epochs=conf['n_epochs'], 
    env_name=env.spec.id, seed=conf['seed']
)
#timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
results_dir = './results/' + agent.env_type + '/' + env.spec.id + '/' + 'epsilon_' + str(conf['epsilon']) + '_c1_' + str(conf['c1']) + '_c2_' + str(conf['c2']) + '/'
plots_dir = results_dir + '/plots/'

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

with open(results_dir + '/conf.txt', 'w') as f:
    f.write(str(conf))

best_score = env.reward_range[0]
score_history = []

learn_iters_per_episode = 0
learn_iters_total = 0
avg_score = 0
avg_score_last_20 = best_score
std_score = 0
std_score_last_20 = 0
time_steps_per_episode = 0
total_time_steps = 0
total_terminations = 0
total_truncations = 0

f_episode_logs = open(results_dir + '/episode_logs.txt', 'w')

for i in range(conf['n_games']):
#i = 0
#while avg_score_last_20 < 0:
    observation, _ = env.reset()
    done = False
    score = 0
    while not done:
        action, value, log_probs = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(
            action
        )
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

    logs_e = f"episode {i}, score {score:.2f}, avg_score_total {avg_score:.2f}, avg_score_last_20 {avg_score_last_20:.2f}, time_steps_per_episode {time_steps_per_episode}, learning_steps {learn_iters_per_episode}, std_score {std_score:.2f}, std_score_last_20 {std_score_last_20:.2f}"
    print(logs_e)
    logs_to_write = f"episode {i}, score {score:.2f}, avg_score_total {avg_score:.2f}, avg_score_last_20 {avg_score_last_20:.2f}, time_steps_per_episode {time_steps_per_episode}, learning_steps {learn_iters_per_episode}, std_score {std_score:.2f}, std_score_last_20 {std_score_last_20:.2f}, total_time_steps {total_time_steps}, total_learning_steps {learn_iters_total}, total_terminations {total_terminations}, total_truncations {total_truncations}"
    f_episode_logs.write(logs_to_write + '\n')

    learn_iters_per_episode = 0
    time_steps_per_episode = 0
    #i += 1

env.close()
with open(results_dir + '/conf.txt', 'w') as f:
    f.write(str(conf))
x = [j + 1 for j in range(conf['n_games'])]
plot_learning_curve(x, score_history, plots_dir)
