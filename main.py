import gymnasium as gym
import numpy as np
from PPOAgent import PPOAgent
from utils import plot_learning_curve
import os

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N = 20
    n_games = 250
    batch_size = 5
    alpha = 0.0003
    n_epochs = 4
    chekpoint_dir = './models/'
    plots_dir = './plots/'

    figure_file = plots_dir + env.spec.id + '.png'

    # Check if the checkpoint directory exists
    if not os.path.exists(chekpoint_dir):
        os.makedirs(chekpoint_dir)  # Create the directory if it does not exist

    # Check if the plots directory exists
    if not os.path.exists('plots/'):
        os.makedirs('plots/')  # Create the directory if it does not exist

    agent = PPOAgent(
        n_actions=env.action_space.n, input_dims=env.observation_space.shape,
        checkpoint_directory=chekpoint_dir, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs,
    )

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, value, log_probs = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(
                action)
            done = terminated or truncated
            n_steps += 1
            score += reward
            agent.store_transition(
                observation, action, log_probs, value, reward, done
            )
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = score
            agent.save_models()

        print(f"episode {i}, score {score}, avg_score {avg_score:.3f}, time_steps {n_steps}, learning_steps {learn_iters}")
        
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
