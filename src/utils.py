import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curve(x, scores, plots_dir, combo=None):
    """Plot the learning curve of the agent."""

    # Start new figure
    plt.figure()

    # Calculate running average and running standard deviation of scores
    window_size = 20
    running_avg = np.zeros(len(scores))
    running_std = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-window_size):(i+1)])
        running_std[i] = np.std(scores[max(0, i-window_size):(i+1)])

    # Use seaborn to plot the running average
    sns.set_theme(style="whitegrid")
    sns.lineplot(x=x, y=running_avg, color="blue")

    # Add the shaded area for standard deviation
    plt.fill_between(x, running_avg - running_std, running_avg + running_std, color='blue', alpha=0.3)

    # Add title and labels
    if combo is not None:
        plt.title(f'Running average of previous {window_size} scores:\n {combo}', fontsize=10)
    else:
        plt.title(f'Running average of previous {window_size} scores')
    plt.xlabel('Episodes')
    plt.ylabel('Score')

    # Save the plot
    plt.savefig(plots_dir + 'running_avg.png')
    plt.close()

