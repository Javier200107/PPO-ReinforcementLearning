import numpy as np

class PPOBuffer:
    """A class to store the experiences of the agent in the environment."""

    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        """Generate batches of experiences."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
                np.array(self.values), np.array(self.rewards), np.array(self.dones), batches

    def store(self, state, action, log_prob, value, reward, done):
        """Store experiences in the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        