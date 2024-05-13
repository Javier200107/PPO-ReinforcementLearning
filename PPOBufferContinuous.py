import numpy as np

class PPOBufferContinuous:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (np.array(self.states, dtype=np.float32), 
                np.array(self.actions, dtype=np.float32), 
                np.array(self.log_probs, dtype=np.float32),
                np.array(self.values, dtype=np.float32), 
                np.array(self.rewards, dtype=np.float32), 
                np.array(self.dones, dtype=np.bool_), 
                batches)

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.log_probs.append(np.array(log_prob, dtype=np.float32))
        self.values.append(np.array(value, dtype=np.float32))
        self.rewards.append(np.array(reward, dtype=np.float32))
        self.dones.append(bool(done))

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
