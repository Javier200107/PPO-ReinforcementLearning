from tensorflow.keras import layers, Model

class ActorNetworkContinuous(Model):
    """Actor network for continuous action spaces."""
    
    def __init__(self, n_actions, input_dims, **kwargs) -> None:
        super(ActorNetworkContinuous, self).__init__(**kwargs)
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.dense1 = layers.Dense(32, activation='relu', input_shape=(input_dims,))
        self.dense2 = layers.Dense(32, activation='relu')
        self.mu = layers.Dense(n_actions)  # Produces means
        self.sigma = layers.Dense(n_actions, activation='softplus')  # Produces standard deviations

    def call(self, inputs):
        """Forward pass."""
        x = self.dense1(inputs)
        x = self.dense2(x)
        mu = self.mu(x)
        sigma = self.sigma(x) + 1e-6  # Ensure non-zero standard deviation
        return mu, sigma

    def get_config(self):
        config = super(ActorNetworkContinuous, self).get_config()
        config.update({
            'n_actions': self.n_actions,
            'input_dims': self.input_dims
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CriticNetworkContinuous(Model):
    """Critic network for continuous action spaces."""
    def __init__(self, input_dims) -> None:
        super(CriticNetworkContinuous, self).__init__()
        self.input_dims = input_dims
        self.dense1 = layers.Dense(32, activation='relu', input_shape=(input_dims,))
        self.dense2 = layers.Dense(32, activation='relu')
        self.dense3 = layers.Dense(16, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, state):
        """Forward pass."""
        value = self.dense1(state)
        value = self.dense2(value)
        value = self.dense3(value)
        return self.output_layer(value)
    
    def get_config(self):
        config = super(CriticNetworkContinuous, self).get_config()
        config.update({
            'input_dims': self.input_dims
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)