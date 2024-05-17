from tensorflow.keras import layers, Model

class ActorNetworkDiscrete(Model):
    """Class for the Actor network for discrete action spaces."""
    def __init__(self, n_actions, input_dims, **kwargs) -> None:
        super(ActorNetworkDiscrete, self).__init__(**kwargs)
        self.n_actions = n_actions  # Number of actions, i.e. output dimension
        self.input_dims = input_dims # Input dimension, i.e. state space dimension
        self.dense1 = layers.Dense(32, activation='relu', input_shape=(input_dims,))
        self.dense2 = layers.Dense(16, activation='relu')
        self.output_layer = layers.Dense(n_actions, activation='softmax') 

    def call(self, state):
        """Forward pass of the network."""
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x) # Output is a probability distribution over actions
    
    def get_config(self):
        config = super(ActorNetworkDiscrete, self).get_config()
        config.update({
            'n_actions': self.n_actions,
            'input_dims': self.input_dims,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CriticNetworkDiscrete(Model):
    """Class for the Critic network for discrete action spaces."""
    def __init__(self, input_dims) -> None:
        super(CriticNetworkDiscrete, self).__init__()
        self.input_dims = input_dims
        self.dense1 = layers.Dense(32, activation='relu', input_shape=(input_dims,))
        self.dense2 = layers.Dense(16, activation='relu')
        self.dense3 = layers.Dense(16, activation='relu')
        self.output_layer = layers.Dense(1, activation=None)

    def call(self, state):
        """Forward pass of the network."""
        value = self.dense1(state)
        value = self.dense2(value)
        value = self.dense3(value)
        return self.output_layer(value) # Output is the value of the state

    def get_config(self):
        config = super(CriticNetworkDiscrete, self).get_config()
        config.update({
            'input_dims': self.input_dims
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)