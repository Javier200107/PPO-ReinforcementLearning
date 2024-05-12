from tensorflow.keras import layers, Model


class ActorNetwork(Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, **kwargs) -> None:
        super(ActorNetwork, self).__init__(**kwargs)
        self.n_actions = n_actions  # Ensure this attribute is initialized
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.dense1 = layers.Dense(fc1_dims, activation='relu')
        self.dense2 = layers.Dense(fc2_dims, activation='relu')
        self.output_layer = layers.Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(ActorNetwork, self).get_config()
        config.update({
            'n_actions': self.n_actions,
            'fc1_dims': self.fc1_dims,
            'fc2_dims': self.fc2_dims
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CriticNetwork(Model):
    def __init__(self, fc1_dims=256, fc2_dims=256) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.dense1 = layers.Dense(fc1_dims, activation='relu')
        self.dense2 = layers.Dense(fc2_dims, activation='relu')
        self.output_layer = layers.Dense(1, activation=None)

    def call(self, state):
        value = self.dense1(state)
        value = self.dense2(value)
        
        return self.output_layer(value)

    def get_config(self):
        config = super(CriticNetwork, self).get_config()
        config.update({
            'fc1_dims': self.fc1_dims,
            'fc2_dims': self.fc2_dims
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)