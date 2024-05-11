from keras import layers, Model


class ActorNetwork(Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256) -> None:
        super(ActorNetwork, self).__init__()
        self.dense1 = layers.Dense(fc1_dims, activation='relu')
        self.dense2 = layers.Dense(fc2_dims, activation='relu')
        self.output_layer = layers.Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        
        return self.output_layer(x)


class CriticNetwork(Model):
    def __init__(self, fc1_dims=256, fc2_dims=256) -> None:
        super(CriticNetwork, self).__init__()
        self.dense1 = layers.Dense(fc1_dims, activation='relu')
        self.dense2 = layers.Dense(fc2_dims, activation='relu')
        self.output_layer = layers.Dense(1, activation=None)

    def call(self, state):
        value = self.dense1(state)
        value = self.dense2(value)
        
        return self.output_layer(value)
