import tensorflow as tf
from keras.optimizers import Adam
import keras
from PPONetworks import ActorNetwork, CriticNetwork
import tensorflow_probability as tfp
from PPOBuffer import PPOBuffer
import numpy as np

class PPOAgent:
    """Proximal Policy Optimization Agent using TensorFlow 2.0.0 and Keras."""

    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                policy_clip=0.2, batch_size=64, n_epochs=10, checkpoint_directory="/models") -> None:
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_directory = checkpoint_directory
        
        self.actor = ActorNetwork(n_actions)
        self.critic = CriticNetwork()
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))

        self.buffer = PPOBuffer(batch_size)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        probabilities = self.actor(state)
        action_probs = tfp.distributions.Categorical(probs=probabilities)
        action = action_probs.sample() # Sample an action from the distribution
        log_probs = action_probs.log_prob(action) # Calculate the log probability of the action
        value = self.critic(state) # Calculate the value of the state

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_probs = log_probs.numpy()[0]

        return action, value, log_probs
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        self.buffer.store(state, action, log_prob, value, reward, done)
    
    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, log_probs_old, values, rewards, dones, batches = self.buffer.generate_batches()

            advantage = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    print(dones[k])
                    print(type(dones[k]))
                    a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - dones[k]) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(states[batch])
                    actions = tf.convert_to_tensor(actions[batch])
                    log_probs_old = tf.convert_to_tensor(log_probs_old[batch])

                    probs = self.actor(states)
                    new_log_probs = tfp.distributions.Categorical(probs=probs).log_prob(actions)

                    critic_value = tf.squeeze(self.critic(states), 1)
                    prob_rato = tf.math.exp(new_log_probs - log_probs_old)

                    weighted_probs = prob_rato * advantage[batch]
                    clipped_probs = tf.clip_by_value(prob_rato, 1 - self.policy_clip, 1 + self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]

                    actor_loss = -tf.reduce_mean(tf.math.minimum(weighted_probs, weighted_clipped_probs))
                    actor_loss = tf.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = keras.losses.MSE(critic_value, returns)
                
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables
                
                actor_gradients = tape.gradient(actor_loss, actor_params)
                critic_gradients = tape.gradient(critic_loss, critic_params)

                self.actor.optimizer.apply_gradients(zip(actor_gradients, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_gradients, critic_params))
            
            self.buffer.clear()
                
    def save_models(self):
        print("... saving models ...")
        self.actor.save(self.checkpoint_directory + "/actor") # Save the whole actor model
        self.critic.save(self.checkpoint_directory + "/critic") # Save the whole critic model

    def load_models(self):
        print("... loading models ...")
        self.actor = keras.models.load_model(self.checkpoint_directory + "/actor")
        self.critic = keras.models.load_model(self.checkpoint_directory + "/critic")