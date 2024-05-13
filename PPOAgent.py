import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras
from PPONetworksDiscrete import ActorNetworkDiscrete, CriticNetworkDiscrete
from PPONetworksContinuous import ActorNetworkContinuous, CriticNetworkContinuous
import tensorflow_probability as tfp
from PPOBufferDiscrete import PPOBufferDiscrete
from PPOBufferContinuous import PPOBufferContinuous
import numpy as np

class PPOAgent:
    """Proximal Policy Optimization Agent using TensorFlow 2.0.0 and Keras."""

    def __init__(self, n_actions, input_dims, checkpoint_directory, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                policy_clip=0.2, n_epochs=10) -> None:
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.checkpoint_directory = checkpoint_directory

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
                    a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - dones[k]) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    batch_indices = tf.convert_to_tensor(batch, dtype=tf.int32)
                    # print("states[batch]: ", states[batch])
                    # print("actions[batch]: ", actions[batch])
                    # print("log_probs_old[batch]: ", log_probs_old[batch])
                    # Safely extract tensors for the current batch
                    batch_states = tf.gather(states, batch_indices)
                    batch_actions = tf.gather(actions, batch_indices)
                    batch_log_probs_old = tf.gather(log_probs_old, batch_indices)

                    actor_loss, critic_loss = self.update_networks(batch_states, batch_actions, batch_log_probs_old, advantage, batch, values)
                
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables
                
                actor_gradients = tape.gradient(actor_loss, actor_params)
                critic_gradients = tape.gradient(critic_loss, critic_params)

                self.actor.optimizer.apply_gradients(zip(actor_gradients, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_gradients, critic_params))
            
            self.buffer.clear()
                
    def save_models(self):
        print("... saving models ...")
        self.actor.save(self.checkpoint_directory + self.env_type + "/" + self.env_name + "/actor") # Save the whole actor model
        self.critic.save(self.checkpoint_directory + self.env_type + "/" + self.env_name + "/critic") # Save the whole critic model

    def load_models(self):
        print("... loading models ...")
        self.actor = keras.models.load_model(self.checkpoint_directory + "/" + self.env_type + self.env_name + "/actor")
        self.critic = keras.models.load_model(self.checkpoint_directory + "/" + self.env_type + self.env_name + "/critic")

class PPOAgentDiscrete(PPOAgent):
    
    def __init__(self, n_actions, input_dims, checkpoint_directory, env_name, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                policy_clip=0.2, batch_size=64, n_epochs=10):
        super(PPOAgentDiscrete, self).__init__(n_actions, input_dims, checkpoint_directory, gamma, alpha, gae_lambda,
                policy_clip, n_epochs)
        self.env_type = 'discrete'
        self.env_name = env_name

        self.actor = ActorNetworkDiscrete(n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.critic = CriticNetworkDiscrete(input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))

        self.buffer = PPOBufferDiscrete(batch_size=batch_size)

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

    def update_networks(self, batch_states, batch_actions, batch_log_probs_old, advantage, batch, values):
        probs = self.actor(batch_states)
        new_log_probs = tfp.distributions.Categorical(probs=probs).log_prob(batch_actions)

        critic_value = tf.squeeze(self.critic(batch_states), 1)
        prob_rato = tf.math.exp(new_log_probs - batch_log_probs_old)

        weighted_probs = prob_rato * advantage[batch]
        clipped_probs = tf.clip_by_value(prob_rato, 1 - self.policy_clip, 1 + self.policy_clip)
        weighted_clipped_probs = clipped_probs * advantage[batch]

        actor_loss = -tf.reduce_mean(tf.math.minimum(weighted_probs, weighted_clipped_probs))
        actor_loss = tf.reduce_mean(actor_loss)

        returns = advantage[batch] + values[batch]
        critic_loss = keras.losses.MSE(critic_value, returns)

        return actor_loss, critic_loss

class PPOAgentContinuous(PPOAgent):
    
    def __init__(self, n_actions, input_dims, checkpoint_directory, env_name, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                policy_clip=0.2, batch_size=64, n_epochs=10):
        super(PPOAgentContinuous, self).__init__(n_actions, input_dims, checkpoint_directory, gamma, alpha, gae_lambda,
                policy_clip, n_epochs)
        self.env_type = 'continuous'
        self.env_name = env_name

        self.actor = ActorNetworkContinuous(n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.critic = CriticNetworkContinuous(input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))

        self.buffer = PPOBufferContinuous(batch_size=batch_size)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu, sigma = self.actor(state)
        dist = tfp.distributions.Normal(mu, sigma)
        action = dist.sample()  # Sample an action from the normal distribution
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        return action.numpy()[0], value.numpy()[0], log_prob.numpy()[0]
    
    def update_networks(self, batch_states, batch_actions, batch_log_probs_old, advantage, batch, values):
        mu, sigma = self.actor(batch_states)  # Obtain mean and standard deviation for actions
        dist = tfp.distributions.Normal(mu, sigma)
        new_log_probs = dist.log_prob(batch_actions)

        critic_value = tf.squeeze(self.critic(batch_states), 1)  # Get value estimate from critic
        prob_ratio = tf.exp(new_log_probs - batch_log_probs_old)  # Calculate the probability ratio

        # Apply the PPO clipped objective function
        advantage_batch_expanded = tf.expand_dims(advantage[batch], axis=-1)
        weighted_probs = prob_ratio * advantage_batch_expanded
        clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
        weighted_clipped_probs = clipped_probs * advantage_batch_expanded

        # Actor loss calculation using the minimum of unclipped and clipped values
        actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_clipped_probs))

        # Calculate the returns and the critic loss
        returns = advantage[batch] + values[batch]  # returns calculation might need adjustment based on your setup
        critic_loss = keras.losses.MSE(critic_value, returns)

        return actor_loss, critic_loss