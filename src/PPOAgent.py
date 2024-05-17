import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras
from src.PPONetworksDiscrete import ActorNetworkDiscrete, CriticNetworkDiscrete
from src.PPONetworksContinuous import ActorNetworkContinuous, CriticNetworkContinuous
import tensorflow_probability as tfp
from src.PPOBuffer import PPOBuffer
import numpy as np

class PPOAgent:
    """Proximal Policy Optimization Agent using TensorFlow 2.12.1 and Keras."""

    def __init__(self, n_actions, input_dims, checkpoint_directory, gamma, gae_lambda, n_epochs, seed) -> None:
        """Initializes the PPO agent."""
        self.n_actions = n_actions # Number of actions
        self.input_dims = input_dims # Observation space
        self.gamma = gamma # Discount factor
        self.gae_lambda = gae_lambda # GAE lambda, used to calculate the advantage
        self.n_epochs = n_epochs # Number of epochs to train the agent
        self.checkpoint_directory = checkpoint_directory # Directory to save the models

        self.seed = np.random.seed(seed)
        self.tf_seed = tf.random.set_seed(seed)

        self.actor_loss_history = [] # List to store actor losses
        self.critic_loss_history = [] # List to store critic losses

    def store_transition(self, state, action, log_prob, value, reward, done):
        """Uses the buffer to store the state, action, log probability, value, reward, and done flag."""
        self.buffer.store(state, action, log_prob, value, reward, done)
    
    def learn(self):
        """
        Trains the agent using the stored transitions. It performs the following steps:
        1. Generate batches of transitions
        2. Calculate the advantage using GAE
        3. Update the networks using the batches and the advantage
        4. Clear the buffer
        """
        for _ in range(self.n_epochs): 
            states, actions, log_probs_old, values, rewards, dones, batches = self.buffer.generate_batches() 

            advantage = np.zeros(len(rewards), dtype=np.float32) # Initialize the advantage array

            for t in range(len(rewards) - 1): # Calculate the advantage using GAE
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1): 
                    a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - dones[k]) - values[k]) 
                    discount *= self.gamma * self.gae_lambda 
                advantage[t] = a_t

            for batch in batches: 
                with tf.GradientTape(persistent=True) as tape: # Use gradient tape to calculate the gradients
                    batch_indices = tf.convert_to_tensor(batch, dtype=tf.int32)
                    batch_states = tf.gather(states, batch_indices)
                    batch_actions = tf.gather(actions, batch_indices)
                    batch_log_probs_old = tf.gather(log_probs_old, batch_indices)

                    actor_loss, critic_loss = self.update_networks(batch_states, batch_actions, batch_log_probs_old, advantage, batch, values) 
                
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables
                
                actor_gradients = tape.gradient(actor_loss, actor_params)
                critic_gradients = tape.gradient(critic_loss, critic_params)

                #actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, 50.0)
                #critic_gradients, _ = tf.clip_by_global_norm(critic_gradients, 50.0)

                self.actor.optimizer.apply_gradients(zip(actor_gradients, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_gradients, critic_params))

                # Store losses for plotting
                self.actor_loss_history.append(actor_loss.numpy())
                self.critic_loss_history.append(critic_loss.numpy())
            
            self.buffer.clear()
                
    def save_models(self):
        print("... saving models ...")
        self.actor.save(self.checkpoint_directory + "/actor") # Save the whole actor model
        self.critic.save(self.checkpoint_directory + "/critic") # Save the whole critic model

    def load_models(self, checkpoint_directory, env_type, env_name):
        print("... loading models ...")
        self.actor = keras.models.load_model(checkpoint_directory + "/" + env_type + "/" + env_name + "/actor")
        self.critic = keras.models.load_model(checkpoint_directory + "/" + env_type + "/" + env_name + "/critic")

class PPOAgentDiscrete(PPOAgent):
    """Proximal Policy Optimization Agent for discrete action spaces."""
    def __init__(self, n_actions, input_dims, checkpoint_directory, env_name, gamma=0.99, actorn_lr=0.0005, critic_lr=0.001, gae_lambda=0.95,
                clip_ratio=0.2, c1=0.5, c2=0.0, batch_size=64, n_epochs=10, seed=0):
        super(PPOAgentDiscrete, self).__init__(n_actions=n_actions, input_dims=input_dims, checkpoint_directory=checkpoint_directory, gamma=gamma, 
                                                 gae_lambda=gae_lambda, n_epochs=n_epochs, seed=seed)
        self.env_type = 'discrete'
        self.env_name = env_name

        self.clip_ratio = clip_ratio
        self.c1 = c1
        self.c2 = c2

        self.actor = ActorNetworkDiscrete(n_actions=n_actions, input_dims=input_dims)
        self.critic = CriticNetworkDiscrete(input_dims=input_dims)

        self.actor.compile(optimizer=Adam(learning_rate=actorn_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.buffer = PPOBuffer(batch_size=batch_size)

    def choose_action(self, observation):
        """Choose an action using the actor network."""
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
        """Update the actor and critic networks using the PPO clipped objective function."""
        probs = self.actor(batch_states)
        dist = tfp.distributions.Categorical(probs=probs)
        new_log_probs = dist.log_prob(batch_actions)
        dist_entropy = dist.entropy()

        critic_value = tf.squeeze(self.critic(batch_states), 1)
        prob_ratio = tf.exp(new_log_probs - batch_log_probs_old)

        weighted_probs = prob_ratio * advantage[batch]
        clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        weighted_clipped_probs = clipped_probs * advantage[batch]

        actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_clipped_probs)) - self.c2 * tf.reduce_mean(dist_entropy)
        returns = advantage[batch] + values[batch]
        critic_loss = self.c1 * keras.losses.MSE(critic_value, returns)

        return actor_loss, critic_loss

class PPOAgentContinuous(PPOAgent):
    """Proximal Policy Optimization Agent for continuous action spaces."""
    
    def __init__(self, n_actions, input_dims, checkpoint_directory, env_name, gamma=0.99, actorn_lr=0.0005, critic_lr=0.001, gae_lambda=0.95,
                clip_ratio=0.2, c1=0.5, c2=0.0, batch_size=64, n_epochs=10, seed=0):
        super(PPOAgentContinuous, self).__init__(n_actions=n_actions, input_dims=input_dims, checkpoint_directory=checkpoint_directory, gamma=gamma, 
                                                 gae_lambda=gae_lambda, n_epochs=n_epochs, seed=seed)
        self.env_type = 'continuous'
        self.env_name = env_name

        self.clip_ratio = clip_ratio
        self.c1 = c1
        self.c2 = c2

        self.actor = ActorNetworkContinuous(n_actions=n_actions, input_dims=input_dims)
        self.critic = CriticNetworkContinuous(input_dims=input_dims)

        self.actor.compile(optimizer=Adam(learning_rate=actorn_lr))
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.buffer = PPOBuffer(batch_size=batch_size)

    def choose_action(self, observation):
        """Choose an action using the actor network."""
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu, sigma = self.actor(state)
        dist = tfp.distributions.Normal(mu, sigma)
        action = dist.sample()  # Sample an action from the normal distribution
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        return action.numpy()[0], value.numpy()[0], log_prob.numpy()[0]
    
    def update_networks(self, batch_states, batch_actions, batch_log_probs_old, advantage, batch, values):
        """Update the actor and critic networks using the PPO clipped objective function."""
        mu, sigma = self.actor(batch_states)  # Obtain mean and standard deviation for actions
        dist = tfp.distributions.Normal(mu, sigma)
        new_log_probs = dist.log_prob(batch_actions)
        dist_entropy = dist.entropy()  # Calculate entropy of the action distribution

        critic_value = tf.squeeze(self.critic(batch_states), 1)  # Get value estimate from critic
        prob_ratio = tf.exp(new_log_probs - batch_log_probs_old)  # Calculate the probability ratio

        # Apply the PPO clipped objective function
        advantage_batch_expanded = tf.expand_dims(advantage[batch], axis=-1)
        weighted_probs = prob_ratio * advantage_batch_expanded
        clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        weighted_clipped_probs = clipped_probs * advantage_batch_expanded

        # Actor loss calculation using the minimum of unclipped and clipped values
        # and adding entropy to encourage exploration
        actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_clipped_probs)) - self.c2 * tf.reduce_mean(dist_entropy)

        # Calculate the returns and the critic loss
        returns = advantage[batch] + values[batch]
        critic_loss = self.c1 * keras.losses.MSE(critic_value, returns)  # Multiply MSE loss by c1

        return actor_loss, critic_loss
