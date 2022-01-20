import tensorflow as tf
from models import Actor, Critic
from buffer import MemoryBuffer

class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None, gamma=0.99,
                n_actions=2, max_size=1000000, tau=0.005, hd1=400, hd2=300, 
                batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.memory = MemoryBuffer(max_size)
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = Actor(n_actions=n_actions)
        self.critic = Critic()
        self.target_actor = Actor(n_actions=n_actions)
        self.target_critic = Critic()

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))

        self.update_weights()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def train(self):
        cl, al = self.learn()
        if cl is not None:
            self.update_weights()

        return cl, al

    def update_weights(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)

        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    # @tf.function
    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, done = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            critic_value_ = tf.squeeze(self.target_critic(next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_gradient, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
            
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_gradient, self.actor.trainable_variables)
        )

        self.update_weights()
        