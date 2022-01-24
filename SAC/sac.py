import tensorflow as tf
from models import *
from buffer import MemoryBuffer

def update_weights(source, target, coef):
    for s, t in zip(source, target):
        t.assign((1.0 - coef) * t + coef * s)

class SAC:
    def __init__(self, action_dim, hidden_unit=256, alpha=1e-3, lr=3e-3, 
                polyak_coef=1e-2, env=None, gamma=99e-2, logprob_epsilon=1e-6,
                batch_size=64, trajectory_len=100000):
        
        self.action_dim = action_dim
        self.trajectory = MemoryBuffer(trajectory_len)
        self.batch_size = batch_size
        self.polyak_coef = polyak_coef
        self.gamma = gamma
        self.alpha = alpha

        self.actor = Actor(action_dim, logprob_epsilon, hidden_unit)
        self.q_net1 = Critic(hidden_unit)
        self.q_net2 = Critic(hidden_unit)
        self.q_net_target1 = Critic(hidden_unit)
        self.q_net_target2 = Critic(hidden_unit)

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        self.q_net1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        self.q_net2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        self.q_net_target1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        self.q_net_target2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        update_weights(self.q_net1.variables, self.q_net_target1.variables, self.polyak_coef)
        update_weights(self.q_net2.variables, self.q_net_target2.variables, self.polyak_coef)

    @tf.function
    def learn(self):
        if len(self.trajectory) < self.batch_size:
            return

        states, actions, next_states, rewards, dones = self.trajectory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        #learning the q values and q targets
        target_actions, action_logprob = self.actor(next_states)

        q_value1 = self.q_net_target1(next_states, target_actions)
        q_value2 = self.q_net_target2(next_states, target_actions)
        next_values = tf.math.minimum(q_value1, q_value2) - self.alpha * action_logprob
        q_target = rewards + self.gamma * (1 - dones) * next_values
        q_target = tf.reshape(q_target, [self.batch_size, 1])

        with tf.GradientTape() as q1_tape:
            q1 = self.q_net1(states, actions)
            q1_loss = tf.keras.losses.MSE(q1, q_target)

        with tf.GradientTape() as q2_tape:
            q2 = self.q_net2(states, actions)
            q2_loss = tf.keras.losses.MSE(q2, q_target)

        q1_grads = q1_tape.gradient(q1_loss, self.q_net1.trainable_variables)
        self.q_net1.optimizer.apply_gradients(zip(q1_grads, self.q_net1.trainable_variables))

        q2_grads = q2_tape.gradient(q2_loss, self.q_net2.trainable_variables)
        self.q_net2.optimizer.apply_gradients(zip(q2_grads, self.q_net2.trainable_variables))

        #learning the policy
        with tf.GradientTape() as actor_tape:
            actions_, actions_logprob = self.actor(states)
            action_q = tf.math.minimum(self.q_net1(states, actions_), self.q_net2(states, actions))

            advantage = tf.stop_gradient(actions_logprob - action_q)
            actor_loss = tf.reduce_mean(advantage * actions_logprob)

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        update_weights(self.q_net1.variables, self.q_net_target1.variables, self.polyak_coef)
        update_weights(self.q_net2.variables, self.q_net_target2.variables, self.polyak_coef)

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        return self.actor(state)[0][0]