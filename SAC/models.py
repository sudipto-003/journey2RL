import tensorflow as tf
import tensorflow_probability as tfp

MAX_LOG_STD = 20
MIN_LOG_STD = -2

class Actor(tf.keras.Model):
    def __init__(self, action_dim, lopprob_epsilon, hidden_unit=256):
        super(Actor, self).__init__()
        self.logprob_epsilon = lopprob_epsilon
        self.h1 = tf.keras.layers.Dense(hidden_unit, activation='relu')
        self.h2 = tf.keras.layers.Dense(hidden_unit, activation='relu')
        self.mean = tf.keras.layers.Dense(action_dim)
        self.log_std = tf.keras.layers.Dense(action_dim)

    @tf.function
    def call(self, state):
        z = self.h2(self.h1(state))
        mean = self.mean(z)
        log_std = self.log_std(z)
        log_std_clipped = tf.clip_by_value(log_std, MAX_LOG_STD, MIN_LOG_STD)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_action = tf.tanh(action)
        log_prob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_action, 2) + self.logprob_epsilon)
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)

        return squashed_action, log_prob


class Critic(tf.keras.Model):
    def __init__(self, hidden_unit=256):
        super(Critic, self).__init__()
        self.h1 = tf.keras.layers.Dense(hidden_unit, activation='relu')
        self.h2 = tf.keras.layers.Dense(hidden_unit, activation='relu')
        self.qvalue = tf.keras.layers.Dense(1, activation=None)

    @tf.function
    def call(self, state, action):
        s_a = tf.concat([state, action], 1)
        return self.qvalue(self.h2(self.h1(s_a)))
