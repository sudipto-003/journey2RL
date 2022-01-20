import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, hd1=512, hd2=512):
        super(Critic, self).__init__()
        self.h_layer1 = tf.keras.layers.Dense(hd1, activation='relu')
        self.h_layer2 = tf.keras.layers.Dense(hd2, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        a_q = self.h_layer1(tf.concat([state, action], axis=1))
        a_q = self.h_layer2(a_q)
        q = self.q(a_q)

        return q


class Actor(tf.keras.Model):
    def __init__(self, n_actions, hd1=512, hd2=512):
        super(Actor, self).__init__()
        self.h_layer1 = tf.keras.layers.Dense(hd1, activation='relu')
        self.h_layer2 = tf.keras.layers.Dense(hd2, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation='tanh')

    def call(self, state):
        prob = self.h_layer1(state)
        prob = self.h_layer2(prob)
        mu = self.mu(prob)

        return mu