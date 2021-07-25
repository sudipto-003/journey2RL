import numpy as np
import tensorflow as tf


class NN(tf.keras.Model):
    def __init__(self, output_dim, units1=32, units2=32):
        super(NN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(units1, activation='relu')
        self.layer2 = tf.keras.layers.Dense(units2, activation='relu')
        self.layer3 = tf.keras.layers.Dense(output_dim, dtype=np.float32)

    def call(self, x):
        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)

        return z3

    
class DQN:
    def __init__(self, num_actions, hunits1=32, hunits2=32, discount=0.99):
        self.q_nn = NN(num_actions, hunits1, hunits2)
        self.q_target = NN(num_actions, hunits1, hunits2)
        self.discount = discount
        self.q_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.num_actions = num_actions

    @tf.function
    def train_dqn(self, states, actions, rewards, next_states, terminals):
        target_qas = self.q_target(next_states)
        max_target_qas = tf.reduce_max(target_qas, axis=-1)
        td_target = rewards + (1. - terminals) * self.discount * max_target_qas
        
        with tf.GradientTape() as tape:
            qas = self.q_nn(states)
            action_masks = tf.one_hot(actions, self.num_actions)
            masked_qs = tf.reduce_sum(action_masks * qas, axis=-1)
            loss = self.loss(td_target, masked_qs)

        gradient = tape.gradient(loss, self.q_nn.trainable_variables)
        self.q_optimizer.apply_gradients(zip(gradient, self.q_nn.trainable_variables))

        return loss

    def copy_weights_target_network(self):
        self.q_target.set_weights(self.q_nn.get_weights())

    def chose_action(self, state, epsilon):
        toss = tf.random.uniform((1, ))
        if toss < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return tf.argmax(self.q_nn(state)[0]).numpy()
