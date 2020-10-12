import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections


class ReplayBuffer(object):
    def __init__(self, max_len=100000):
        self.storage = collections.deque(maxlen=max_len)

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size=32):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s1, s1_length, a, r, s2, s2_length, done = [], [], [], [], [], [], []

        for i in ind:
            d1, d2, d3, d4, d5, d6, d7 = self.storage[i]
            s1.append(np.array(d1, copy=False))
            s1_length.append(np.array(d2, copy=False))
            a.append(np.array(d3, copy=False))
            r.append(np.array(d4, copy=False))
            s2.append(np.array(d5, copy=False))
            s2_length.append(np.array(d6, copy=False))
            done.append(np.array(d7, copy=False))

        return np.array(s1), np.array(s1_length), np.array(a), np.array(r), np.array(s2), np.array(s2_length), np.array(
            done)

    def get_size(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()


class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, hidden_size,
                 namescope='default', max_seq_length=32, base=2):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.namescope = namescope
        self.max_seq_length = max_seq_length
        self.base = base

        self.inp = tf.placeholder(shape=[None, self.max_seq_length, self.s_dim], dtype=tf.float64,
                                  name=self.namescope + '_inp')  # 输入state

        self.length = tf.placeholder(tf.int32, [None])  # 序列长度

        self.out, self.scaled_out = self.create_actor_network(self.namescope + 'main_actor')  # 输出动作，和输入的状态

        self.network_params = tf.trainable_variables()  # actor的所有数据

        self.target_out, self.target_scaled_out = self.create_actor_network(
            self.namescope + 'target_actor')  # 创建targetnetwork

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]  # 按照创建的顺序构建的

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

    def create_actor_network(self, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            s = self.inp
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden_size)
            _, net = tf.nn.dynamic_rnn(basic_cell, s, dtype=tf.float64, sequence_length=self.length)
            net = slim.fully_connected(net, self.hidden_size, activation_fn=tf.nn.relu,
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
            net = slim.fully_connected(net, self.hidden_size, activation_fn=tf.nn.relu)
            net = slim.fully_connected(net, int(self.a_dim), activation_fn=tf.nn.tanh)
            scaled_out = (net + 1) * self.action_bound  # action空间要变掉
            return net, scaled_out

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict(self, inputs, length):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inp: inputs, self.length: length
        })

    def predict_target(self, inputs, length):  # 预测
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.inp: inputs, self.length: length
        })


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, inp_actions, hidden_size, namescope='default',
                 max_seq_length=32):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.inp_actions = inp_actions
        self.hidden_size = hidden_size
        self.namescope = namescope
        self.max_seq_length = max_seq_length
        tf.set_random_seed(1)

        self.inp = tf.placeholder(shape=[None, self.max_seq_length, self.s_dim], dtype=tf.float64)
        self.action = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float64)
        self.length = tf.placeholder(dtype=tf.int32, shape=[None])

        self.total_out = self.create_critic_network(self.namescope + 'main_critic', self.inp_actions)
        self.out = self.create_critic_network(self.namescope + 'main_critic', self.action,
                                              reuse=True)  # 要重用里面的变量，所以要设为true,创建时参数一样

        self.target_out = self.create_critic_network(self.namescope + 'target_critic', self.action)

        self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.namescope + 'main_critic')
        self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       self.namescope + 'target_critic')  # 得到target的variables
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float64, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.out - self.predicted_q_value))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.network_params)

    def create_critic_network(self, scope, actions, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden_size)
            _, states = tf.nn.dynamic_rnn(basic_cell, self.inp, dtype=tf.float64, sequence_length=self.length)
            input_s = tf.reshape(states, [-1, self.hidden_size])
            net = tf.concat([input_s, actions], axis=1)
            net = slim.fully_connected(net, self.hidden_size)
            net = slim.fully_connected(net, self.hidden_size)
            net = slim.fully_connected(net, 1, activation_fn=None)
        return net

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict(self, inputs, length, action):
        return self.sess.run(self.out, feed_dict={
            self.inp: inputs,
            self.action: action,
            self.length: length
        })

    def predict_target(self, inputs, length, action):
        return self.sess.run(self.target_out, feed_dict={
            self.inp: inputs,
            self.action: action,
            self.length: length
        })


class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound=1, discount_factor=0.99,
                 seed=1, actor_lr=1e-3, critic_lr=1e-3, batch_size=32, namescope='default',
                 tau=0.005, policy_noise=0.1, noise_clip=0.5, hidden_size=64, max_seq_length=32,
                 memory_capacity=100000):
        np.random.seed(int(seed))
        tf.set_random_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.actor = Actor(self.sess, state_dim, action_dim, action_bound,
                           actor_lr, tau, int(batch_size), self.hidden_size, namescope=namescope + str(seed),
                           max_seq_length=max_seq_length)
        self.critic = Critic(self.sess, state_dim, action_dim, critic_lr, tau,
                             self.actor.scaled_out, self.hidden_size, namescope=namescope + str(seed),
                             max_seq_length=max_seq_length)
        self.actor_loss = -tf.reduce_mean(self.critic.total_out)
        self.actor_train_step = tf.train.AdamOptimizer(actor_lr).minimize(self.actor_loss,
                                                                          var_list=self.actor.network_params)
        self.action_bound = action_bound
        self.sess.run(tf.global_variables_initializer())
        self.replay_buffer = ReplayBuffer(memory_capacity)

    def train(self, iterations):
        actor_loss, critic_loss = [], []
        for i in range(iterations):
            s1, s1_length, a, r, s2, s2_length, done = self.replay_buffer.sample(self.batch_size)
            temp_action = np.reshape(self.actor.predict_target(s2, s2_length), [self.batch_size, -1])  # 要加入length参数
            next_action = temp_action
            target_q = self.critic.predict_target(s2, s2_length, next_action)

            y_i = np.reshape(r, [self.batch_size, 1]) + (1 - np.reshape(done, [self.batch_size,
                                                                               1])) * self.discount_factor * np.reshape(
                target_q, [self.batch_size, 1])

            # 需要加入length
            c_loss, _ = self.sess.run([self.critic.loss, self.critic.train_step],
                                      feed_dict={self.critic.inp: s1,
                                                 self.critic.length: s1_length,
                                                 self.critic.action: np.reshape(
                                                     a, [self.batch_size,
                                                         int(self.action_dim)]),
                                                 self.critic.predicted_q_value: np.reshape(
                                                     y_i, [-1, 1])})
            a_loss, _ = self.sess.run([self.actor_loss, self.actor_train_step],
                                      feed_dict={self.actor.inp: s1, self.actor.length: s1_length,
                                                 self.critic.inp: s1, self.critic.length: s1_length})
            self.actor.update_target_network()
            self.critic.update_target_network()
            actor_loss.append(a_loss)
            critic_loss.append(c_loss)
        return np.mean(actor_loss), np.mean(critic_loss)

    def get_action(self, s, length):
        s = np.array(s).reshape([length, self.state_dim])
        if length < self.max_seq_length:  # 补0
            padding_mat = np.zeros([self.max_seq_length - length, self.state_dim])
            s = np.vstack((s, padding_mat))
        return self.actor.predict(np.reshape(s, [-1,self.max_seq_length, self.actor.s_dim]), [length])

    def store(self, s1, s1_length, a, r, s2, s2_length, done):  # 存储的时候需要存储length
        # 需要对state进行padding
        s1 = np.array(s1).reshape([s1_length, self.state_dim])
        s2 = np.array(s2).reshape([s2_length, self.state_dim])
        if s1_length < self.max_seq_length:  # 补0
            padding_mat = np.zeros([self.max_seq_length - s1_length, self.state_dim])
            s1 = np.vstack((s1, padding_mat))
        if s2_length < self.max_seq_length:
            padding_mat = np.zeros([self.max_seq_length - s2_length, self.state_dim])
            s2 = np.vstack((s2, padding_mat))

        self.replay_buffer.add((s1, s1_length, a, r, s2, s2_length, done))

    def get_params(self):
        return self.sess.run(self.actor.network_params[:len(self.actor.network_params)])

    def eval_critic(self, s, s_length, a):
        s = np.array(s).reshape([-1, self.max_seq_length, self.state_dim])
        s_length = np.array(s_length).flatten()
        a = np.reshape(a, [-1, int(self.action_dim)])
        return self.critic.predict(s, s_length, a)
