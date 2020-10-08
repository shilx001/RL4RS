import numpy as np
import tensorflow as tf
import collections

################## hyper parameters ##################

LR_A = 0.001
LR_C = 0.0001
GAMMA = 0.99
TAU = 0.001
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 32
HIDDEN_SIZE = 64
REPLAY_START = 100
STD_DEV = 0.01
tf.set_random_seed(1)


################## DDPG algorithm ##################

class DDPG(object):
    def __init__(self, a_dim, s_dim, max_seq_length, a_bound):
        '''
        :param a_dim: action dimension
        :param s_dim: state dimension
        :param max_seq_length: maximum sequence length for RNN
        :param a_bound: action bound
        '''
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)  # use deque to store transitions.
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound, self.max_seq_length = a_dim, s_dim, a_bound, max_seq_length
        self.replay_start = REPLAY_START
        self.S1 = tf.placeholder(tf.float64, [None, self.max_seq_length, self.s_dim])
        self.S1_length = tf.placeholder(tf.int32, [None])
        self.S2 = tf.placeholder(tf.float64, [None, self.max_seq_length, self.s_dim])
        self.S2_length = tf.placeholder(tf.int32, [None])

        self.R = tf.placeholder(tf.float64, [None, ], name='reward')
        self.output_action = self._build_a(self.S1, self.S1_length)  # 输出action,不是实际的action
        self.done = tf.placeholder(tf.float64, [None, ], name='done')
        self.a = tf.placeholder(tf.float64, [None, self.a_dim])
        q = self._build_c(self.S1, self.S1_length, self.output_action)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        a_ = self._build_a(self.S2, self.S2_length, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S2, self.S2_length, a_, reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q, 这里我改为reduce_sum
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            self.predict_q = self._build_c(self.S1, self.S1_length, self.a, reuse=True)
            q_target = self.R + (1 - self.done) * GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.predict_q)
            self.c_loss = td_error
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, length):
        s = np.array(s).reshape([length, self.s_dim])
        if length < self.max_seq_length:  # 补0
            padding_mat = np.zeros([self.max_seq_length - length, self.s_dim])
            s = np.vstack((s, padding_mat))
        return self.sess.run(self.output_action,
                             feed_dict={self.S1: np.reshape(s, [-1, self.max_seq_length, self.s_dim]),
                                        self.S1_length: [length]})

    def learn(self):
        if len(self.memory) < self.replay_start:
            return 0, 0
        all_index = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)  # 得到所有的index
        sample_s1 = []
        sample_s1_length = []
        sample_action = []
        sample_reward = []
        sample_s2 = []
        sample_s2_length = []
        sample_done = []
        for index in all_index:
            element = self.memory[index]
            sample_s1.append(element[0])
            sample_s1_length.append(element[1])
            sample_action.append(element[2])
            sample_reward.append(element[3])
            sample_s2.append(element[4])
            sample_s2_length.append(element[5])
            sample_done.append(element[6])
        sample_s1 = np.array(sample_s1).reshape([BATCH_SIZE, self.max_seq_length, self.s_dim])
        sample_s1_length = np.array(sample_s1_length)
        sample_action = np.array(sample_action).reshape([BATCH_SIZE, int(self.a_dim)])
        sample_reward = np.array(sample_reward)
        sample_s2 = np.array(sample_s2).reshape([BATCH_SIZE, self.max_seq_length, self.s_dim])
        sample_s2_length = np.array(sample_s2_length)
        sample_done = np.array(sample_done)
        self.sess.run(self.atrain, feed_dict={self.S1: sample_s1, self.S1_length: sample_s1_length})
        self.sess.run(self.ctrain, feed_dict={self.S1: sample_s1, self.S1_length: sample_s1_length,
                                              self.S2: sample_s2, self.S2_length: sample_s2_length,
                                              self.a: sample_action,
                                              self.R: sample_reward,
                                              self.done: sample_done})
        actor_loss, critic_loss = self.sess.run([self.a_loss, self.c_loss],
                                                feed_dict={self.S1: sample_s1, self.S1_length: sample_s1_length,
                                                           self.S2: sample_s2, self.S2_length: sample_s2_length,
                                                           self.a: sample_action,
                                                           self.R: sample_reward,
                                                           self.done: sample_done})
        return actor_loss, critic_loss

    def eval_critic(self, s, s_length, a):
        # 根据state和action对critic进行估值
        s = np.array(s).reshape([s_length, self.s_dim])
        a = np.reshape(a, [-1, int(self.a_dim)])
        if s_length < self.max_seq_length:  # 补0
            padding_mat = np.zeros([self.max_seq_length - s_length, self.s_dim])
            s = np.vstack((s, padding_mat))
        return self.sess.run(self.predict_q, feed_dict={self.S1: np.reshape(s, [-1, self.max_seq_length, self.s_dim]),
                                                        self.S1_length: [s_length], self.a: a})

    def store_transition(self, s1, s1_length, a, r, s2, s2_length, done):
        # 需要reshape一下，且需要把s1这些给补0掉。
        s1 = np.array(s1).reshape([s1_length, self.s_dim])
        s2 = np.array(s2).reshape([s2_length, self.s_dim])
        if s1_length < self.max_seq_length:  # 补0
            padding_mat = np.zeros([self.max_seq_length - s1_length, self.s_dim])
            s1 = np.vstack((s1, padding_mat))
        if s2_length < self.max_seq_length:
            padding_mat = np.zeros([self.max_seq_length - s2_length, self.s_dim])
            s2 = np.vstack((s2, padding_mat))
        transition = (s1, s1_length, a, r, s2, s2_length, done)
        self.memory.append(transition)

    def _build_a(self, s, s_length, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=HIDDEN_SIZE)
            #basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=HIDDEN_SIZE)
            _, states = tf.nn.dynamic_rnn(basic_cell, s, dtype=tf.float64, sequence_length=s_length)
            h1 = tf.layers.dense(states, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=STD_DEV))
            h2 = tf.layers.dense(h1, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=STD_DEV))
            h3 = tf.layers.dense(h2, units=self.a_dim, activation=tf.nn.tanh, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=STD_DEV))
            return (h3 + 1) * self.a_bound  # output range should scale to [0, self.a.bound]

    def _build_c(self, s, s_length, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=HIDDEN_SIZE)  # state representation
            #basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=HIDDEN_SIZE)
            _, states = tf.nn.dynamic_rnn(basic_cell, s, dtype=tf.float64, sequence_length=s_length)
            input_s = tf.reshape(states, [-1, HIDDEN_SIZE])
            input_a = tf.reshape(a, [-1, int(self.a_dim)])
            input_all = tf.concat([input_s, input_a], axis=1)
            h1 = tf.layers.dense(input_all, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=STD_DEV))
            h2 = tf.layers.dense(h1, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=STD_DEV))
            h3 = tf.layers.dense(h2, units=1, activation=None, trainable=trainable,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=STD_DEV))
            return h3
