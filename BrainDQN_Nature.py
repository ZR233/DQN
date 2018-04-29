import tensorflow as tf 
import numpy as np 
import random
from collections import deque 
import os


class Config(object):
    # 动作数
    actions = 3
    # 处理后图像高
    n_x = 80
    # 处理后图像底
    n_y = 80
    # 输入深度
    n_z = 3

    # 一批的数量
    batch_size = 32
    # 全连接层节点数
    dense_uints = 512
    # 记忆池容量
    memory_size=50000
    # 学习率
    learning_rate=0.000001
    reward_decay=0.9

    # timesteps to observe before training
    observe = 100. 
    # frames over which to anneal epsilon
    explore = 200000. 
    # decay rate of past observations
    gamma = 0.99
    # 0.001 # final value of epsilon
    final_epsilon = 0
    # 0.01 # starting value of epsilon
    initial_epsilon = 0.3
    # Q Target 更新间隔
    update_time = 100
    frame_per_action = 1
    # 模型保存间隔
    save_step = 10000

class BrainDQN:

    def __init__(self, Config):
        self.actions = Config.actions
        self.n_x = Config.n_x
        self.n_y = Config.n_y
        self.n_z = Config.n_z
        self.batch_size = Config.batch_size
        self.dense_uints = Config.dense_uints
        self.memory_size = Config.memory_size
        self.gamma = Config.gamma
        self.observe = Config.observe
        self.explore = Config.explore
        self.update_time = Config.update_time
        self.initial_epsilon = Config.initial_epsilon
        self.final_epsilon = Config.final_epsilon
        self.save_step = Config.save_step
        self.learning_rate = Config.learning_rate

        self.step_this = 0
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.global_step = tf.Variable(0,trainable=False, name= 'global_step')
        self.step_in = tf.placeholder('int32')
        self.update_step = tf.assign(self.global_step,self.step_in)


        self.epsilon = Config.initial_epsilon
        self.epsilon_v = tf.Variable(Config.initial_epsilon, trainable=False,name='epsilon')
        self.epsilon_in = tf.placeholder('float32')
        self.update_epsilon = tf.assign(self.epsilon_v,self.epsilon_in)


        self.frame_per_action = Config.frame_per_action
        # init Q network
        with tf.name_scope('Q_net'):
            self.stateInput,self.QValue,self.W_conv1,self.b_conv1,\
            self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,\
            self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

        # init Target Q Network
        with tf.name_scope('Q_target'):
            self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,\
            self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,\
            self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
        with tf.name_scope('Q_target_update'):
            self.copyTargetQNetworkOperation = \
                [self.W_conv1T.assign(self.W_conv1),
                self.b_conv1T.assign(self.b_conv1),
                self.W_conv2T.assign(self.W_conv2),
                self.b_conv2T.assign(self.b_conv2),
                self.W_conv3T.assign(self.W_conv3),
                self.b_conv3T.assign(self.b_conv3),
                self.W_fc1T.assign(self.W_fc1),
                self.b_fc1T.assign(self.b_fc1),
                self.W_fc2T.assign(self.W_fc2),
                self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("\n成功加载模型：", checkpoint.model_checkpoint_path,'\n')
        else:
            print("\n未加载模型\n")
        self.timeStep = self.sess.run(self.global_step)
        self.epsilon = self.sess.run(self.epsilon_v)

        self.pic_src = tf.placeholder("float",[None,None,3])
        grayed_image = tf.image.rgb_to_grayscale(self.pic_src)
        self.dealed_pic = tf.image.resize_images(grayed_image,[80,80])
        self.writer = tf.summary.FileWriter('log',tf.get_default_graph())

    def createQNetwork(self):
        # input layer
        with tf.name_scope('input'):
            stateInput = tf.placeholder("float",[None,80,80,4],'s')
        # network weights
        # shape=(80,80,4)
        with tf.name_scope('layer1-conv1'):
            W_conv1 = self.weight_variable('w',[8,8,4,32])
            b_conv1 = self.bias_variable('b', [32])
            h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        # shape=(20,20,32)
        with tf.name_scope('layer2-pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)
        # shape=(10,10,32)
        with tf.name_scope('layer3-conv2'):
            W_conv2 = self.weight_variable('w', [4,4,32,64])
            b_conv2 = self.bias_variable('b', [64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)
        # shape=(5,5,64)
        with tf.name_scope('layer4-conv3'):
            W_conv3 = self.weight_variable('w', [3,3,64,64])
            b_conv3 = self.bias_variable('b', [64])
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
            # shape=(5,5,64)
            h_conv3_flat = tf.reshape(h_conv3,[-1,1600])

        with tf.name_scope('layer5-fc1'):
            # shape=(1,1600)
            W_fc1 = self.weight_variable('w', [1600,self.dense_uints])
            b_fc1 = self.bias_variable('b', [self.dense_uints])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
        with tf.name_scope('layer6-fc2'):
            # shape=(1, dense_uints)
            W_fc2 = self.weight_variable('w',[self.dense_uints,self.actions])
            b_fc2 = self.bias_variable('b',[self.actions])
        with tf.name_scope('Q_eval'):
            # shape=(1, actions)
            # Q Value layer
            QValue = tf.matmul(h_fc1,W_fc2) + b_fc2
        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)
	
    def createTrainingMethod(self):
        with tf.name_scope('train_input'):
            self.actionInput = tf.placeholder("float",[None,self.actions])
            self.yInput = tf.placeholder("float", [None]) 
        with tf.name_scope('loss_function'):
            Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
            self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
            # self.cost = tf.reduce_mean(tf.square(self.yInput - self.actionInput))
            self.trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
        y_batch = []
        a_batch = []
        QValue_batch = self.sess.run(self.QValueT, feed_dict={self.stateInputT:nextState_batch})
        for i in range(0, self.batch_size):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(QValue_batch[i]))
            action_array = np.zeros(self.actions,int)
            action_array[action_batch[i]]=1
            a_batch.append(action_array)



        self.sess.run(self.trainStep,feed_dict={
            self.yInput : y_batch,
            self.actionInput : a_batch,
            self.stateInput : state_batch
            })

        # save network every 100000 iteration
        if self.timeStep % self.save_step == 0:
            self.saver.save(self.sess, os.path.join('saved_networks', 'network' + '-dqn'), global_step = self.timeStep)
            print('保存模型')
        if self.timeStep % self.update_time == 0:
            self.copyTargetQNetwork()

    def setPerception(self,nextObservation,action,reward,terminal):
        #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # grayed_image = tf.image.rgb_to_grayscale(nextObservation)
        # nextObservation = tf.image.resize_images(grayed_image,[80,80])

        nextObservation = self.sess.run(self.dealed_pic,feed_dict={
            self.pic_src: nextObservation
        })
        newState = np.append(
            self.currentState[:,:,1:],
            nextObservation,
            axis = 2
            )
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > self.memory_size:
            self.replayMemory.popleft()
        if self.step_this > self.observe:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.step_this <= self.observe:
            state = "observe"
        elif self.timeStep > self.observe and self.timeStep <= self.observe + self.explore:
            state = "explore"
        else:
            state = "train"

        print ("TIMESTEP", self.timeStep, "/ STATE", state, \
            "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1
        self.sess.run(self.update_step,feed_dict={self.step_in:self.timeStep})
        self.step_this += 1

    def getAction(self):
        QValue = self.sess.run(self.QValue,feed_dict= {self.stateInput:[self.currentState]})[0]
        if self.timeStep % self.frame_per_action == 0:
            if random.random() <= self.epsilon:
                action = random.randrange(self.actions)
            else:
                action = np.argmax(QValue)
        else:
            action = 0 # do nothing

		# change episilon
        if self.epsilon > self.final_epsilon and self.timeStep > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon)/self.explore
            self.sess.run(self.update_epsilon, feed_dict= {self.epsilon_in:self.epsilon})
        return action

    def setInitState(self,observation):
        # grayed_image = tf.image.rgb_to_grayscale(observation)
        # resized = tf.image.resize_images(grayed_image,[80,80])
        # observation = self.sess.run(tf.squeeze(resized,2))

        observation = self.sess.run(tf.squeeze(self.dealed_pic,2),feed_dict={
            self.pic_src: observation
        })
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)


    def weight_variable(self, name, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, name, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")