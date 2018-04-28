import numpy as np
import tensorflow as tf
import os

np.random.seed(1)
tf.set_random_seed(1)

class Config(object):
    # 卷积层过滤器尺寸
    filter_size = 5
    # 卷积层过滤器深度
    filter_deep = 10
    # 动作数
    n_actions = 3

    # 处理后图像高
    n_x = 50
    # 处理后图像底
    n_y = 50

    # 输入图像高
    in_n_x = 32
    # 输入图像底
    in_n_y = 32
    # 输入图像深度
    n_z = 3

    # 一批的数量
    batch_size = 5
    # 全连接层节点数
    dense_uints = 512
    # 记忆池容量
    memory_size=50
    # 学习率
    learning_rate=0.01
    reward_decay=0.9
    gamma = reward_decay
    e_greedy=0.9
    replace_target_iter=100
    save_step = 300
    # 模型文件路径
    model_path = "model"
    # 模型文件名
    model_name = "model.ckpt"
    # 日志文件路径
    log_path = "log"

    e_greedy_increment = 0.0008
    # 训练回合数
    episodes = 1000


def inference(Config, s, train = False):
    """ s.shape=(batch_size, n_x, n_y, n_z)\n
        a.shape=(batch_size,1)\n
        r.shape=(batch_size,1)\n
        s_.shape=(batch_size, n_x, n_y, n_z)
    """
     # ------------------ build evaluate_net ------------------
    with tf.variable_scope('eval_net'):
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable(
                "weight", [Config.n_x, Config.n_y, Config.n_z, Config.filter_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable('biase', [Config.filter_deep],
                                            initializer=tf.constant_initializer(0.0))

            conv1 = tf.nn.conv2d(s, conv1_weights, strides=[1, 1, 1, 1],
                                    padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        with tf.variable_scope('layer2-conv2'):
            conv2_weights = tf.get_variable(
                "weight", [Config.n_x, Config.n_y, Config.filter_deep, Config.filter_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable('biase', [Config.filter_deep],
                                            initializer=tf.constant_initializer(0.0))

            conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1],
                                    padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        with tf.variable_scope('layer3-conv3'):
            conv3_weights = tf.get_variable(
                "weight", [Config.n_x, Config.n_y, Config.filter_deep, Config.filter_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases = tf.get_variable('biase', [Config.filter_deep],
                                            initializer=tf.constant_initializer(0.0))

            conv3 = tf.nn.conv2d(relu2, conv3_weights, strides=[1, 1, 1, 1],
                                    padding='SAME')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
            pool_shape = relu3.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(relu3, [-1, nodes])
        #全连接层1
        with tf.variable_scope('layer4-dense1'):
            
            fc1_weights = tf.get_variable(
                "weight", [nodes, Config.dense_uints],
                initializer= tf.truncated_normal_initializer(stddev= 0.1))
            
            fc1_biases = tf.get_variable('biase', [Config.dense_uints],
                initializer= tf.constant_initializer(0.1))

            fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
            if train:
                fc1 = tf.nn.dropout(fc1, 0.5)
        #全连接层2
        with tf.variable_scope('layer5-dense2'):
            fc2_weights = tf.get_variable(
                "weight", [Config.dense_uints,Config.n_actions],
                initializer= tf.truncated_normal_initializer(stddev= 0.1))
            
            fc2_biases = tf.get_variable('biase', [Config.n_actions],
                initializer= tf.constant_initializer(0.1))

            q_eval = tf.matmul(fc1,fc2_weights) + fc2_biases

 
    return q_eval


def eval(Config, s, sess = None):
    with tf.name_scope('eval'):
        # input State
        s = tf.placeholder(tf.float32, [1, Config.n_x, Config.n_y, Config.n_z], name='s')
        # input Next State
        s_ = tf.placeholder(tf.float32, [1, Config.n_x, Config.n_y, Config.n_z], name='s_')  
        # input Reward
        r = tf.placeholder(tf.float32, [1, ], name='r')  
        # input Action
        a = tf.placeholder(tf.int32, [1, ], name='a')
    inference(Config, s, train = False)


def train(Config, env):
    epsilon_max = Config.e_greedy
    epsilon = 0 if Config.e_greedy_increment is not None else epsilon_max
    epsilon_increment = Config.e_greedy_increment
    global_step = tf.Variable(0, trainable=False)
    # ------------------tarin inputs-----------------------

    with tf.name_scope('tarin'):
        # input State
        s = tf.placeholder(tf.float32, [None, Config.n_x, Config.n_y, Config.n_z], name='s')
        # input Next State
        s_ = tf.placeholder(tf.float32, [None, Config.n_x, Config.n_y, Config.n_z], name='s_')  
        # input Reward
        r = tf.placeholder(tf.float32, [None, ], name='r')  
        # input Action
        a = tf.placeholder(tf.int32, [None, ], name='a')

    q_eval= inference(Config, s, train=True)

    # ------------------ build target_net ------------------
    with tf.variable_scope('target_net'):
        with tf.variable_scope('layer1-conv1'):
            conv1_weights_t = tf.get_variable(
                "weight", [Config.n_x, Config.n_y, Config.n_z, Config.filter_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases_t = tf.get_variable('biase', [Config.filter_deep],
                                            initializer=tf.constant_initializer(0.0))

            conv1_t = tf.nn.conv2d(s_, conv1_weights_t, strides=[1, 1, 1, 1],
                                    padding='SAME')
            relu1_t = tf.nn.relu(tf.nn.bias_add(conv1_t, conv1_biases_t))
        with tf.variable_scope('layer2-conv2'):
            conv2_weights_t = tf.get_variable(
                "weight", [Config.n_x, Config.n_y, Config.filter_deep, Config.filter_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases_t = tf.get_variable('biase', [Config.filter_deep],
                                            initializer=tf.constant_initializer(0.0))

            conv2_t = tf.nn.conv2d(relu1_t, conv2_weights_t, strides=[1, 1, 1, 1],
                                    padding='SAME')
            relu2_t = tf.nn.relu(tf.nn.bias_add(conv2_t, conv2_biases_t))
        with tf.variable_scope('layer3-conv3'):
            conv3_weights_t = tf.get_variable(
                "weight", [Config.n_x, Config.n_y, Config.filter_deep, Config.filter_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_biases_t = tf.get_variable('biase', [Config.filter_deep],
                                            initializer=tf.constant_initializer(0.0))

            conv3_t = tf.nn.conv2d(relu2_t, conv3_weights_t, strides=[1, 1, 1, 1],
                                    padding='SAME')
            relu3_t = tf.nn.relu(tf.nn.bias_add(conv3_t, conv3_biases_t))
            pool_shape_t = relu3_t.get_shape().as_list()
            nodes_t = pool_shape_t[1] * pool_shape_t[2] * pool_shape_t[3]
            reshaped_t = tf.reshape(relu3_t, [-1, nodes_t])
        #全连接层1
        with tf.variable_scope('layer4-dense1'):
            
            fc1_weights_t = tf.get_variable(
                "weight", [nodes_t, Config.dense_uints],
                initializer= tf.truncated_normal_initializer(stddev= 0.1))
            
            fc1_biases_t = tf.get_variable('biase', [Config.dense_uints],
                initializer= tf.constant_initializer(0.1))

            fc1_t = tf.nn.relu(tf.matmul(reshaped_t,fc1_weights_t) + fc1_biases_t)
            if train:
                fc1_t = tf.nn.dropout(fc1_t, 0.5)
        #全连接层2
        with tf.variable_scope('layer5-dense2'):
            fc2_weights_t = tf.get_variable(
                "weight", [Config.dense_uints,Config.n_actions],
                initializer= tf.truncated_normal_initializer(stddev= 0.1))
            
            fc2_biases_t = tf.get_variable('biase', [Config.n_actions],
                initializer= tf.constant_initializer(0.1))

            q_next = tf.matmul(fc1_t, fc2_weights_t) + fc2_biases_t

    with tf.variable_scope('q_target'):
        q_target_ = r + Config.gamma * tf.reduce_max(q_next, axis=1, name='Qmax_s_')  # shape=(batch_size, )
        q_target = tf.stop_gradient(q_target_)
    with tf.variable_scope('q_eval'):
        a_indices = tf.stack([tf.range(tf.shape(a)[0], dtype=tf.int32), a], axis=1)
        q_eval_wrt_a = tf.gather_nd(params=q_eval, indices=a_indices)  # shape=(batch_size, )
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval_wrt_a, name='TD_error'))
    with tf.variable_scope('train'):
        train_op = tf.train.RMSPropOptimizer(Config.learning_rate).minimize(loss,global_step=global_step)

    memory_s = []
    memory_s_ = []
    memory_a = []
    memory_r = []
    temp_s = np.zeros((Config.n_x, Config.n_y, Config.n_z))
    for i in range(Config.memory_size):
        memory_s.append(temp_s)
        memory_s_.append(temp_s)
        memory_a.append(0.)
        memory_r.append(0.)
    mem_count = 0
    total_steps =0
    learn_step_counter = 0
    cost_his =[]
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    with tf.variable_scope('soft_replacement'):
        target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(Config.log_path, tf.get_default_graph())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(
            Config.model_path
        )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('加载已保存的变量')

        for i_episode in range(Config.episodes):
            observation = env.reset()
            observation = tf.image.resize_images(observation, [Config.n_x, Config.n_y],method=0).eval()
            ep_r = 0
            while True:
                # time.sleep(0.1)
                env.render()    # 刷新环境
                # 选行为
                if np.random.uniform() < epsilon:
                    # forward feed the observation and get q value for every actions
                    actions_value = sess.run(q_eval, feed_dict={s: [observation]})
                    action = np.argmax(actions_value)
                else:
                    action = np.random.randint(0, Config.n_actions)

                observation_, reward, done, _ = env.step(action) # 获取下一个 state
                # 保存这一组记忆
                if mem_count==Config.memory_size:
                    mem_count = 0

                observation_ = tf.image.resize_images(observation_, [Config.n_x, Config.n_y],method=0).eval()
                memory_s[mem_count] = observation
                memory_a[mem_count] = action
                memory_r[mem_count] = reward
                memory_s_[mem_count] = observation_
                
                # 学习
                if total_steps > Config.memory_size*2:
                    # check to replace target parameters
                    if learn_step_counter % Config.replace_target_iter == 0:
                        sess.run(target_replace_op)
                        print('\ntarget_params_replaced\n')
                    if learn_step_counter % Config.save_step == 0:
                        print('保存模型和日志')
                        saver.save(
                            sess, os.path.join(Config.model_path, Config.model_name),
                            global_step=global_step
                        )
                    # 从 memory 中随机抽取 batch_size 这么多记忆
                    eval_s = []
                    eval_s_ =[]
                    eval_a = []
                    eval_r = []
                    sample_index = np.random.choice(Config.memory_size, size=Config.batch_size)
                    for sp in sample_index:
                        eval_s.append(memory_s[sp].tolist())
                        eval_s_.append(memory_s_[sp].tolist())
                        eval_a.append(memory_a[sp])
                        eval_r.append(memory_r[sp])
                    eval_s = np.array(eval_s)
                    eval_a = np.array(eval_a)
                    eval_r = np.array(eval_r)
                    eval_s_ = np.array(eval_s_)

                    _, cost = sess.run(
                        [train_op, loss],
                        feed_dict={
                            s: eval_s,
                            a: eval_a,
                            r: eval_r,
                            s_: eval_s_,
                        })

                    cost_his.append(cost)

                    # increasing epsilon
                    epsilon = epsilon + epsilon_increment if epsilon < epsilon_max else epsilon_max
                    learn_step_counter += 1

                ep_r += reward
                if done:
                    print('episode: ', i_episode,
                        'ep_r: ', round(ep_r, 2),
                        ' epsilon: ', round(epsilon, 2))
                    break

                observation = observation_
                total_steps += 1
        writer.close()
