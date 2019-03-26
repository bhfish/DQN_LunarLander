import numpy as np
import gym
import random
import tensorflow as tf
import time
from collections import deque
import matplotlib.pyplot as plt

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True,  # Use Rectified Linear Unit (ReLU)?
                 use_leaky_relu=False):  # Use Leaky ReLU?
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases
    
    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    # Use leaky_ReLU?
    if use_leaky_relu:
        layer = tf.nn.leaky_relu(layer)

    return layer

def vectorize_action(action):
    v = np.zeros(4)
    v[action] = 1
    return v

env = gym.make("LunarLander-v2")

save_model = False            # flag indicating whether the model should be saved and/or tested
reward_threshold = 200        # a succussful episode collects at least 200 rewards

total_episodes = 5000         # Total episodes
max_steps = 1000              # Max steps per episode
#total_test_episodes = 100    # Total test episodes

learning_rate = 0.001         # Learning rate
gamma = 0.98                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.001           # Minimum exploration probability 
decay_rate = 0.008            # Exponential decay rate for exploration prob

# Log parameters
timestamp_str = time.strftime('%Y%m%d-%H%M%S',time.localtime())
summary_folder = './summaries_replay/'
model_dir = './models/model-'+ timestamp_str+'/model'
reward_que_size = 100 
replay_que_size = 10000

# Graph parameters
state_space_dim = len(env.reset())
action_space_n = env.action_space.n
num_features_ly1 = state_space_dim + action_space_n
num_features_ly2 = 64
num_outputs_ly2 = 64

# Construct graph
s = tf.placeholder(tf.float32, shape=[1, state_space_dim], name='state')
a = tf.placeholder(tf.float32, shape=[1, action_space_n], name='action')
input_sa = tf.concat([s, a] , axis = 1)
q_true_ph = tf.placeholder(tf.float32, shape=[1, 1], name='q_true')

layer_fc1 = new_fc_layer(input=input_sa,
                         num_inputs=num_features_ly1,
                         num_outputs=num_features_ly2)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=num_features_ly2,
                         num_outputs=num_outputs_ly2)
Q = new_fc_layer(input=layer_fc2,
                 num_inputs=num_outputs_ly2,
                 num_outputs=1,
                 use_relu = False,
                 use_leaky_relu = True)

loss = tf.square(q_true_ph - Q)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss,var_list=tf.trainable_variables())

# Summary placeholders
r_ph = tf.placeholder(tf.float32, shape=[1, 1], name='reward')
r_ep_ph = tf.placeholder(tf.float32, shape=[1, 1], name='ep_reward')
r_avg_ph = tf.placeholder(tf.float32, shape=[1, 1], name='avg_reward')
step_ph = tf.placeholder(tf.float32, shape=[1, 1], name='total_steps')

# Construct summaries
r_final_summary = tf.summary.scalar(name='r_summary', tensor=tf.reshape(r_ph, []))
r_avg_summary = tf.summary.scalar(name='r_avg_summary', tensor=tf.reshape(r_avg_ph, []))
r_ep_summary = tf.summary.scalar(name='r_ep_summary', tensor=tf.reshape(r_ep_ph, []))
step_summary = tf.summary.scalar(name='step_summary', tensor=tf.reshape(step_ph, []))
merged_summary = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(summary_folder, sess.graph)
    reward_que = deque(maxlen=reward_que_size)
    replay_buffer = list()
    for episode in range(total_episodes):
        # Reset the environment
        print("episode: "+str(episode))
        step = 0
        done = False  
        save_model = False
        state = env.reset()
        ep_buffer = list()
        reward_episode = 0
        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0,1)
            # exploitation
            if exp_exp_tradeoff > epsilon:
                Q_arr = np.zeros(action_space_n)
                for action_id in range(action_space_n):
                    Q_arr[action_id] = sess.run(Q, feed_dict={s: [state], a: [vectorize_action(action_id)]})
                action = np.argmax(Q_arr)

            # exploration
            else:
                action = env.action_space.sample()

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)
            # add transition to replay buffer
            ep_buffer.append((state, new_state, reward, action, done))
            reward_episode += reward
            
            # update state
            state = new_state

            # visualize some training process
            if episode%100>90:
                env.render()

            # output some middle results to help monitor training process
            if (step)%50==0:
                print("step: {}, r: {}".format(step, reward))

            # If done : finish episode
            if done == True:
                break

        # monitor training process
        print("step: {}, r: {}".format(step, reward))
        print("==total reward: {}".format(reward_episode))

        # add episode replay memo to the main replay memory
        replay_buffer.extend(ep_buffer)
        # prioritize winning experience: add winning experience to replay buffer twice
        if reward_episode >= 200:
            replay_buffer.extend(ep_buffer)

        # Reduce epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # save summary
        avg_reward = np.mean(reward_que)
        reward_que.append(reward_episode)
        summary = sess.run(merged_summary, feed_dict={r_ph: [[reward]], \
                                                    r_ep_ph: [[reward_episode]], \
                                                    r_avg_ph: [[avg_reward]],\
                                                    step_ph: [[step]]})
        writer.add_summary(summary, episode)
        
        # if avg_reward surpass threshold, save model
        if avg_reward>reward_threshold:
            save_model = True
            reward_threshold = avg_reward
       
        if save_model:
            # save model
            saved_path = saver.save(sess, model_dir, global_step=episode)

        # replay
        if len(replay_buffer) >= 5000:
            random.shuffle(replay_buffer)
            for i in range(2000):
                rpl_state, rpl_new_state, rpl_reward, rpl_action, rpl_done = replay_buffer.pop()
                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                q_next = 0
                if not rpl_done:                   
                    rpl_Q_arr_new = np.zeros(action_space_n)
                    for rpl_action_id in range(action_space_n):
                        rpl_Q_arr_new[rpl_action_id] = sess.run(Q, feed_dict={s: [rpl_new_state], a: [vectorize_action(rpl_action_id)]})
                    q_next = np.amax(rpl_Q_arr_new)
                q_true = rpl_reward + gamma * q_next
                sess.run(train_op, feed_dict={s: [rpl_state], a: [vectorize_action(rpl_action)], q_true_ph: [[q_true]]})

env.close()   

