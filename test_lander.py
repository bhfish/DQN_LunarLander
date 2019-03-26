import numpy as np
import gym
import random
import tensorflow as tf
import time
from collections import deque
import sys
import matplotlib.pyplot as plt

model_folder = 'model-20190318-022048'
model_no = 3307
model_to_restore = './models/'+model_folder+'/model-'+str(model_no)
meta_filename = model_to_restore+'.meta'

total_test_episode = 100

def vectorize_action(action):
    v = np.zeros(4)
    v[action] = 1
    return v

env = gym.make("LunarLander-v2")
action_space_n = env.action_space.n
tf.reset_default_graph()
saver = tf.train.import_meta_graph(meta_filename)
graph = tf.get_default_graph()

with tf.Session() as sess:    
    saver.restore(sess, model_to_restore)
    s = graph.get_tensor_by_name("state:0")
    a = graph.get_tensor_by_name("action:0")
    Q = graph.get_tensor_by_name("LeakyRelu:0")
    scores = []
    for episode in range(total_test_episode):
        done = False
        score = 0
        state = env.reset()
        while not done:        
            Q_arr = np.zeros(action_space_n)
            for action_id in range(action_space_n):
                Q_arr[action_id] = sess.run(Q, feed_dict={s: [state], a: [vectorize_action(action_id)]})
            action = np.argmax(Q_arr)
            new_state, reward, done, info = env.step(action)
            env.render()
            score += reward
            state = new_state

        print("episode: {}, total score: {}".format(episode, score))
        scores.append(score)
    avg_score = np.average(scores)
    print("=====Average score: {}".format(avg_score))
    plt.plot(scores, "ro")
    plt.title("The reward per trial for 100 trials using trained agent\n \
                   (average reward = {:.2f})".format(avg_score))
    plt.axhline(y=avg_score, color='r', linestyle='-', label='average={:.2f}'.format(avg_score))
    plt.axhline(y=200, color='b', linestyle='--', label='reward=200')
    plt.legend()
    plt.xlabel('Trial No.')
    plt.ylabel('Reward')
    plt.savefig(model_folder+'_'+str(model_no)+".png")

env.close()