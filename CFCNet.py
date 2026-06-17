import os
import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf
from tensorflow.keras import utils as np_utils
tf.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES']='1'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40) # 改变这个百分比即可
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()
tf.set_random_seed(1230)

def model(x, w1_1, w2_1,w2_2, w2, w3, w_o, p_keep_conv, p_keep_hidden):
    # l1a_1 = tf.nn.relu(tf.nn.conv3d(x, w1_1, strides=[1, 1, 1, 1, 1], padding='VALID'))
    # l1a_1 = tf.nn.bias_add(l1a_1, b1_1)
    # l1a_1 = tf.nn.dropout(l1a_1, p_keep_conv)

    l2a_1 = tf.nn.relu(tf.nn.conv3d(x, w2_1, strides=[1, 1, 1, 1, 1], padding='VALID'))
    l2a_1 = tf.nn.bias_add(l2a_1, b1_2)
    l2a_1= tf.nn.dropout(l2a_1, p_keep_conv)

    l2a_2 = tf.nn.relu(tf.nn.conv3d(l2a_1, w2_2, strides=[1, 1, 1, 1, 1], padding='VALID'))
    l2a_2 = tf.nn.bias_add(l2a_2, b1_3)
    l2a_2 = tf.nn.dropout(l2a_2, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv3d(l2a_2, w2, strides=[1, 1, 1, 1, 1], padding='VALID'))
    l2a = tf.nn.bias_add(l2a, b2_1)
    l2a = tf.nn.dropout(l2a, p_keep_conv)

    l2 = tf.reshape(l2a, [-1, w3.get_shape().as_list()[0]])
    l3 = tf.nn.relu(tf.matmul(l2, w3))
    l3 = tf.nn.dropout(l3, p_keep_hidden)

    pyx = tf.matmul(l3, w_o)
    return pyx

sub=1
bandnum = 7
mkpath = os.path.join(os.getcwd() + "/BCI42adata/p" )
print('------','/A0%d.mat' % (sub))
mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (sub))

trX = mat_data['train_data'][:,:,:,:bandnum]
trX = trX.reshape(-1, 22, 22, bandnum, 1)
print(trX.shape)
trY = mat_data['train_labels']
trY = np_utils.to_categorical(trY-1)  

teX = mat_data['test_data'][:,:,:,:bandnum]
teX = teX.reshape(-1, 22, 22, bandnum, 1)
teY = mat_data['test_labels']
teY = np_utils.to_categorical(teY-1)

x = tf.placeholder("float", [None, 22, 22, bandnum, 1])
y_ = tf.placeholder("float", [None, 4])

kernel1=3
kernel2=1
kernel3 = 1

feature_map1=40
feature_map2 =100
feature_map3 =50
feature_map4=80

w1_1 = tf.get_variable("w_1", shape=[kernel1, kernel1, kernel2, 1, feature_map1],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
w2_1 = tf.get_variable("w_2", shape=[kernel3, kernel3, kernel3, 1, feature_map2],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))

w2_2 = tf.get_variable("w2_2", shape=[kernel1, kernel1, kernel2, feature_map2, feature_map3],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
w2 = tf.get_variable("w2", shape=[23-kernel1, 23-kernel1, bandnum+1-kernel2, feature_map3,feature_map4],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
w3 = tf.get_variable("w3", shape=[feature_map4, feature_map4], initializer=tf.truncated_normal_initializer(stddev=0.01))
w_o = tf.get_variable("w_o", shape=[feature_map4, 4], initializer=tf.truncated_normal_initializer(stddev=0.01))

b1_1 = tf.get_variable("b1_1", shape=[feature_map1], initializer=tf.constant_initializer(0.0))
b1_2 = tf.get_variable("b1_2", shape=[feature_map2], initializer=tf.constant_initializer(0.0))
b1_3 = tf.get_variable("b1_3", shape=[feature_map3], initializer=tf.constant_initializer(0.0))
b2_1 = tf.get_variable("b2_1", shape=[feature_map4], initializer=tf.constant_initializer(0.0))

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
y = model(x, w1_1, w2_1,w2_2, w2, w3, w_o, p_keep_conv, p_keep_hidden)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_op = tf.train.AdamOptimizer(0.0005).minimize(cost)
predict_op = tf.argmax(y, 1)

itr=500
acc = [0] * itr
tracc = [0] * itr

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(itr):
        keep_drop=0.8
        sess.run(train_op, feed_dict={x: trX, y_: trY, p_keep_conv: keep_drop, p_keep_hidden: keep_drop})
        #print(100 * np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={x: teX, y_: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})))
        tracc[i] = 100 * sess.run(accuracy, feed_dict={x: trX, y_: trY, p_keep_conv: keep_drop, p_keep_hidden: keep_drop})
        acc[i] = 100 * np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={x: teX, y_: teY, p_keep_conv: keep_drop, p_keep_hidden: keep_drop}))
        print(f"epoch: {i}/{itr},  train acc:{tracc[i]}, test acc:{acc[i]}")
    print(max(acc))