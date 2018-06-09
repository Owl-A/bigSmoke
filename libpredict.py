import tensorflow as tf
from numpy import genfromtxt
import numpy as np

### Default Values

__n=20              # Number of days prior data
__n_nodes_hl1=40    # nodes in layer one of TDNN
__epochs=100        # iterations for convergence of TDNN 
__batch_size=100    # the batch size for batch gradient descent
__n_cls=1           # the number of perceptrons in output layer
__margin=10.0       # [ DEBUG ] tolerance to misprediction in currency units 
__debug=1           # [ DEBUG ] enable debug mode

def eval_neural_network(data, w1, w2, b1, b2, n=__n, n_nodes_hl1=__n_nodes_hl1, n_cls=__n_cls, batch_size = __batch_size):
    l1 = tf.add(tf.matmul(data,w1), b1)
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1,w2) + b2
    return output
def neural_network_model(data, n=__n, n_nodes_hl1=__n_nodes_hl1, n_cls=__n_cls, batch_size = __batch_size):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n, n_nodes_hl1]),name='w1'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]),name='b1')}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_cls]), name='w2'),
                    'biases':tf.Variable(tf.random_normal([n_cls]), name='b2'),}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']
    return output
def get_file_data(savefile, n=__n, n_nodes_hl1=__n_nodes_hl1, n_cls=__n_cls, epochs=__epochs, margin=__margin, batch_size = __batch_size):

    tmp = genfromtxt(savefile)
    dat = np.zeros(tmp.shape[0],dtype=np.float32)
    dat[:] = tmp

    train_x = np.zeros([n,dat.shape[0]-n], dtype=np.float32)
    train_y = np.zeros([dat.shape[0]-n,1], dtype=np.float32)
    for i in range(n):
      train_x[i,:] = dat[i:i+dat.shape[0]-n]
    train_y[:,0] = dat[n:]
    train_x = np.transpose(train_x)
    if __debug == 1:
        test_x = train_x[500:]
        test_y = train_y[500:]
        train_x = train_x[:500]
        train_y = train_y[:500]
    
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y
def train_and_save_neural_network(trainDataFile, savefile, n=__n, n_nodes_hl1=__n_nodes_hl1, n_cls=__n_cls, epochs=__epochs, margin=__margin, batch_size = __batch_size):
    x = tf.placeholder('float', [None, n])
    y = tf.placeholder('float')

    if __debug == 1 :
        train_x, train_y, test_x, test_y = get_file_data(trainDataFile)
    else:
        train_x, train_y = get_file_data(trainDataFile)
    
    prediction = neural_network_model(x, n=n, n_nodes_hl1=n_nodes_hl1, n_cls=n_cls, batch_size = batch_size)
    cost = (tf.norm(prediction - y))**2
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(1, int(train_x.shape[0]/batch_size)):
                epoch_x =  train_x[batch_size*(i-1):batch_size*i]
                epoch_y =  train_y[batch_size*(i-1):batch_size*i]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                if __debug==1:
                    print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)

        if __debug==1 :
            correct = tf.greater(float(margin),  tf.abs(prediction-y))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

        saver = tf.train.Saver()
        try:
            saver.save(sess, savefile, global_step=1000)
        except:
            print("Unable to save the file!")
        sess.close()
def load_and_run_neural_network(savefile, Data, n=__n, n_nodes_hl1=__n_nodes_hl1, n_cls=__n_cls, epochs=__epochs, margin=__margin, batch_size = __batch_size):
    x = tf.placeholder('float', [None, n])
    y = tf.placeholder('float')
    Data = np.reshape(np.array(Data), [1,n])
    with tf.Session() as sess:
        try :
            loader = tf.train.import_meta_graph(savefile+'.meta')
            loader.restore(sess,savefile)
        except :
            print("could not read the savefile")
            return None
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        b1 = graph.get_tensor_by_name("b1:0")
        b2 = graph.get_tensor_by_name("b2:0")
        Data = Data.astype(np.float32)
        res = eval_neural_network(Data, w1, w2, b1, b2, n=n, n_nodes_hl1=n_nodes_hl1, n_cls=n_cls, batch_size=batch_size)
        #loader.restore(sess, tf.train.latest_checkpoint('./'))
        #accuracy = graph.get_tensor_by_name("acc:0")
        #val = sess.run(accuracy, feed_dict={x: Data, y: Data})
        #val = val.eval({x: Data}) 
        #for i in tf.all_variables():
        #    print i.name
        res = sess.run(res)[0][0]
        sess.close()
        print(res)