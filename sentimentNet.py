import pickle

import numpy as np
import tensorflow as tf
from clean import clean_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Lib to lemmatize
lemmatizer = WordNetLemmatizer()

# Load lexicon to gather the lexicon size
with open('data/lexicon.pickle', 'rb') as f:
    lexicon = pickle.load(f)

# Count the lines in train set
file = open("data/train_set_shuffled.csv")
numlines = len(file.readlines())

# Number of input nodes
n_nodes_input = len(lexicon)

# Number of hidden layer nodes
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000

# Number of output classes
n_classes = 2

# Batch size to process
batch_size = 256
total_batches = int(numlines / batch_size)

# Epoch Count
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

# Layer definitions
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Declaration of the neural network
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output


saver = tf.train.Saver()
tf_log = 'data/tf.log'


# Test the neural network
def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess, "data/model.ckpt")
            except Exception as e:
                print(str(e))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('data/test_set_vec.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Test completed with', counter, 'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


#test_neural_network()


# Train the neural network
def train_neural_network(x):

    # Clean files on disk
    clean_model()

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
            print('STARTING:', epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "data/model.ckpt")
            epoch_loss = 1
            with open('data/train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            features[lexicon.index(word.lower())] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={
                            x: np.array(batch_x),
                            y: np.array(batch_y)
                        })

                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch Loss:', c, )

            saver.save(sess, "data/model.ckpt")
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')
            epoch += 1

    test_neural_network()





def use_neural_network(input_data):
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "data/model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                features[lexicon.index(word.lower())] += 1

        features = np.array(list(features))

        # 4 means positive [0,1], argmax: 1
        # 0 means negative [1,0], argmax: 0

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
        if result[0] == 1:
            print('Positive:', input_data)
        elif result[0] == 0:
            print('Negative:', input_data)


# train_neural_network(x)
use_neural_network("This is not good at all.")
use_neural_network("I really hate the taste of coca cola.")
use_neural_network("I hate sitting in the rain.")
use_neural_network("i love like singing in the sun")
use_neural_network("I would like to test your delicious toasts")
