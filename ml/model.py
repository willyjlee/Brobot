from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import random

if not os.path.exists('data/words.pickle') or not os.path.exists('data/input.npy') or not os.path.exists('data/output.npy'):
    print('Run prep.py first')
    exit()
with open('data/words.pickle', 'rb') as rp:
    words = pickle.load(rp)
with open('data/rev_words.pickle', 'rb') as rrp:
    rev_words = pickle.load(rrp)

input = np.load('data/input.npy')
output = np.load('data/output.npy')

print('input:', input.shape)
vocab_size = len(words)
batch_size = 20
l_units = 50
# keep at vocab size?
num_encoder_symbols = vocab_size
num_decoder_symbols = vocab_size
embedding_size = l_units
input_max_size = 8
num_times = 300000

def get_train_batch(train_in, train_out, batch_size, dict):
    indices = random.sample(range(train_in.shape[0]), batch_size)
    train_in = train_in[indices]
    train_out = train_out[indices]
    encoder_in_batch = []
    target_in_batch = []
    for i in range(input_max_size):
        encoder_in_batch.append(np.array(train_in[:, i]))
        target_in_batch.append(np.array(train_out[:, i]))
    decoder_in_batch = target_in_batch[:-1]
    decoder_in_batch.insert(0, np.full(batch_size, dict['{GO}']))
    return encoder_in_batch, target_in_batch, decoder_in_batch

# given list of [batch_size]
def sentence(ids, rev_dict, index):
    str = []
    for i in range(len(ids)):
        str.append(rev_dict[ids[i][index]])
    return ' '.join(str)

tf.reset_default_graph()

# change to tf.nn.bidirectional_dyanmic_rnn ?
# add multi rnn cell
cell = tf.contrib.rnn.BasicLSTMCell(l_units, state_is_tuple=True)

encoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name='encoder_inputs' + str(i)) for i in range(input_max_size)]
decoder_inputs = [tf.placeholder(tf.int32, shape=(None,), name='decoder_inputs' + str(i)) for i in range(input_max_size)]
targets = [tf.placeholder(tf.int32, shape=(None,), name='targets' + str(i)) for i in range(input_max_size)]

print(len(decoder_inputs))

feed_previous = tf.placeholder(tf.bool, name='feed_previous')

# switch to tf.contrib.seq2seq.sequence_loss ?
# inputs are embedded internally
out_logits, state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    feed_previous=feed_previous
)

prediction = tf.argmax(out_logits, 2, name='prediction')

loss_weights = [tf.ones_like([batch_size], dtype=tf.float32) for _ in range(input_max_size)]

loss = tf.contrib.legacy_seq2seq.sequence_loss(
    out_logits,
    targets,
    loss_weights
)

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)


sess = tf.Session()

saver = tf.train.Saver()
# check for saved model
if os.listdir('models/'):
    checkpoint = tf.train.latest_checkpoint('models/')
    saver.restore(sess, checkpoint)
    print(checkpoint, 'model loaded...')
else:
    sess.run(tf.global_variables_initializer())

print('Session made...')
for i in range(num_times):
    feed_dict = dict()
    encoder_in, target_in, decoder_in = get_train_batch(input, output, batch_size, words)

    # building the feed_dict
    for j in range(input_max_size):
        feed_dict[encoder_inputs[j]] = encoder_in[j]
        feed_dict[decoder_inputs[j]] = decoder_in[j]
        feed_dict[targets[j]] = target_in[j]

    feed_dict[feed_previous] = False

    _, l, pred = sess.run(
        [optimizer, loss, prediction], feed_dict=feed_dict
    )
    if i % 5 == 0 and i != 0:
        print('Loss: %f' % l)
    if i % 50 == 0 and i != 0:
        print('Step:', i)
        print('Validating...')

        # batch_size only 1
        encoder_in, _, _ = get_train_batch(input, output, 1, words)

        # decoder_in[0] is index np.array([words['{GO}']]) ?
        decoder_in = [np.array([words['{GO}']]) for _ in range(input_max_size)]
        target_in = [np.array([0]) for _ in range(input_max_size)]

        for j in range(input_max_size):
            feed_dict[encoder_inputs[j]] = encoder_in[j]
            feed_dict[decoder_inputs[j]] = decoder_in[j]
            feed_dict[targets[j]] = target_in[j]

        # rand_index = random.randint(0, batch_size - 1)
        feed_dict[feed_previous] = True
        pred, target, dec_in, enc_in= sess.run(
            [prediction, targets, decoder_inputs, encoder_inputs], feed_dict=feed_dict
        )

        print('Input:', sentence(enc_in, rev_words, 0))
        print('Dec_in:', sentence(dec_in, rev_words, 0))
        print('Target:', sentence(target, rev_words, 0))
        print('Output:', sentence(pred, rev_words, 0))
    if i % 7500 == 0 and i != 0:
        print('Saving model...')
        path = saver.save(sess, 'models/model.ckpt', global_step=i)
        print('Model saved to: %s' % path)


