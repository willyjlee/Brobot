import pickle
import tensorflow as tf
import numpy as np

input_max_size = 8
special = ['{EOS}', '{PAD}']

# run on import
print('Runninng process...')
with open('ml/data/words.pickle', 'rb') as wp:
    words = pickle.load(wp)
with open('ml/data/rev_words.pickle', 'rb') as rp:
	rev_words = pickle.load(rp)
print('Loaded words...')

# make session
sess = tf.Session()

print('Created session...')
# load meta graph and restore variables
checkpoint = tf.train.latest_checkpoint('ml/models/')
print("Checkpoint: " + checkpoint)
saver = tf.train.import_meta_graph(checkpoint + '.meta')
saver.restore(sess, checkpoint)

print('Restored...')

# get tensors
graph = tf.get_default_graph()
prediction = graph.get_tensor_by_name("prediction:0")
feed_previous = graph.get_tensor_by_name("feed_previous:0")

print('Finished')

# input = string
def get_prediction(sentence):
	ids = to_ids(sentence)

	# feed_dict
	feed_dict = dict()
	for i in range(input_max_size):
		feed_dict[graph.get_tensor_by_name('encoder_inputs' + str(i) + ':0')] = ids[i]
		# try with {GO}?
		# batch_size == 1
		feed_dict[graph.get_tensor_by_name('decoder_inputs' + str(i) + ':0')] = np.array([words['{GO}']])
		feed_dict[graph.get_tensor_by_name('targets' + str(i) + ':0')] = np.array([0])
	feed_dict[feed_previous] = True

	out_predict = sess.run(prediction, feed_dict=feed_dict)
	return to_sentence(out_predict)

# string sentence
def to_ids(sentence):
	ids = []
	for word in sentence.split(' '):
		if not word:
			continue
		if len(ids) >= input_max_size:
			break
		print(word)
		if word in words:
			ids.insert(0, np.array([words[word]]))
	ids.reverse()
	# padding
	while len(ids) < input_max_size:
		ids.insert(0, np.array([words['{PAD}']]))
	return ids

# input_max_size x nparray(1, )
def to_sentence(ids):
	sentence = []
	for index in ids:
		if rev_words[index[0]] == '{EOS}':
			break
		sentence.append(rev_words[index[0]])
	return ' '.join(sentence)