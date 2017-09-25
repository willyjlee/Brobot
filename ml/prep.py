from __future__ import print_function
import numpy as np
import pickle

max_len = 8

special = [',', '.', '?', ':', '!', '"', '\'']
read_special = ['{GO}', '{EOS}', '{PAD}']

def add_dict(dict, word, index, rev_dict):
    if word not in dict:
        dict[word] = index
        rev_dict[index] = word
        index = index + 1
    return index

def save_words(sentences):

    words = dict()
    rev_words = dict()
    index = 0
    for sentence in sentences:
        for word in sentence.split(' '):
            if not word:
                continue
            # check special characters at end?
            index = add_dict(words, word, index, rev_words)

    for token in read_special:
        index = add_dict(words, token, index, rev_words)
    with open('data/words.pickle', 'wb') as fw:
        pickle.dump(words, fw, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/rev_words.pickle', 'wb') as rfw:
        pickle.dump(rev_words, rfw, protocol=pickle.HIGHEST_PROTOCOL)
    print('words:', len(words))
    print('rev_words', len(rev_words))
    return words

def sentences():
    arr = []
    with open('data/movie_lines.txt', 'rb') as f:
        lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue
            tokens = line.rstrip('\n').split(' +++$+++ ')
            sentence = tokens[4].strip()
            arr.append(sentence)
    return arr

# encode sentences with ids
# 2 np arrays [num_sentence/2, max_len]
def translate(sentences, word_dict):
    input = np.full((len(sentences), max_len), -1, dtype='int32')
    output = np.full((len(sentences), max_len), -1, dtype='int32')

    index = 0
    for i, sentence in enumerate(sentences):
        if i + 1 >= len(sentences):
            break
        words = [word for word in sentence.split(' ') if word]
        out_words = [word for word in sentences[i+1].split(' ') if word]
        # id input sentences
        # taking part of long sequences
        if len(words) > max_len or len(out_words) > max_len - 2:
            continue
        in_list = []
        in_length = min(len(words), max_len)
        for i in range(max_len - in_length):
            in_list.append(word_dict['{PAD}'])
        for i in range(in_length):
            in_list.append(word_dict[words[in_length - 1 - i]])

        # don't ignore too long sequences
        # out_list = [word_dict['{GO}']]
        out_list = []
        for i in range(max_len):
            if i < len(out_words):
                out_list.append(word_dict[out_words[i]])
            else:
                out_list.append(word_dict['{EOS}'])
                break
        while len(out_list) < max_len:
            out_list.append(word_dict['{PAD}'])

        # debugging
        # print('Input:', get_sentence(in_list, rev_words))
        # print('Output:', get_sentence(out_list, rev_words))

        input[index, :] = in_list
        output[index, :] = out_list
        index = index + 1

    input = input[:index]
    output = output[:index]
    np.save('data/input.npy', input)
    np.save('data/output.npy', output)

sentences = sentences()
words = save_words(sentences)
translate(sentences, words)
