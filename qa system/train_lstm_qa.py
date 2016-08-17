import mxnet as mx
import numpy as np
import scipy.io
import re
import logging
import matplotlib.pyplot as plt
import os

import lstm_qa as lstm
from dataSupport_qa import SequencesIterQA, revocab 

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# read each line from the target file and extract unique word
def gen_vocab(readfile, vocab=None):
    if vocab is None:
        vocab = {}
        vocab['#'] = 0
    print('vocab len: {0}'.format(len(vocab)))
    idx = len(vocab)
    content = read_content(readfile)
    words = []
    for line in content:
        if len(re.findall('[0-9]', line)) > 0 or len(re.findall('[()\']', line)) > 0:
            tmp = [word for word in re.split('\s+|[,!?;]', line) if word.strip()]
        else:
            tmp = [word for word in re.split('\s+|[,!?;"()]', line) if word.strip()]
        words.extend(tmp)
    
    print('text: {0}'.format(len(words)))

    for character in words:
        if len(character) == 0: continue
        if character not in vocab:
           vocab[character] = idx
           idx += 1
    return vocab

# Read from doc, each line is a element in content
def read_content(path):
    lines = [line.strip() for line in open(path).readlines() if line.strip()]
    return lines
    

def text2id(sentence, the_vocab):
    # print('before: ', sentence)
    if len(sentence) == 0: return []
    if len(re.findall('[0-9]', sentence)) > 0 or len(re.findall('[()\']', sentence)) > 0:
        words = [word for word in re.split('\s+|[,!?;]', sentence) if word.strip()]
    else:
        words = [word for word in re.split('\s+|[,!?;"()]', sentence) if word.strip()]    
    words = [the_vocab[w] for w in words if len(w) > 0]
    # print('after: ', words)
    return words

def splitwords(sentencelist):
    if len(sentencelist) == 0: return []
    wordslist = []
    for sentence in sentencelist:
        if len(sentence) == 0: continue
        if len(re.findall('[0-9]', sentence)) > 0 or len(re.findall('[()\']', sentence)) > 0:
            words = [word for word in re.split('\s+|[,!?;]', sentence) if word.strip()]
        else:
            words = [word for word in re.split('\s+|[,!?;"()]', sentence) if word.strip()]
        wordslist.append(words)
    return wordslist

# Evaluation metric 1
def Perplexity(label, pred):
    # print('pred\'s shape: {0}'.format(pred.shape))
    # nums = []
    # print('label: {0}'.format(idx2text(num2word, label[:,0])))

    # label = np.array([label[i] for i in range(label.shape[1])])
    # label = label.reshape((-1,))
    label = label.T.reshape((-1,))
    # print('label i shape: {0}'.format(label.shape))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

# Evaluation metric 2
def CrossEntropySoftmax(label, pred):
    # nums = []   
    label = label.T.reshape((-1),)
    # print('label\'shape: {0}'.format(label.shape))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(pred[i][int(label[i])] + 1e-8)
    return loss / pred.shape[0]

def idx2text(re_vocab, nums):
    text = [re_vocab[num] for num in nums]
    return text

# save vocabulary
def save_vocab(savefile, vocabulary):
    with open(savefile, 'wt') as writehandle:
        for word in vocabulary:
            writehandle.write(word+' ')

# save dict data
def save_dict(savefile, data):
    if not type(data) == dict:
        raise TypeError('The method need a dict type data.')
    with open(savefile, 'wt') as writehandle:
        for k,v in data.items():
            record = k+':'+str(v)+'\n'
            writehandle.write(record)

if __name__ == '__main__':
    print('************Strat Training*************')

    qa_train = '../data/coco_qa/concateQA/concateqa_train.txt'
    qa_val = '../data/coco_qa/concateQA/concateqa_val.txt'
    image_train = '../data/coco_qa/images/train/images_train2014.txt'
    vgg_feats_path = '../data/coco_qa/image_fatures/vgg_feats.mat'
    image_map_ids = '../data/coco_qa/image_fatures/coco_vgg_IDMap.txt'

    # build vocabulary according to the questions and answers in training set and validation set
    vocabulary = gen_vocab(qa_train)
    print('only train: {0}'.format(len(vocabulary)))
    vocabulary = gen_vocab(qa_val, vocabulary)
    print('train and val: {0}'.format(len(vocabulary)))
    # print(sorted(vocabulary.items(), key=lambda vocab:vocab[1], reverse=False))
    re_vocab = revocab(vocabulary)

    # load image features
    image_feats = scipy.io.loadmat(vgg_feats_path)['feats']
    images_ids = read_content(image_map_ids)
    image_id = {}# key is str, value is int
    for line in images_ids:
        tmp = line.split(' ')
        image_id[tmp[0]] = int(tmp[1])
   
    batch_size = 32
    num_hidden = 512
    num_embed = 256
    num_lstm_layer = 2
    content = read_content(qa_train)
    content = [len(words) for words in splitwords(content)]
    buckets = [max(content) + 1]
    # buckets = []

    print('maximum of qa: {0}'.format(buckets[0]))

    num_epoch = 75
    lr = 0.00005
    momentum = 0.9

    params = dict()
    params['num_hidden'] = num_hidden
    params['num_lstm_layer'] = num_lstm_layer
    params['num_embed'] = num_embed
    save_dict('../param/params', params) # save network's parameters

    # ctx = mx.gpu()
    devs = [mx.context.gpu(i) for i in range(1)]
    save_vocab('../param/vocabulary', vocabulary) # save those unique words
    print('len of vocabulary: ', len(vocabulary))

    init_c = [('l{0}_init_c'.format(l), (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l{0}_init_h'.format(l), (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    batch_img_feats = ('img_batch_feats', (batch_size, image_feats.shape[0]))
    init_states = init_c + init_h
    # print('at the start file: {0}'.format(init_states))

    trainIter = SequencesIterQA(qa_train, image_train, vocabulary, re_vocab, 
                                image_feats, image_id, buckets, batch_size, init_states, batch_img_feats,
                                text2id=text2id, read_content=read_content, id2text=idx2text)

    def gen_sym(seq_len):
        # # seq_len because input word seq_len dosen't need the last word and label seq_len dosen't need the fisrt word
        # # used for buickets is not []
        # seq_len because input word seq_len dosen't need the last word and label seq_len dosen't need the fisrt word
        return lstm.unroll_lstm(num_lstm_layer, seq_len, len(vocabulary), batch_size,
                                num_hidden, num_embed, len(vocabulary), dropout=0.5)

    if len(buckets) == 1:
        symbol = gen_sym(buckets[0])
    else:
        symbol = gen_sym
    
    # load model if the pre-trained model is existed
    model_prefix = './model/QAmodel'
    load_epoch = 11
    model_args = {}
    if model_prefix is not None and load_epoch is not None:
        print('load previous model.')
        tmp = mx.model.FeedForward.load(model_prefix, load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params}
    rescale_grad = 1. / batch_size

    # model_args['learning_rate'] = lr
    # model_args['wd'] = 0.0002
    # model_args['momentum'] = momentum
    
    optimizer = mx.optimizer.Adam(learning_rate = lr, wd = 0.0002, rescale_grad = rescale_grad)  
    model = mx.model.FeedForward(
        ctx=mx.gpu(),
        symbol=symbol,
        optimizer=optimizer,
        num_epoch=num_epoch,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args
    )

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    save_model_prefix = './model/QAmodel'
    checkpoint = mx.callback.do_checkpoint(save_model_prefix)

    model.fit(
        X = trainIter,
        eval_metric = mx.metric.np(CrossEntropySoftmax),
        epoch_end_callback = checkpoint,
        batch_end_callback = mx.callback.Speedometer(batch_size, 50)
    )
    