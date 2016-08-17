import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random
import bisect

from PIL import Image
from LSTMInferrence_qa import LSTMInferrenceModel
from dataSupport_qa import revocab

# read each line from the target file and extract unique word
def gen_vocab(readfile, vocab=None):
    if vocab is None:
        vocab = {}
        vocab['#'] = 0
    idx = len(vocab)
    content = read_content(readfile)
    words = []
    for line in content:
        if len(re.findall('[0-9]', line)) > 0 or len(re.findall('[()\']', line)) > 0:
            tmp = [word for word in re.split('\s+|[,!?;]', line) if word.strip()]
        else:
            tmp = [word for word in re.split('\s+|[,!?;"()]', line) if word.strip()]
        words.extend(tmp)

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
    
    if len(sentence) == 0: return []
    if len(re.findall('[0-9]', sentence)) > 0 or len(re.findall('[()\']', sentence)) > 0:
        words = [word for word in re.split('\s+|[,!?;]', sentence) if word.strip()]
    else:
        words = [word for word in re.split('\s+|[,!?;"()]', sentence) if word.strip()]    
    words = [the_vocab[w] for w in words if len(w) > 0]
    
    return words
    
def load_dict_data(loadfile, separator=':'):
    params = {}
    for line in open(loadfile, 'r').readlines():
        k_v = line.strip().split(separator)
        params[k_v[0]] = int(k_v[1])
    return params

def makeInput(word, word2num, arr):
    if type(word) == int:
        ind = word
    else:
        if len(word) == 0:
            return -1
        ind = word2num[word]
        
    tmp = np.zeros((1,))
    tmp[0] = ind
    arr[:] = tmp

def makeOutput(prob, num2word):
    ind = np.maxarg(prob)[0]

    try:
        char = num2word[ind]
    except:
        char = ''
    return char

def isEqual(correct, predict):
    if len(correct) == 0 or len(predict) == 0:
        return False
    if not isinstance(correct, list):
        if len(re.findall('[0-9]', correct)) > 0 or len(re.findall('[()\']', correct)) > 0:
            correct = [word for word in re.split('\s+|[,!?;]', correct) if word.strip()]
        else:
            correct = [word for word in re.split('\s+|[,!?;"()]', correct) if word.strip()]
    for c, p in zip(correct, predict):
        if c != p:
            return False
    return True

if __name__ == '__main__':

    # these two qa files use to build dictionary.
    qa_train = '../data/coco_qa/concateQA/concateqa_train.txt'
    qa_val = '../data/coco_qa/concateQA/concateqa_val.txt'
    questions_val = '../data/coco_qa/questions/val/questions_val2014.txt'
    answers_val = '../data/coco_qa/answers/val/answers_val2014_modal.txt'
    image_val = '../data/coco_qa/images/val/images_val2014_all.txt'
    real_images = '/media/leo/qa\ images/val2014/'
    vgg_feats_path = '../data/coco_qa/image_fatures/vgg_feats.mat'
    image_map_ids = '../data/coco_qa/image_fatures/coco_vgg_IDMap.txt'
    params_file = './param/params'
    
    ques_val = read_content(questions_val)
    ans_val = read_content(answers_val)
    img_val = read_content(imagePath)
    image_names = os.listdir(real_images)

    # build vocabulary according to the questions and answers in training set and validation set
    vocabulary = gen_vocab(qa_train)
    print('only train: {0}'.format(len(vocabulary)))
    vocabulary = gen_vocab(qa_val, vocabulary)
    print('train and val: {0}'.format(len(vocabulary)))
    re_vocab = revocab(vocabulary)
    params = load_dict_data(params_file)

    # load image features
    img_feats = scipy.io.loadmat(vgg_feats_path)['feats']
    images_ids = read_content(image_map_ids)
    image_id = {}# key is str, value is int
    for line in images_ids:
        tmp = line.split(' ')
        image_id[tmp[0]] = int(tmp[1])

    print('vocabulary:\n{0}'.format(len(vocabulary)))

    # load model form check_point
    load_epoch = None
    __, arg_params, __ = mx.model.load_checkpoint('./model/QAmodel', load_epoch)

    # build an inferential model
    model = LSTMInferrenceModel(
            num_lstm_layer = params['num_lstm_layer'], input_size = len(vocabulary),
            num_hidden = params['num_hidden'], image_feats = img_feats, num_embed = params['num_embed'],
            images = images_val, label_dim = len(vocabulary), arg_params = arg_params, dropout=0.5
        )
    
    endchar = '#'
    accuracy = 0
    for i in range(len(ques_val)):
        q = ques_val[i]
        a = ans_val[i]
        img = img_val[i]

        indata = mx.nd.zeros((1,))
        next_char = ''
        i = 0
        newSentence = True
        ques = text2id(q, vocabulary)
        outputs = []
        ignore_length = len(ques)
        # produce predicted answer
        while next_char != endchar:
            
            if i <= ignore_length - 1:
                next_char = ques[i]
            else:
                next_char = outputs[-1]
            makeInput(next_char, vocabulary, indata)

            prob = model.forward(indata, img_id, newSentence)
            newSentence = False

            if i >= ignore_length - 1:
               next_char = makeOutput(prob, revocab, vocabulary)
               if next_char == '#': break
               outputs.append(next_char)
            i += 1
        # count the correct prediction
        if (isEqual(a, outputs)): accuracy += 1
        
        # show current validation image
        target_real_image = real_images
        for img_name in image_names:
            if img in img_name:
                target_real_image += img_name
                break
        img = Image.open(target_real_image)
        plt.figure(target_real_image.split('.')[0])
        plt.title('answers: {0}\npredicted answers: {1}'.format(a, ' '.outputs))
        plt.imshow(img)
        plt.show()
    
    # print the accuracy on validation set
    print('The accuracy on validation: {0}%'.format(accuracy))

