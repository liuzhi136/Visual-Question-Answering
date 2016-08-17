import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import extract_feature
import os

from PIL import Image
from LSTMInferrence_qa import LSTMInferrenceModel

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


if __name__ == '__main__':

    # these two qa files use to build dictionary.
    qa_train = '../data/coco_qa/concateQA/concateqa_train.txt'
    qa_val = '../data/coco_qa/concateQA/concateqa_val.txt'
    params_file = './param/params'

    # build vocabulary according to the questions and answers in training set and validation set
    vocabulary = gen_vocab(qa_train)
    print('only train: {0}'.format(len(vocabulary)))
    vocabulary = gen_vocab(qa_val, vocabulary)
    print('train and val: {0}'.format(len(vocabulary)))
    re_vocab = revocab(vocabulary)
    params = load_dict_data(params_file)

    # print('vocabulary:\n{0}'.format(len(vocabulary)))

    # load cnn model
    # extract image feature
    load_cnn_epoch = 1
    model_prefix = './model/cnn_model/Inception-7'
    cnn_model = mx.model.FeedForward.load(model_prefix, load_cnn_epoch, ctx=mx.gpu())
    internals = cnn_model.symbol.get_internals()
    fea_symbol = internals['flatten_output']
    feature_extractor =  mx.model.FeedForward(ctx = mx.gpu(), symbol = fea_symbol, numpy_batch_size=1,
                                             arg_params = cnn_model.arg_params, aux_params = cnn_model.aux_params,
                                             allow_extra_params = True)
    
    image_path = input('please input the image path').strip()
    path = '/media/leo/新加卷/qa images/train2014/COCO_train2014_000000000009.jpg' # used to test
    img_feat = extract_feature(image_path, feature_extractor)
    # print(img_feat.shape)

    # load LSTM model form check_point
    load_lstm_epoch = 11
    __, lstm_arg_params, __ = mx.model.load_checkpoint('./model/QAmodel', load_epoch)

    # build an inferential model
    model = LSTMInferrenceModel(
            num_lstm_layer = params['num_lstm_layer'], input_size = len(vocabulary),
            num_hidden = params['num_hidden'], image_feats = img_feat, num_embed = params['num_embed'],
            label_dim = len(vocabulary), arg_params = lstm_arg_params, dropout=0.5
        )
    # it may raise an error 
    # because the trained lstm network used image features prodeced by vggnet 
    # whose feature dimension is 4096
    # and this program use google net to extract image feature, 
    # which will prodece 2048 dimensions feature vector

    # I suggest that if you want to perform this program interactively, 
    # you should use this google net to extract all training and testing image features firstly.
    # then, you should use these features to train a new lstm network.
    # after that, you can correctly perform this interactive program.
    
    while True:
        question = input('please input question').strip()
        endchar = '#'

        indata = mx.nd.zeros((1,))
        next_char = ''
        i = 0
        newSentence = True
        ques = text2id(question, vocabulary)
        outputs = []
        ignore_length = len(ques)
        # produce predicted answer
        while next_char != endchar:
            
            if i <= ignore_length - 1:
                next_char = ques[i]
            else:
                next_char = outputs[-1]
            makeInput(next_char, vocabulary, indata)

            prob = model.forward(indata, newSentence)
            newSentence = False

            if i >= ignore_length - 1:
                next_char = makeOutput(prob, revocab, vocabulary)
                if next_char == '#': break
                outputs.append(next_char)
            i += 1
            
        # show current image and predicted answer
        img = Image.open(image_path)
        plt.figure(target_real_image.split('.')[0])
        plt.title('answers: {0}\npredicted answers: {1}'.format(a, ' '.outputs))
        plt.imshow(img)
        plt.show()