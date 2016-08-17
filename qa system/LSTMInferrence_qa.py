import mxnet as mx
import numpy as np

from lstm_qa import LSTMParam2, LSTMState, lstm_inference_symbol

class LSTMInferrenceModel(object):

    def __init__(self, num_lstm_layer, input_size, num_hidden,
                 image_feats, num_embed, label_dim, arg_params, images=None, ctx=mx.cpu(), dropout=0.):
        
        self.sym = lstm_inference_symbol(
            num_lstm_layer, input_size, num_hidden, num_embed, label_dim, dropout
        )
        batch_size = 1
        self.image_feats = image_feats
        self.batch_size = batch_size
        # all image names
        if images is not None:
            self.images = images

        init_c = [('l{0}_init_c'.format(i), (batch_size, num_hidden)) for i in range(num_lstm_layer)]
        init_h = [('l{0}_init_h'.format(i), (batch_size, num_hidden)) for i in range(num_lstm_layer)]
        img_batch_feats = ('img_batch_feats', (batch_size, image_feats.shape[0]))

        data_shape = [('data', (batch_size,))]
        input_shape = dict(init_c + init_h + data_shape + img_batch_feats)

        self.executor = self.sym.simple_bind(ctx = mx.cpu(), **input_shape)

        for key in self.executor.arg_dict.keys():
            if key in arg_params:
                print('key: {0}\nvalue:{1}, executor\'s shape: {2}'.format(key, arg_params[key].shape, self.executor.arg_dict[key].shape))
                arg_params[key].copyto(self.executor.arg_dict[key])

        state_name = []
        for i in range(num_lstm_layer):
            state_name.append('l{0}_init_c'.format(i))
            state_name.append('l{0}_init_h'.format(i))
        self.state_dict = dict(zip(state_name, self.executor.outputs[1:]))

        self.input_arr = mx.nd.zeros(data_shape[0][1])
    
    # each forward will accept a word'index and return a vector of probability of next word
    def forward(self, input_data, img_id=None, new_seq=False):
        if new_seq:
            for key in self.state_dict.keys():
                self.executor.arg_dict[key][:] = 0.

        if img_id is not None:
            batch_img_feat = mx.nd.array(self.get_image_feats(img_id))
        else:
            batch_img_feat = ma.nd.array(self.image_feats)
        batch_img_feat.copyto(self.executor.arg_dict['img_batch_feats']) # fetch the image feature corresponding the question
        input_data.copyto(self.executor.arg_dict['data'])
        
        self.executor.forward()
        # for k,v in self.state_dict.items():
        #     print(k, ': ', v.asnumpy())
        for key in self.state_dict.keys():
            self.state_dict[key].copyto(self.executor.arg_dict[key])
        prob = self.executor.outputs[0].asnumpy()
        return prob
    
    def get_image_feats(self, idx):
        img_feats = np.zeros((self.batch_size, self.image_feats.shape[0]))
        if not isinstance(idx, list):# if idx is a number, then put it in a list
            idx = [idx]
        for i in range(len(idx)):
            # id = idx[i]
            # img_id = self.images[id]
            # map_id = self.feat_id_map[img_id]
            # print('img id: ', img_id, 'map_id: ', map_id)
            img_feats[i, :] = self.image_feats[:, self.feat_id_map[self.images[i]]]
        return img_feats
        