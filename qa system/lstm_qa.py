# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam2 = namedtuple("LSTMParam2", ["i2h_weight_f", "i2h_bias_f", "h2h_weight_f", "h2h_bias_f",
                                       "i2h_weight_i", "i2h_bias_i", "h2h_weight_i", "h2h_bias_i",
                                       "i2h_weight_t", "i2h_bias_t", "h2h_weight_t", "h2h_bias_t",
                                       "i2h_weight_o", "i2h_bias_o", "h2h_weight_o", "h2h_bias_o",
                                       ])                                    

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """ LSTM Cell symbol """
    if dropout > 0:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    # forget gate
    i2h_f = mx.sym.FullyConnected(data=indata, weight=param.i2h_weight_f, bias=param.i2h_bias_f,
                                    num_hidden=num_hidden, name="t{0}_l{1}_i2h_f".format(seqidx, layeridx))
    h2h_f = mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight_f, bias=param.h2h_bias_f,
                                    num_hidden=num_hidden, name="t{0}_l{1}_h2h_f".format(seqidx, layeridx))
    forget_gate = mx.sym.Activation(i2h_f+h2h_f, act_type="sigmoid")
    # input gate
    i2h_i = mx.sym.FullyConnected(data=indata, weight=param.i2h_weight_i, bias=param.i2h_bias_i,
                                    num_hidden=num_hidden, name="t{0}_l{1}_i2h_i".format(seqidx, layeridx))
    h2h_i = mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight_i, bias=param.h2h_bias_i,
                                    num_hidden=num_hidden, name="t{0}_l{1}_h2h_i".format(seqidx, layeridx))
    in_gate = mx.sym.Activation(i2h_i+h2h_i, act_type="sigmoid")
    # transform gate
    i2h_t = mx.sym.FullyConnected(data=indata, weight=param.i2h_weight_t, bias=param.i2h_bias_t,
                                    num_hidden=num_hidden, name="t{0}_l{1}_i2h_t".format(seqidx, layeridx))
    h2h_t = mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight_t, bias=param.h2h_bias_t,
                                    num_hidden=num_hidden, name="t{0}_l{1}_h2h_t".format(seqidx, layeridx))
    transform_gate = mx.sym.Activation(i2h_t+h2h_t, act_type="sigmoid")
    # output gate
    i2h_o = mx.sym.FullyConnected(data=indata, weight=param.i2h_weight_o, bias=param.i2h_bias_o,
                                    num_hidden=num_hidden, name="t{0}_l{1}_i2h_o".format(seqidx, layeridx))
    h2h_o = mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight_o, bias=param.h2h_bias_o,
                                    num_hidden=num_hidden, name="t{0}_l{1}_h2h_o".format(seqidx, layeridx))
    out_gate = mx.sym.Activation(i2h_o+h2h_o, act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * transform_gate)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


# we define a new unrolling function here because the original
# one in lstm.py concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def unroll_lstm(num_lstm_layer, seq_len, input_size, batch_size,
                num_hidden, num_embed, num_label, dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    pred_weight = mx.sym.Variable("pred_weight")
    pred_bias = mx.sym.Variable("pred_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam2(i2h_weight_f=mx.sym.Variable("l{0}_i2h_f_weight".format(i)),
                                      i2h_bias_f=mx.sym.Variable("l{0}_i2h_f_bias".format(i)),
                                      h2h_weight_f=mx.sym.Variable("l{0}_h2h_f_weight".format(i)),
                                      h2h_bias_f=mx.sym.Variable("l{0}_h2h_f_bias".format(i)),
                                      i2h_weight_i=mx.sym.Variable("l{0}_i2h_i_weight".format(i)),
                                      i2h_bias_i=mx.sym.Variable("l{0}_i2h_i_bias".format(i)),
                                      h2h_weight_i=mx.sym.Variable("l{0}_h2h_i_weight".format(i)),
                                      h2h_bias_i=mx.sym.Variable("l{0}_h2h_i_bias".format(i)),
                                      i2h_weight_t=mx.sym.Variable("l{0}_i2h_t_weight".format(i)),
                                      i2h_bias_t=mx.sym.Variable("l{0}_i2h_t_bias".format(i)),
                                      h2h_weight_t=mx.sym.Variable("l{0}_h2h_t_weight".format(i)),
                                      h2h_bias_t=mx.sym.Variable("l{0}_h2h_t_bias".format(i)),
                                      i2h_weight_o=mx.sym.Variable("l{0}_i2h_o_weight".format(i)),
                                      i2h_bias_o=mx.sym.Variable("l{0}_i2h_o_bias".format(i)),
                                      h2h_weight_o=mx.sym.Variable("l{0}_h2h_o_weight".format(i)),
                                      h2h_bias_o=mx.sym.Variable("l{0}_h2h_o_bias".format(i)),
                                    ))
        
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    img_feat = mx.sym.Variable('img_batch_feats')
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    # print('seq_len: {0}, embed_output_dim: {1}'.format(seq_len, num_embed))
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    # this hidden_all append hidden according the input data's order
    for seqidx in range(seq_len):
        hidden = mx.sym.Concat(*[img_feat,wordvec[seqidx]], dim=1)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            x = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)
    
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=pred_weight, bias=pred_bias, name='pred')

    ################################################################################
    # Make label the same shape as our produced data path
    # I did not observe big speed difference between the following two ways

    # label = mx.sym.transpose(data=label)
    # label = mx.sym.Reshape(data=label, shape=(-1,))

    # in order to keep consistent with input data'hidden_all order, the label also concate label's data in the same order
    label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
    label = [label_slice[t] for t in range(seq_len)]
    label = mx.sym.Concat(*label, dim=0)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    ################################################################################

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm

def lstm_inference_symbol(num_lstm_layer, input_size,
                          num_hidden, num_embed, num_label, dropout=0.):
    seqidx = 0
    embed_weight=mx.sym.Variable("embed_weight")
    pred_weight = mx.sym.Variable("pred_weight")
    pred_bias = mx.sym.Variable("pred_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam2(i2h_weight_f=mx.sym.Variable("l{0}_i2h_f_weight".format(i)),
                                      i2h_bias_f=mx.sym.Variable("l{0}_i2h_f_bias".format(i)),
                                      h2h_weight_f=mx.sym.Variable("l{0}_h2h_f_weight".format(i)),
                                      h2h_bias_f=mx.sym.Variable("l{0}_h2h_f_bias".format(i)),
                                      i2h_weight_i=mx.sym.Variable("l{0}_i2h_i_weight".format(i)),
                                      i2h_bias_i=mx.sym.Variable("l{0}_i2h_i_bias".format(i)),
                                      h2h_weight_i=mx.sym.Variable("l{0}_h2h_i_weight".format(i)),
                                      h2h_bias_i=mx.sym.Variable("l{0}_h2h_i_bias".format(i)),
                                      i2h_weight_t=mx.sym.Variable("l{0}_i2h_t_weight".format(i)),
                                      i2h_bias_t=mx.sym.Variable("l{0}_i2h_t_bias".format(i)),
                                      h2h_weight_t=mx.sym.Variable("l{0}_h2h_t_weight".format(i)),
                                      h2h_bias_t=mx.sym.Variable("l{0}_h2h_t_bias".format(i)),
                                      i2h_weight_o=mx.sym.Variable("l{0}_i2h_o_weight".format(i)),
                                      i2h_bias_o=mx.sym.Variable("l{0}_i2h_o_bias".format(i)),
                                      h2h_weight_o=mx.sym.Variable("l{0}_h2h_o_weight".format(i)),
                                      h2h_bias_o=mx.sym.Variable("l{0}_h2h_o_bias".format(i)),
                                    ))
        
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    data = mx.sym.Variable("data")
    img_feat = mx.sym.Variable('img_batch_feats')

    wordvec = mx.sym.Embedding(data=data,
                              input_dim=input_size,
                              output_dim=num_embed,
                              weight=embed_weight,
                              name="embed")
    hidden = [img_feat, wordvec]
    # stack LSTM
    for i in range(num_lstm_layer):
        if i==0:
            dp=0.
        else:
            dp = dropout
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[i],
                          param=param_cells[i],
                          seqidx=seqidx, layeridx=i, dropout=dp)
        hidden = next_state.h
        last_states[i] = next_state
    # decoder
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)
    fc = mx.sym.FullyConnected(data=hidden, num_hidden=num_label,
                               weight=pred_weight, bias=pred_bias, name='pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)
