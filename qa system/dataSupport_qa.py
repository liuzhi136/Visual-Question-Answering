# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

# words is the unique words list. this method use to generate two map: word to num and num to word.
def revocab(words):
    revocab = {v:k for k, v in words.items()}
    return revocab

def default_text2id(sentence, the_vocab):
    print(sentence)
    words = sentence.split(' ')
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words

def default_gen_buckets(sentences, batch_size, the_vocab):
    len_dict = {}
    max_len = -1
    # count the number of sentence for each unique length
    for sentence in sentences:
        words = default_text2id(sentence, the_vocab)
        if len(words) == 0:
            continue
        if len(words) > max_len:
            max_len = len(words)
        if len(words) in len_dict:
            len_dict[len(words)] += 1
        else:
            len_dict[len(words)] = 1
    print('the default generated buckets:\n', len_dict)

    tl = 0
    buckets = []
    # create each bucket by batch_size, this operation will be merge different length of sentence into one bucket
    # when the number of len less than batch_size
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this    
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets



class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class SequencesIterQA(mx.io.DataIter):
    def __init__(self, QApath, imagePath, vocab, re_vocab, image_feat, 
                 feat_id_map, buckets, batch_size, init_states, batch_img_feats,
                 data_name='data', label_name='label',
                 text2id=None, read_content=None, id2text=None):
        super(SequencesIterQA, self).__init__()

        if text2id == None:
            self.text2id = default_text2id
        else:
            self.text2id = text2id
        if id2text != None: self.id2text = id2text
        if read_content == None:
            self.read_content = default_read_content
        else:
            self.read_content = read_content

        # read training set data.
        sentences = self.read_content(QApath)
        images = self.read_content(imagePath)
        
        print('Build buckets!')
        if len(buckets) == 0:
            # these buckets contains all each unique length in the training data for each bucket.
            buckets = default_gen_buckets(sentences, batch_size, vocab)
        # the length of all unique word
        self.vocab_size = len(vocab)
        self.data_name = data_name
        self.label_name = label_name
        self.image_feats = image_feat
        self.feat_id_map = feat_id_map
        self.images = images
        self.batch_img_feats = batch_img_feats

        # Beacuse of each bucket correspond to a different length of sentence, sort buckets by its length.
        buckets.sort()
        self.buckets = buckets
        # create a list of data corresponding to the buckets'length
        self.data = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        for sentence in sentences:
            sentence = self.text2id(sentence, vocab)
            if len(sentence) == 0:
                continue
            # if current sentence's length less than or equal to the ith bucket, then put the sentence into the ith data list
            # after this for loop done, self.data contains len(buckets) sentence list.
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    self.data[i].append(sentence)
                    break
            # we just ignore the sentence it is longer than the maximum
            # bucket size here
        
        # convert data into ndarrays for better speed during training
        data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                sentence = self.data[i_bucket][j]
                data[i_bucket][j, :len(sentence)] = sentence
        self.data = data

        # Get the size of each bucket, so that we could sample uniformly from the bucket. 
        # In other words, get the number of sentence for each data bucket in self.data
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            # print the number of sentence for each bucket
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        print('init_states: {0}'.format(init_states))
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key))] + init_states + [batch_img_feats]
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            # calculate the number to truncate by batch_size for each bucket
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
            # recount the sentence in ith bucket 
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]

        # make a bucket plan and its element is the form [0...0, 1...1, 2...2...] 0,1,2 indicates the ith bucket and n is the number for truncation
        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        print('-------bucket_plan---------: \n{0}'.format(bucket_plan))
        np.random.shuffle(bucket_plan)
        # for each sentence set x in self.data, we generate len(x) number in the range from 0 to len(x)-1 in random order.
        # each number correspond to a sentence's index.
        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        # initialize the start id, the id start from 0 for each sentence set x in self.data
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = []
        # for each bucket, initilize the data and label
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))
            self.data_buffer.append(data)
            self.label_buffer.append(label)

    def __iter__(self):

        # for each bucket, get one block sentences
        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            i_idx = self.bucket_curr_idx[i_bucket]
            # get batch_size sentence index from the all_idx
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            
            # print('*********this perform data parallelism************')
            init_state_names = [x[0] for x in self.init_states]
            data[:] = self.data[i_bucket][idx]
            self.batch_img_feats_array = self.get_image_feats(idx)
            label = self.label_buffer[i_bucket]
            label[:, :-1] = data[:, 1:]
            label[:, -1] = 0

            for sentence in data:
                assert len(sentence) == self.buckets[i_bucket]
                       
            data_all = [mx.nd.array(data)] + self.init_state_arrays + [mx.nd.array(self.batch_img_feats_array)]
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names + [self.batch_img_feats[0]]
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all, self.buckets[i_bucket])
            yield data_batch


    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
    
    def get_image_feats(self, idx):
        img_feats = np.zeros((self.batch_size, self.image_feats.shape[0]))
        for i in range(len(idx)):
            # id = idx[i]
            # img_id = self.images[id]
            # map_id = self.feat_id_map[img_id]
            # print('img id: ', img_id, 'map_id: ', map_id)
            img_feats[i, :] = self.image_feats[:, self.feat_id_map[self.images[idx[i]]]]
        return img_feats