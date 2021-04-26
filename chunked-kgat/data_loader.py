import copy
import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    sent_a, title, sent_b, claim = sentence
    tokens_a = tokenizer.tokenize(sent_a)

    tokens_b = None
    tokens_t = None
    if sent_b and title:
        tokens_t = tokenizer.tokenize(title)
        tokens_b = tokenizer.tokenize(sent_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4 - len(tokens_t))
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens =  ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b and tokens_t:
        tokens = tokens + tokens_t + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + len(tokens_t) + 2)
    #print (tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids

def tok2int_subclaim(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    subclaim, _, _, claim = sentence
    tokens_subclaim = tokenizer.tokenize(subclaim)
    tokens_claim = tokenizer.tokenize(claim)

    _truncate_seq_pair(tokens_claim, tokens_subclaim, max_seq_length - 3)

    tokens =  ["[CLS]"] + tokens_claim + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    tokens = tokens + tokens_subclaim + ["[SEP]"]
    segment_ids += [1] * (len(tokens_subclaim) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def tok2int_list(src_list, tokenizer, max_seq_length, max_subclaims_cnt, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    
    subclaim_ids_padding = list()
    subclaim_msk_padding = list()
    subclaim_seg_padding = list()
    for step_claim, sub_claim_evis in enumerate(src_list):
        sce_inp_padding = list()
        sce_msk_padding = list()
        sce_seg_padding = list()
        
        sce_subclaim_ids_padding = list()
        sce_subclaim_msk_padding = list()
        sce_subclaim_seg_padding = list()
        for step, sent in enumerate(sub_claim_evis):
            input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
            subclaim_ids, subclaim_mask, subclaim_seg = tok2int_subclaim(sent, tokenizer, max_seq_length)
            sce_inp_padding.append(input_ids)
            sce_msk_padding.append(input_mask)
            sce_seg_padding.append(input_seg)
            
            sce_subclaim_ids_padding.append(subclaim_ids)
            sce_subclaim_msk_padding.append(subclaim_mask)
            sce_subclaim_seg_padding.append(subclaim_seg)
        if max_seq_size != -1:
            sce_inp_padding = sce_inp_padding[:max_seq_size]
            sce_msk_padding = sce_msk_padding[:max_seq_size]
            sce_seg_padding = sce_seg_padding[:max_seq_size]
            sce_inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(sce_inp_padding)))
            sce_msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(sce_msk_padding)))
            sce_seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(sce_seg_padding)))
            
            sce_subclaim_ids_padding = sce_subclaim_ids_padding[:max_seq_size]
            sce_subclaim_msk_padding = sce_subclaim_msk_padding[:max_seq_size]
            sce_subclaim_seg_padding = sce_subclaim_seg_padding[:max_seq_size]
            sce_subclaim_ids_padding += ([[0] * max_seq_length] * (max_seq_size - len(sce_subclaim_ids_padding)))
            sce_subclaim_msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(sce_subclaim_msk_padding)))
            sce_subclaim_seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(sce_subclaim_seg_padding)))
        
        inp_padding.append(sce_inp_padding)
        msk_padding.append(sce_msk_padding)
        seg_padding.append(sce_seg_padding)
        
        subclaim_ids_padding.append(sce_subclaim_ids_padding[0])
        subclaim_msk_padding.append(sce_subclaim_msk_padding[0])
        subclaim_seg_padding.append(sce_subclaim_seg_padding[0])
    
    inp_padding += [[[0] * max_seq_length] * max_seq_size] * (max_subclaims_cnt - len(src_list))
    msk_padding += [[[0] * max_seq_length] * max_seq_size] * (max_subclaims_cnt - len(src_list))
    seg_padding += [[[0] * max_seq_length] * max_seq_size] * (max_subclaims_cnt - len(src_list))
    
    subclaim_ids_padding += [[0] * max_seq_length] * (max_subclaims_cnt - len(src_list))
    subclaim_msk_padding += [[0] * max_seq_length] * (max_subclaims_cnt - len(src_list))
    subclaim_seg_padding += [[0] * max_seq_length] * (max_subclaims_cnt - len(src_list))
    return inp_padding, msk_padding, seg_padding, subclaim_ids_padding, subclaim_msk_padding, subclaim_seg_padding


def get_max_subclaims_cnt_in_batch(batch):
    max_length = 0
    for d in batch:
        max_length = max(max_length, len(d))
     
    return max_length

class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, label_map, tokenizer, args, test=False, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.label_map = label_map
        self.threshold = args.threshold
        self.data_path = data_path
        examples = self.read_file(data_path)
        self.examples = examples
        inputs, labels = list(zip(* examples))
        self.inputs = inputs
        self.labels = labels
        self.test = test

        self.total_num = len(examples)
        if self.test:
            self.total_step = self.total_num / batch_size #np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
            self.shuffle()
        self.step = 0


    def read_file(self, data_path):
        examples = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                evi_list = []
                claim = instance['claim']
                for evidence in instance['evidence']:
                    evi_list.append([None, self.process_wiki_title(evidence[0]),
                                     self.process_sent(evidence[2]), self.process_sent(claim)])
                label = self.label_map[instance['label']]
                evi_list = evi_list[:self.evi_num]

                claim_list = list()
                for claim in instance['claims']:
                    claim = self.process_sent(claim)
                    t_evi_list = copy.deepcopy(evi_list)
                    
                    for evi in t_evi_list:
                        evi[0] = claim
                    claim_list.append(t_evi_list)
                examples.append([claim_list, label])
        return examples


    def shuffle(self):
        # randomize the order first
        np.random.shuffle(self.examples)
        # sort based on the #sub-claims. python sort guaranteed to be stable, hence the above randomize function would works
        l = list(self.examples)
        l.sort(key=lambda data: len(data[0]), reverse=True)
        self.examples = np.array(l)

    def process_sent(self, sentence):
        sentence = re.sub(" LSB.*?RSB", "", sentence)
        sentence = re.sub("LRB RRB ", "", sentence)
        sentence = re.sub("LRB", " ( ", sentence)
        sentence = re.sub("RRB", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub("LRB", " ( ", title)
        title = re.sub("RRB", " )", title)
        title = re.sub("COLON", ":", title)
        return title


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            labels = self.labels[self.step * self.batch_size : (self.step+1)*self.batch_size]
            inp_padding_inputs, msk_padding_inputs, seg_padding_inputs = [], [], []
            sc_inp_padding_inputs, sc_msk_padding_inputs, sc_seg_padding_inputs = [], [], [] # for claim+[sep]+subclaim
            max_subclaims_cnt = get_max_subclaims_cnt_in_batch(inputs)
            for step in range(len(inputs)):
                inp, msk, seg, sc_inp, sc_msk, sc_seg = tok2int_list(inputs[step], self.tokenizer, self.max_len, max_subclaims_cnt, self.evi_num)
                inp_padding_inputs += inp
                msk_padding_inputs += msk
                seg_padding_inputs += seg
                
                sc_inp_padding_inputs += sc_inp
                sc_msk_padding_inputs += sc_msk
                sc_seg_padding_inputs += sc_seg

            inp_tensor_input = Variable(
                torch.LongTensor(inp_padding_inputs)).view(-1, max_subclaims_cnt, self.evi_num, self.max_len)
            msk_tensor_input = Variable(
                torch.LongTensor(msk_padding_inputs)).view(-1, max_subclaims_cnt, self.evi_num, self.max_len)
            seg_tensor_input = Variable(
                torch.LongTensor(seg_padding_inputs)).view(-1, max_subclaims_cnt, self.evi_num, self.max_len)
            
            sc_inp_tensor_input = Variable(
                torch.LongTensor(sc_inp_padding_inputs)).view(-1, max_subclaims_cnt, self.max_len)
            sc_msk_tensor_input = Variable(
                torch.LongTensor(sc_msk_padding_inputs)).view(-1, max_subclaims_cnt, self.max_len)
            sc_seg_tensor_input = Variable(
                torch.LongTensor(sc_seg_padding_inputs)).view(-1, max_subclaims_cnt, self.max_len)
            
            lab_tensor = Variable(
                torch.LongTensor(labels))
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
                
                sc_inp_tensor_input = sc_inp_tensor_input.cuda()
                sc_msk_tensor_input = sc_msk_tensor_input.cuda()
                sc_seg_tensor_input = sc_seg_tensor_input.cuda()
                
                lab_tensor = lab_tensor.cuda()
            self.step += 1
            return (inp_tensor_input, msk_tensor_input, seg_tensor_input), (sc_inp_tensor_input, sc_msk_tensor_input, sc_seg_tensor_input), lab_tensor
        else:
            self.step = 0
            if not self.test:
                self.shuffle()
                inputs, labels = list(zip(*self.examples))
                self.inputs = inputs
                self.labels = labels
            raise StopIteration()

class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data_path, label_map, tokenizer, args, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.label_map = label_map
        self.threshold = args.threshold
        self.data_path = data_path
        examples = self.read_file(data_path)
        self.examples = examples
        inputs, ids = list(zip(* examples))
        self.inputs = inputs
        self.ids = ids

        self.total_num = len(examples)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0

    def process_sent(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)

        return sentence

    def process_wiki_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title


    def read_file(self, data_path):
        examples = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                evi_list = []
                claim = instance['claim']
                for evidence in instance['evidence']:
                    evi_list.append([None, self.process_wiki_title(evidence[0]),
                                     self.process_sent(evidence[2]), self.process_sent(claim)])
                id = instance['id']
                evi_list = evi_list[:self.evi_num]

                claim_list = list()
                for claim in instance['claims']:
                    claim = self.process_sent(claim)
                    t_evi_list = copy.deepcopy(evi_list)
                    
                    for evi in t_evi_list:
                        evi[0] = claim
                    claim_list.append(t_evi_list)
                examples.append([claim_list, id])
        return examples


    def shuffle(self):
        # randomize the order first
        np.random.shuffle(self.examples)
        # sort based on the #sub-claims. python sort guaranteed to be stable, hence the above randomize function would works
        l = list(self.examples)
        l.sort(key=lambda data: len(data[0]), reverse=True)
        self.examples = np.array(l)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        if self.step < self.total_step:        
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            ids = self.ids[self.step * self.batch_size : (self.step+1)*self.batch_size]
            inp_padding_inputs, msk_padding_inputs, seg_padding_inputs = [], [], []
            sc_inp_padding_inputs, sc_msk_padding_inputs, sc_seg_padding_inputs = [], [], [] # for claim+[sep]+subclaim
            max_subclaims_cnt = get_max_subclaims_cnt_in_batch(inputs)
            for step in range(len(inputs)):
                inp, msk, seg, sc_inp, sc_msk, sc_seg = tok2int_list(inputs[step], self.tokenizer, self.max_len, max_subclaims_cnt, self.evi_num)
                inp_padding_inputs += inp
                msk_padding_inputs += msk
                seg_padding_inputs += seg
                
                sc_inp_padding_inputs += sc_inp
                sc_msk_padding_inputs += sc_msk
                sc_seg_padding_inputs += sc_seg

            inp_tensor_input = Variable(
                torch.LongTensor(inp_padding_inputs)).view(-1, max_subclaims_cnt, self.evi_num, self.max_len)
            msk_tensor_input = Variable(
                torch.LongTensor(msk_padding_inputs)).view(-1, max_subclaims_cnt, self.evi_num, self.max_len)
            seg_tensor_input = Variable(
                torch.LongTensor(seg_padding_inputs)).view(-1, max_subclaims_cnt, self.evi_num, self.max_len)
            
            sc_inp_tensor_input = Variable(
                torch.LongTensor(sc_inp_padding_inputs)).view(-1, max_subclaims_cnt, self.max_len)
            sc_msk_tensor_input = Variable(
                torch.LongTensor(sc_msk_padding_inputs)).view(-1, max_subclaims_cnt, self.max_len)
            sc_seg_tensor_input = Variable(
                torch.LongTensor(sc_seg_padding_inputs)).view(-1, max_subclaims_cnt, self.max_len)
            
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
                
                sc_inp_tensor_input = sc_inp_tensor_input.cuda()
                sc_msk_tensor_input = sc_msk_tensor_input.cuda()
                sc_seg_tensor_input = sc_seg_tensor_input.cuda()
            self.step += 1
            return (inp_tensor_input, msk_tensor_input, seg_tensor_input), (sc_inp_tensor_input, sc_msk_tensor_input, sc_seg_tensor_input), ids
        else:
            self.step = 0
            raise StopIteration()
