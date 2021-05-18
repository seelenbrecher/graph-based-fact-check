import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable

from prepare_concept import GraphConstructor
from prepare_concept import CONCEPT_DUMMY_IDX

def do_need_append_token(span, concept):
    span = str(span).lower()
    word_concept = concept[2].lower() # get the ori word, not the concept
    
    if span == word_concept: # exact match, no need to append more token
        return False
    if not word_concept.startswith(span+' '): # current span is completly different with the concept. no need to append
        return False
    return True

def match_span_concept(span, concept):
    span = str(span).lower()
    word_concept = concept[2].lower() # get the ori word, not the concept
    
    if span == word_concept:
        return True
    return False

def inside_bound(idx, bound):
    return idx >= bound[0] and idx < bound[1]

def at_the_end_of_bound(idx, bound):
    return idx == bound[1] - 1

def bert_concept_alignment(tokens, c_concepts, e_concepts, claim_bound, evi_bound):
    n_token = len(tokens)
    concepts = c_concepts
    n_concept = len(concepts)
    
    token_id2concept_id = np.zeros((n_token), dtype=int)
    tok2id = np.zeros((n_token), dtype=int)
    merged_tokens = []
    for i in range(n_token):
        token_id2concept_id[i] = i
        tok2id[i] = i
        
    s_id = 0
    e_id = 0
    concept_id = 0
    current_span = ''
    
    tok_cur_id = 0
    while s_id != n_token:
        
        # if not in the claim/evi boundary
        if (not(inside_bound(s_id, claim_bound)) and
            not(inside_bound(s_id, evi_bound))):
            tok2id[s_id] = tok_cur_id
            tok_cur_id += 1
            
            # add token to merged_tokens if it is either claim, title, or evidence
            if s_id <= evi_bound[1]:
                merged_tokens.append(tokens[s_id])
            
            token_id2concept_id[s_id] = -1
            s_id += 1
            continue
        
        # reset concepts based on the claim/evi boundary
        if (s_id == claim_bound[0]):
            concept_id = 0
            current_span = ''
            concepts = c_concepts
            n_concept = len(concepts)
        elif (s_id == evi_bound[0]):
            concept_id = 0
            current_span = ''
            concepts = e_concepts
            n_concept = len(concepts)
        
        
        # process sub-word level
        next_token_id = s_id + 1
        current_span = tokens[s_id]
        while next_token_id < n_token and str(tokens[next_token_id]).startswith('##'):
            current_span += tokens[next_token_id][2:] # remove ##
            next_token_id += 1
        
        # let's see if combining next token will form a concept
        e_id = next_token_id
        while concept_id < n_concept and do_need_append_token(current_span, concepts[concept_id]) \
        and (at_the_end_of_bound(e_id, claim_bound) or at_the_end_of_bound(e_id, evi_bound)):
            current_span += (' ' + tokens[e_id])
            e_id += 1
        
        # if current span match with current concept
        if concept_id < n_concept and match_span_concept(current_span, concepts[concept_id]):
            token_id2concept_id[s_id:e_id] = concepts[concept_id][4]
            concept_id += 1
            
            tok2id[s_id:e_id] = tok_cur_id
            tok_cur_id += 1
            
            merged_tokens.append(current_span)
        else:
            token_id2concept_id[s_id:e_id] = -1
            
            tok2id[s_id:e_id] = tok_cur_id
            tok_cur_id += 1
            
            merged_tokens.append(current_span)
            
        s_id = e_id

    # merge same token into span
    final_tok2id = list(range(0, tok2id[-1] + 1))
    final_token_id2concept_id = []
    
    last_SEP = len(tokens) - 1 - tokens[::-1].index('[SEP]')
    input_masks = []
    segment_ids = []

    for idx in range(n_token):
        if idx == 0 or tok2id[idx] == -1 or tok2id[idx] != tok2id[idx - 1] and idx <= last_SEP:
            final_token_id2concept_id.append(token_id2concept_id[idx])
            input_masks.append(1)
            if inside_bound(idx, evi_bound):
                segment_ids.append(1)
            else:
                segment_ids.append(0)
    
    assert len(input_masks) == len(final_token_id2concept_id)
    assert len(segment_ids) == len(final_token_id2concept_id)
    assert len(merged_tokens) == len(final_token_id2concept_id)
    
    # create token pooling mask to pool subword level to span level. we use average pooling
    token_pooling_mask = np.zeros((n_token, n_token), dtype=float)
    s_id = 0
    e_id = 0
    cur_id = 0
    while s_id != n_token:
        e_id = s_id + 1
        
        while e_id < n_token and tok2id[s_id] != -1 and tok2id[s_id] == tok2id[e_id]:
            e_id += 1

        n = e_id - s_id
        token_pooling_mask[s_id:e_id, cur_id] = (1/n)
        s_id = e_id
        cur_id += 1
    
    assert len(token_pooling_mask) == n_token

    return merged_tokens, final_token_id2concept_id, input_masks, segment_ids, token_pooling_mask


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


def tok2int_sent(sentence, tokenizer, max_seq_length, graph_constructor):
    """Loads a data file into a list of `InputBatch`s."""
    sent_a, title, sent_b, c_concepts, e_concepts = sentence
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
    
    
    # align bert tokenizer and concepts
    tokens += padding
    c_start_id, c_end_id = 1, 1 + len(tokens_a) #exclusive
    if tokens_b:
        e_start_id, e_end_id = 1 + len(tokens_a) + 1 + len(tokens_t) + 1, 1 + len(tokens_a) + 1 + len(tokens_t) + 1 + len(tokens_b)
    else:
        e_start_id, c_end_id = -1, -1
    merged_tokens, tok2concept, comb_input_mask, comb_segment_ids, tok_pool_mask = bert_concept_alignment(tokens, c_concepts, e_concepts,
                                                                                               (c_start_id, c_end_id), 
                                                                                               (e_start_id, e_end_id))
    
    n_pad = (max_seq_length - len(tok2concept))
    tok2concept += [-1] * n_pad
    comb_input_mask += [0] * n_pad
    comb_segment_ids += [0] * n_pad
    
    assert len(tok2concept) == max_seq_length
    assert len(tok_pool_mask) == max_seq_length
    assert len(comb_input_mask) == max_seq_length
    assert len(comb_segment_ids) == max_seq_length
    
    # create graph based on bert concept alignment
    head_indices, tail_indices, rel_ids = graph_constructor.construct_concept_graph(merged_tokens, tok2concept, 
                                                                                         comb_input_mask)
    
    bert_input = (input_ids, input_mask, segment_ids)
    combine_input = (tok2concept, comb_input_mask, comb_segment_ids)
    graph_input = (head_indices, tail_indices, rel_ids)
    return bert_input, combine_input, graph_input, tok_pool_mask



def tok2int_list(src_list, tokenizer, max_seq_length, graph_constructor, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    
    tok2concept_padding = list()
    comb_msk_padding = list()
    comb_seg_padding = list()
    
    head_indices_padding = list()
    tail_indices_padding = list()
    rel_ids_padding = list()
    
    tok_pool_mask_padding = list()
    for step, sent in enumerate(src_list):
        bert_input, combine_input, graph_input, tok_pool_mask = tok2int_sent(sent, tokenizer, max_seq_length, graph_constructor)
        input_ids, input_mask, input_seg = bert_input
        tok2concept, comb_mask, comb_seg = combine_input
        head_indices, tail_indices, rel_ids = graph_input
        
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)
        
        tok2concept_padding.append(tok2concept)
        comb_msk_padding.append(comb_mask)
        comb_seg_padding.append(comb_seg)
        
        head_indices_padding.append(head_indices)
        tail_indices_padding.append(tail_indices)
        rel_ids_padding.append(rel_ids)
        
        tok_pool_mask_padding.append(tok_pool_mask)
    if max_seq_size != -1:
        inp_padding = inp_padding[:max_seq_size]
        msk_padding = msk_padding[:max_seq_size]
        seg_padding = seg_padding[:max_seq_size]
        
        tok2concept_padding = tok2concept_padding[:max_seq_size]
        comb_msk_padding = comb_msk_padding[:max_seq_size]
        comb_seg_padding = comb_seg_padding[:max_seq_size]
        
        tok_pool_mask_padding = tok_pool_mask_padding[:max_seq_size]
        
        inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
        msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
        seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
        
        tok2concept_padding += ([[0] * max_seq_length] * (max_seq_size - len(tok2concept_padding)))
        comb_msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(comb_msk_padding)))
        comb_seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(comb_seg_padding)))
        
        head_indices_padding += ([[0] * max_seq_length] * (max_seq_size - len(head_indices_padding)))
        tail_indices_padding += ([[0] * max_seq_length] * (max_seq_size - len(tail_indices_padding)))
        rel_ids_padding += ([[-1] * max_seq_length] * (max_seq_size - len(rel_ids_padding)))
        
        tok_pool_mask_padding += ([[[0] * max_seq_length] * max_seq_length] * (max_seq_size - len(tok_pool_mask_padding)))
    
    bert_inp_padding = (inp_padding, msk_padding, seg_padding)
    comb_inp_padding = (tok2concept_padding, comb_msk_padding, comb_seg_padding)
    graph_inp_padding = (head_indices_padding, tail_indices_padding, rel_ids_padding)
    return bert_inp_padding, comb_inp_padding, graph_inp_padding, tok_pool_mask_padding


def graph_inp_add_padding(edge_idx_inputs, rel_idx_inputs):
    max_len = 0
    for item in rel_idx_inputs:
        max_len = max(max_len, len(item))
        
    # append edge_idx
    for edge in edge_idx_inputs:
        for item in edge:
            n_pad = max_len - len(item)
            item += [0] * n_pad
            
    # append for rel_idx_inputs
    for rel in rel_idx_inputs:
        n_pad = max_len - len(rel)
        rel += [-1] * n_pad
    
    return max_len


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
        self.graph_constructor = GraphConstructor(args)

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
                claim = instance['claim']
                evi_list = list()
                for evidence in instance['evidence']:
                    item = [self.process_sent(claim), self.process_wiki_title(evidence[0]),
                                     self.process_sent(evidence[2])]

                    # append claim concepts and evi concepts
                    item.extend([instance['claim_concepts'], evidence[4]]) 
                    evi_list.append(item)
                label = self.label_map[instance['label']]
                evi_list = evi_list[:self.evi_num]
                examples.append([evi_list, label])
        return examples


    def shuffle(self):
        np.random.shuffle(self.examples)

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
            tok2concept_inputs, comb_msk_padding_inputs, comb_seg_padding_inputs = [], [], []
            edge_idx_inputs, rel_idx_inputs = [], []
            tpool_inputs = []
            for step in range(len(inputs)):
                bert_inp, comb_inp, graph_inp, tpool = tok2int_list(inputs[step], self.tokenizer, self.max_len, 
                                                                    self.graph_constructor, self.evi_num)
                inp, msk, seg = bert_inp
                t2c, comb_msk, comb_seg = comb_inp
                head_idx, tail_idx, rel_idx = graph_inp
                
                inp_padding_inputs += inp
                msk_padding_inputs += msk
                seg_padding_inputs += seg
                
                tok2concept_inputs += t2c
                comb_msk_padding_inputs += comb_msk
                comb_seg_padding_inputs += comb_seg
                
                edge_idx_inputs += ([head_idx, tail_idx])
                rel_idx_inputs += rel_idx
                
                tpool_inputs += tpool

            inp_tensor_input = Variable(
                torch.LongTensor(inp_padding_inputs)).view(-1, self.evi_num, self.max_len)
            msk_tensor_input = Variable(
                torch.LongTensor(msk_padding_inputs)).view(-1, self.evi_num, self.max_len)
            seg_tensor_input = Variable(
                torch.LongTensor(seg_padding_inputs)).view(-1, self.evi_num, self.max_len)
            
            tok2concept_tensor_input = Variable(
                torch.LongTensor(tok2concept_inputs)).view(-1, self.evi_num, self.max_len)
            tok2concept_tensor_input[tok2concept_tensor_input==-1] = CONCEPT_DUMMY_IDX
            comb_msk_tensor_input = Variable(
                torch.LongTensor(comb_msk_padding_inputs)).view(-1, self.evi_num, self.max_len)
            comb_seg_tensor_input = Variable(
                torch.LongTensor(comb_seg_padding_inputs)).view(-1, self.evi_num, self.max_len)
            
            max_len = graph_inp_add_padding(edge_idx_inputs, rel_idx_inputs)
            edge_idx_tensor_input = Variable(
                torch.LongTensor(edge_idx_inputs)).view(-1, 2, self.evi_num, max_len)
            edge_idx_tensor_input = edge_idx_tensor_input.transpose(1, 2)
            rel_idx_tensor_input = Variable(
                torch.LongTensor(rel_idx_inputs)).view(-1, self.evi_num, max_len)
            rel_idx_tensor_input[rel_idx_tensor_input==-1] = 0 # avoid -1 label. 
            rel_msk_tensor_input = Variable(
                torch.LongTensor(rel_idx_inputs)).view(-1, self.evi_num, max_len)
            rel_msk_tensor_input[rel_msk_tensor_input!=-1] = 1 # set mask, 1 for valid rel, 0 for not
            rel_msk_tensor_input[rel_msk_tensor_input==-1] = 0
            
            
            tpool_tensor_input = Variable(
                torch.FloatTensor(tpool_inputs)).view(-1, self.evi_num, self.max_len, self.max_len)
            lab_tensor = Variable(
                torch.LongTensor(labels))
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
                
                tok2concept_tensor_input = tok2concept_tensor_input.cuda()
                comb_msk_tensor_input = comb_msk_tensor_input.cuda()
                comb_seg_tensor_input = comb_seg_tensor_input.cuda()
                
                edge_idx_tensor_input = edge_idx_tensor_input.cuda()
                rel_idx_tensor_input = rel_idx_tensor_input.cuda()
                rel_msk_tensor_input = rel_msk_tensor_input.cuda()
                
                tpool_tensor_input = tpool_tensor_input.cuda()
                lab_tensor = lab_tensor.cuda()
            self.step += 1
            
            bert_inp_tensor = inp_tensor_input, msk_tensor_input, seg_tensor_input
            comb_inp_tensor = tok2concept_tensor_input, comb_msk_tensor_input, comb_seg_tensor_input
            graph_inp_tensor = edge_idx_tensor_input, rel_idx_tensor_input, rel_msk_tensor_input
            return (bert_inp_tensor, comb_inp_tensor, graph_inp_tensor, tpool_tensor_input), lab_tensor
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
        
        self.graph_constructor = GraphConstructor(args)
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
                claim = instance['claim']
                evi_list = list()
                for evidence in instance['evidence']:
                    item = [self.process_sent(claim), self.process_wiki_title(evidence[0]),
                                     self.process_sent(evidence[2])]
                    
                    # append claim and evidence concepts
                    item.extend([instance['claim_concepts'], evidence[4]]) 
                    evi_list.append(item)
                id = instance['id']
                evi_list = evi_list[:self.evi_num]
                examples.append([evi_list, id])
        return examples


    def shuffle(self):
        np.random.shuffle(self.examples)

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
            tok2concept_inputs, comb_msk_padding_inputs, comb_seg_padding_inputs = [], [], []
            edge_idx_inputs, rel_idx_inputs = [], []
            tpool_inputs = []
            for step in range(len(inputs)):
                bert_inp, comb_inp, graph_inp, tpool = tok2int_list(inputs[step], self.tokenizer, self.max_len, 
                                                                    self.graph_constructor, self.evi_num)
                inp, msk, seg = bert_inp
                t2c, comb_msk, comb_seg = comb_inp
                head_idx, tail_idx, rel_idx = graph_inp
                
                inp_padding_inputs += inp
                msk_padding_inputs += msk
                seg_padding_inputs += seg
                
                tok2concept_inputs += t2c
                comb_msk_padding_inputs += comb_msk
                comb_seg_padding_inputs += comb_seg

                edge_idx_inputs += ([head_idx, tail_idx])
                rel_idx_inputs += rel_idx
                
                tpool_inputs += tpool

            inp_tensor_input = Variable(
                torch.LongTensor(inp_padding_inputs)).view(-1, self.evi_num, self.max_len)
            msk_tensor_input = Variable(
                torch.LongTensor(msk_padding_inputs)).view(-1, self.evi_num, self.max_len)
            seg_tensor_input = Variable(
                torch.LongTensor(seg_padding_inputs)).view(-1, self.evi_num, self.max_len)
            
            tok2concept_tensor_input = Variable(
                torch.LongTensor(tok2concept_inputs)).view(-1, self.evi_num, self.max_len)
            tok2concept_tensor_input[tok2concept_tensor_input==-1] = CONCEPT_DUMMY_IDX
            comb_msk_tensor_input = Variable(
                torch.LongTensor(comb_msk_padding_inputs)).view(-1, self.evi_num, self.max_len)
            comb_seg_tensor_input = Variable(
                torch.LongTensor(comb_seg_padding_inputs)).view(-1, self.evi_num, self.max_len)

            max_len = graph_inp_add_padding(edge_idx_inputs, rel_idx_inputs)
            edge_idx_tensor_input = Variable(
                torch.LongTensor(edge_idx_inputs)).view(-1, 2, self.evi_num, max_len)
            edge_idx_tensor_input = edge_idx_tensor_input.transpose(1, 2)
            rel_idx_tensor_input = Variable(
                torch.LongTensor(rel_idx_inputs)).view(-1, self.evi_num, max_len)
            rel_idx_tensor_input[rel_idx_tensor_input==-1] = 0 # avoid -1 label. 
            rel_msk_tensor_input = Variable(
                torch.LongTensor(rel_idx_inputs)).view(-1, self.evi_num, max_len)
            rel_msk_tensor_input[rel_msk_tensor_input!=-1] = 1 # set mask, 1 for valid rel, 0 for not
            rel_msk_tensor_input[rel_msk_tensor_input==-1] = 0
            
            tpool_tensor_input = Variable(
                torch.FloatTensor(tpool_inputs)).view(-1, self.evi_num, self.max_len, self.max_len)

            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
                
                tok2concept_tensor_input = tok2concept_tensor_input.cuda()
                comb_msk_tensor_input = comb_msk_tensor_input.cuda()
                comb_seg_tensor_input = comb_seg_tensor_input.cuda()

                edge_idx_tensor_input = edge_idx_tensor_input.cuda()
                rel_idx_tensor_input = rel_idx_tensor_input.cuda()
                rel_msk_tensor_input = rel_msk_tensor_input.cuda()
                
                tpool_tensor_input = tpool_tensor_input.cuda()

            self.step += 1
            
            bert_inp_tensor = inp_tensor_input, msk_tensor_input, seg_tensor_input
            comb_inp_tensor = tok2concept_tensor_input, comb_msk_tensor_input, comb_seg_tensor_input
            graph_inp_tensor = edge_idx_tensor_input, rel_idx_tensor_input, rel_msk_tensor_input
            return (bert_inp_tensor, comb_inp_tensor, graph_inp_tensor, tpool_tensor_input), ids
        else:
            self.step = 0
            raise StopIteration()
