import re
import csv
import numpy as np
import torch
import json
import argparse
import multiprocessing
import math
from tqdm import tqdm
from queue import Queue
from transformers import *
from collections import OrderedDict
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer


# the different between original construct_concept_graph is how we build the edges.
# Now, we take all edges into considerations
# and we do full 2-hop relations
# we retain the intermediate nodes as well

# https://github.com/luxigner/lfu_cache/blob/master/cache.py

class CacheNode(object):
    def __init__(self, key, value, freq_node, pre, nxt):
        self.key = key
        self.value = value
        self.freq_node = freq_node
        self.pre = pre  # previous CacheNode
        self.nxt = nxt  # next CacheNode

    def free_myself(self):
        if self.freq_node.cache_head == self.freq_node.cache_tail:
            self.freq_node.cache_head = self.freq_node.cache_tail = None
        elif self.freq_node.cache_head == self:
            self.nxt.pre = None
            self.freq_node.cache_head = self.nxt
        elif self.freq_node.cache_tail == self:
            self.pre.nxt = None
            self.freq_node.cache_tail = self.pre
        else:
            self.pre.nxt = self.nxt
            self.nxt.pre = self.pre

        self.pre = None
        self.nxt = None
        self.freq_node = None


class FreqNode(object):
    def __init__(self, freq, pre, nxt):
        self.freq = freq
        self.pre = pre  # previous FreqNode
        self.nxt = nxt  # next FreqNode
        self.cache_head = None  # CacheNode head under this FreqNode
        self.cache_tail = None  # CacheNode tail under this FreqNode

    def count_caches(self):
        if self.cache_head is None and self.cache_tail is None:
            return 0
        elif self.cache_head == self.cache_tail:
            return 1
        else:
            return '2+'

    def remove(self):
        if self.pre is not None:
            self.pre.nxt = self.nxt
        if self.nxt is not None:
            self.nxt.pre = self.pre

        pre = self.pre
        nxt = self.nxt
        self.pre = self.nxt = self.cache_head = self.cache_tail = None

        return (pre, nxt)

    def pop_head_cache(self):
        if self.cache_head is None and self.cache_tail is None:
            return None
        elif self.cache_head == self.cache_tail:
            cache_head = self.cache_head
            self.cache_head = self.cache_tail = None
            return cache_head
        else:
            cache_head = self.cache_head
            self.cache_head.nxt.pre = None
            self.cache_head = self.cache_head.nxt
            return cache_head

    def append_cache_to_tail(self, cache_node):
        cache_node.freq_node = self

        if self.cache_head is None and self.cache_tail is None:
            self.cache_head = self.cache_tail = cache_node
        else:
            cache_node.pre = self.cache_tail
            cache_node.nxt = None
            self.cache_tail.nxt = cache_node
            self.cache_tail = cache_node

    def insert_after_me(self, freq_node):
        freq_node.pre = self
        freq_node.nxt = self.nxt

        if self.nxt is not None:
            self.nxt.pre = freq_node

        self.nxt = freq_node

    def insert_before_me(self, freq_node):
        if self.pre is not None:
            self.pre.nxt = freq_node

        freq_node.pre = self.pre
        freq_node.nxt = self
        self.pre = freq_node


class LFUCache(object):

    def __init__(self, capacity):
        self.cache = {}  # {key: cache_node}
        self.capacity = capacity
        self.freq_link_head = None

    def get(self, key):
        if key in self.cache:
            cache_node = self.cache[key]
            freq_node = cache_node.freq_node
            value = cache_node.value

            self.move_forward(cache_node, freq_node)

            return value
        else:
            return -1

    def set(self, key, value):
        if self.capacity <= 0:
            return -1

        if key not in self.cache:
            if len(self.cache) >= self.capacity:
                self.dump_cache()

            self.create_cache(key, value)
        else:
            cache_node = self.cache[key]
            freq_node = cache_node.freq_node
            cache_node.value = value

            self.move_forward(cache_node, freq_node)

    def move_forward(self, cache_node, freq_node):
        if freq_node.nxt is None or freq_node.nxt.freq != freq_node.freq + 1:
            target_freq_node = FreqNode(freq_node.freq + 1, None, None)
            target_empty = True
        else:
            target_freq_node = freq_node.nxt
            target_empty = False

        cache_node.free_myself()
        target_freq_node.append_cache_to_tail(cache_node)

        if target_empty:
            freq_node.insert_after_me(target_freq_node)

        if freq_node.count_caches() == 0:
            if self.freq_link_head == freq_node:
                self.freq_link_head = target_freq_node

            freq_node.remove()

    def dump_cache(self):
        head_freq_node = self.freq_link_head
        self.cache.pop(head_freq_node.cache_head.key)
        head_freq_node.pop_head_cache()

        if head_freq_node.count_caches() == 0:
            self.freq_link_head = head_freq_node.nxt
            head_freq_node.remove()

    def create_cache(self, key, value):
        cache_node = CacheNode(key, value, None, None, None)
        self.cache[key] = cache_node

        if self.freq_link_head is None or self.freq_link_head.freq != 0:
            new_freq_node = FreqNode(0, None, None)
            new_freq_node.append_cache_to_tail(cache_node)

            if self.freq_link_head is not None:
                self.freq_link_head.insert_before_me(new_freq_node)

            self.freq_link_head = new_freq_node
        else:
            self.freq_link_head.append_cache_to_tail(cache_node)

def add_concept_args(parser):
    parser.add_argument('--concept_emb', default='../checkpoint/transe/glove.transe.sgd.ent.npy', type=str)
    parser.add_argument('--concept_num', default=799274, type=int)
    parser.add_argument('--concept_dim', default=768, type=int)
    
    parser.add_argument('--relation_emb', default='../checkpoint/transe/glove.transe.sgd.rel.npy', type=str)
    parser.add_argument('--relation_num', default=17, type=int)
    parser.add_argument('--relation_dim', default=100, type=int)
    
    parser.add_argument('--node_dim', default=768, type=int)
    
    parser.add_argument('--use_concept', action='store_true', default=False)
    
    parser.add_argument('--concept_vocab_path', default='conceptnet5.7/concept.txt')
    parser.add_argument('--relation_vocab_path', default='conceptnet5.7/relation.txt')
    parser.add_argument('--c2r_vocab_path', default='conceptnet5.7/conceptnet-en-5.7.0.csv')
    
    return parser

def add_span_gat_args(parser):
    parser.add_argument('--span_use_gat', default=False, action='store_true')
    parser.add_argument('--span_gat_n_layers', default=3, type=int)
    #last elem act as classification layers on the original imp. I don't use the classification layer, but to make the changes as little as possible, i put this as dummy, and stop the model until last-layer - 1
    parser.add_argument('--span_gat_n_heads', default=[4, 4, 1], type=int, nargs="*") 
    parser.add_argument('--span_gat_n_features', default=[768, 192, 192, 1], type=int, nargs="*")
    parser.add_argument('--span_gat_add_skip_conn', default=False, action='store_true')
    parser.add_argument('--span_gat_bias', default=True)
    parser.add_argument('--span_gat_dropout', default=0.6, type=float)
    parser.add_argument('--span_gat_log_attention_weights', default=False, action='store_true')

    return parser

class GraphConstructor():

    # from conceptnet5.5 paper
    SYMMTERIC_RELATIONS = [
        'Antonym',
        'DistinctFrom',
        'EtymologicallyRelatedTo',
        'LocatedNear',
        'RelatedTo',
        'SimilarTo',
        'Synonym'
    ]

    def __init__(self, args):
        self.concept_vocab_path = args.concept_vocab_path
        self.relation_vocab_path = args.relation_vocab_path
        self.c2r_vocab_path = args.c2r_vocab_path
        
        self.concept2id = None
        self.id2concept = None
        self.rel2id = None
        self.id2rel = None
        self.connections = None
        
        self.load_concept_vocab()
        self.load_relation_vocab()
        self.load_c2r_vocab()
        self.lfu_cache = LFUCache(capacity=2487810)
    
    
    def load_concept_vocab(self):
        vocab = []
        with open(self.concept_vocab_path, "r", encoding="utf8") as f:
            vocab = [l.strip() for l in list(f.readlines())]
        self.concept2id = {}
        self.id2concept = {}
        for indice, cp in enumerate(vocab):
            self.concept2id[cp] = indice
            self.id2concept[indice] = cp

    def load_relation_vocab(self):
        vocab = []
        with open(self.relation_vocab_path, "r", encoding="utf8") as f:
            vocab = [l.strip() for l in list(f.readlines())]
        self.rel2id = {}
        self.id2rel = {}
        for indice, rel in enumerate(vocab):
            self.rel2id[rel] = indice
            self.id2rel[indice] = rel

    def _add_relations(self, head, rel, tail):
        self.relation_freq[self.rel2id[rel]] += 1
        if head not in self.direct_connections:
            self.direct_connections[head] = {}
                
        if tail not in self.direct_connections[head]:
            self.direct_connections[head][tail] = []
        
        if rel not in self.direct_connections[head][tail]:
            self.direct_connections[head][tail].append(rel)

        # add reverse, to get connection from tail to head
        if tail not in self.reverse_direct_connections:
            self.reverse_direct_connections[tail] = {}
        
        if head not in self.reverse_direct_connections[tail]:
            self.reverse_direct_connections[tail][head] = []

        if rel not in self.reverse_direct_connections[tail][head]:
            self.reverse_direct_connections[tail][head].append(rel)


    def load_c2r_vocab(self):
        self.direct_connections = OrderedDict()
        self.reverse_direct_connections = OrderedDict()
        self.relation_freq = [0] * len(self.rel2id)
        with open(self.c2r_vocab_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for indice, row in enumerate(csv_reader):
                rel, head, tail, weight = row
                
                self._add_relations(head, rel, tail)
                
                if rel in self.SYMMTERIC_RELATIONS:
                    self._add_relations(tail, rel, head)
    
    def _relations_score(self, path):
        # based on the frequency relation (rel) appears among all the other relations
        # smaller score of relation means it appear less.
        # in terms of relation alone, we will favor smaller score, because the smaller the score, it can tell us more specific meaning in the relation
        # for example, RelatedTo has the highest frequency, yet, it doesn't give so much information.
        # on the other hand, AtLocation, e.t.c has lower frequeny, and it explain the relation better.
        score = 1
        for rel in path:
            if self.id2rel[rel] == 'FormOf':
                continue
            s = self.relation_freq[rel] / sum(self.relation_freq)
            score *= s
        return score
    
    def _get_path_from_paths(self, paths):
        path_cnt = {}
        for path in paths:
            if str(path) not in path_cnt:
                path_cnt[str(path)] = 0
            path_cnt[str(path)] += 1
        
        maks = -1
        most_probable_path = []
        for path in paths:
            if path_cnt[str(path)] > maks or path_cnt[str(path)] == maks and self._relations_score(path) < self._relations_score(most_probable_path):
                maks = path_cnt[str(path)]
                most_probable_path = path
        
        return most_probable_path
    
    def _extract_concept_relation(self, c_source, c_target):
        if c_source == None or c_target == None:
            return []
        
        if c_source not in self.concept2id or c_target not in self.concept2id:
            return []
        
        if c_source == c_target:
            return [self.rel2id['IsSame']]
        
        paths = self.lfu_cache.get('{}_{}'.format(c_source, c_target))
        if paths != -1 and paths is not None:
            return paths

        paths = []

        word_forms = []
        reachable_concepts_head = []
        reachable_rel_head = {}

        # 1st hop
        if c_source in self.direct_connections:
            for next_concept in list(set(self.direct_connections[c_source])):
                if next_concept not in self.concept2id:
                    continue
                for next_rel in self.direct_connections[c_source][next_concept]:
                    if next_rel not in self.rel2id:
                        continue
                    reachable_concepts_head.append(next_concept)
                    
                    if next_concept not in reachable_rel_head:
                        reachable_rel_head[next_concept] = []
                    if next_rel not in reachable_rel_head:
                        reachable_rel_head[next_concept].append([self.rel2id[next_rel]])

                    if next_rel == 'FormOf' and next_concept not in word_forms:
                        word_forms.append(next_concept)

        # 1st hop from word forms
        for source in word_forms:
            if source in self.direct_connections:
                for next_concept in list(set(self.direct_connections[source])):
                    if next_concept not in self.concept2id:
                        continue
                    for next_rel in self.direct_connections[source][next_concept]:
                        if next_rel not in self.rel2id:
                            continue
                        reachable_concepts_head.append(next_concept)
                    
                        if next_concept not in reachable_rel_head:
                            reachable_rel_head[next_concept] = []
                        if next_rel not in reachable_rel_head:
                            reachable_rel_head[next_concept].append([self.rel2id['FormOf'], self.rel2id[next_rel]])

        # compute 1st hop first
        for next_concept in reachable_concepts_head:
            if next_concept == c_target:
                for rel in reachable_rel_head[next_concept]:
                    if rel != [] and rel not in paths:
                        paths.append(rel)
        
        # can be done in 1-hop, then dont need to do 2 hops
        if len(paths) > 0:
            paths = self._get_path_from_paths(paths)
            self.lfu_cache.set('{}_{}'.format(c_source, c_target), paths)
            return paths
        
        if c_target in self.reverse_direct_connections:
            for next_concept in list(set(self.reverse_direct_connections[c_target])):
                if next_concept not in self.concept2id:
                        continue
                if next_concept in reachable_concepts_head:
                    for next_rel_from_tail in self.reverse_direct_connections[c_target][next_concept]:
                        if next_rel_from_tail not in self.rel2id:
                            continue
                        for next_rel_from_head in reachable_rel_head[next_concept]:
                            path = next_rel_from_head.copy()
                            path.extend([self.rel2id[next_rel_from_tail]])
                            if path != [] and path not in paths:
                                paths.append(path)
        
        paths = self._get_path_from_paths(paths)
        self.lfu_cache.set('{}_{}'.format(c_source, c_target), paths)
        return paths

    def _make_connection(self, head_info, tail_info):
        """
        check if head and tail create an edge to the graph
        """
        blacklisted = ['[SEP]', '[CLS]']

        idx_head, head, c_head = head_info
        idx_tail, tail, c_tail = tail_info

        if head in blacklisted or tail in blacklisted:
            return []

        # based on concepts
        relations = self._extract_concept_relation(c_head, c_tail) 

        return relations

    def construct_concept_graph(self, tokens, concepts, input_mask):
        n_token = len(tokens)

        head_indices = []
        tail_indices = []
        rel_ids = []
        for idx_head, head in enumerate(tokens):
            for idx_tail, tail in enumerate(tokens):
                if idx_head == idx_tail:
                    continue
                c_head = None if concepts[idx_head] == -1 else self.id2concept[concepts[idx_head]]
                c_tail = None if concepts[idx_tail] == -1 else self.id2concept[concepts[idx_tail]]
                relations = self._make_connection((idx_head, head, c_head), (idx_tail, tail, c_tail))
                
#                 if len(relations) > 0:
#                     print('{} -> {}'.format(c_head, c_tail))
#                     for rel in relations:
#                         print(self.id2rel[rel], end=' ')
#                     print(' ')
#                     print('==============================')
                
                if len(relations) > 0:
                    head_indices.append(idx_head)
                    tail_indices.append(idx_tail)
                    rel_ids.append(relations)
#         [print(tokens[h],'->',tokens[t], ':', rels[r]) for tokens, h, t, rels, r in zip([tokens]*len(head_indices), head_indices, tail_indices, [self.id2rel] * len(head_indices), rel_ids)]
        assert len(head_indices) == len(tail_indices)
        assert len(head_indices) == len(rel_ids)
        return head_indices, tail_indices, rel_ids

def do_need_append_token(span, concept, space=True):
    span = str(span).lower()
    word_concept = concept[2].lower() # get the ori word, not the concept
    
    if span == word_concept: # exact match, no need to append more token
        return False
    if space and not word_concept.startswith(span+' '): # current span is completly different with the concept. no need to append
        return False
    if not space and not word_concept.startswith(span):
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
    return idx == bound[1]


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
        and not (at_the_end_of_bound(e_id, claim_bound) or at_the_end_of_bound(e_id, evi_bound)):
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

def bert_concept_alignment_roberta(tokens, c_concepts, e_concepts, claim_bound, evi_bound):
    tokens = [tok[1:] if isinstance(tok, str) and tok[0] == 'Ġ' else tok for tok in tokens]
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
        while concept_id < n_concept \
        and not (at_the_end_of_bound(e_id, claim_bound) or at_the_end_of_bound(e_id, evi_bound)):
            need_append = do_need_append_token(current_span, concepts[concept_id], space=True)
            if need_append:
                current_span += (' ' + tokens[e_id])
                e_id += 1
                continue
            need_append = do_need_append_token(current_span, concepts[concept_id], space=False)
            if need_append:
                current_span += tokens[e_id]
                e_id += 1
                continue
            break
        
        # if current span match with current concept
        if concept_id < n_concept and match_span_concept(current_span, concepts[concept_id]):
            token_id2concept_id[s_id:e_id] = concepts[concept_id][4]
            concept_id += 1
            
            tok2id[s_id:e_id] = tok_cur_id
            tok_cur_id += 1
            
            merged_tokens.append(current_span)
            s_id = e_id
        else:
            token_id2concept_id[s_id:s_id+1] = -1
            
            tok2id[s_id:s_id+1] = tok_cur_id
            tok_cur_id += 1
            
            merged_tokens.append(tokens[s_id])
            
            s_id += 1

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

def tok2int_sent(sentence, tokenizer, max_seq_length, graph_constructor, roberta=False):
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
    
    if roberta:
        merged_tokens, tok2concept, comb_input_mask, comb_segment_ids, tok_pool_mask = bert_concept_alignment_roberta(tokens, c_concepts, e_concepts,
                                                                                               (c_start_id, c_end_id), 
                                                                                               (e_start_id, e_end_id))
    else:
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
    
    return head_indices, tail_indices, rel_ids


def process_sent(sentence):
    sentence = re.sub(" LSB.*?RSB", "", sentence)
    sentence = re.sub("LRB RRB ", "", sentence)
    sentence = re.sub("LRB", " ( ", sentence)
    sentence = re.sub("RRB", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)

    return sentence

def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub("LRB", " ( ", title)
    title = re.sub("RRB", " )", title)
    title = re.sub("COLON", ":", title)
    return title

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

def worker(args, batch, n_batch):
    input_data = []
    print('load data', batch)
    with open(args.input, 'r') as f_in:
        for step, line in enumerate(f_in):
            data = json.loads(line)
            input_data.append(data)
    print('end load data', batch)

    n_data = len(input_data)
    item_cnt = math.ceil(n_data/n_batch)
    s_id = batch * item_cnt
    e_id = min((batch + 1) * item_cnt, len(input_data))
    
    print('process batch = {}/{}, start_id = {}, end_id = {}'.format(batch, n_batch, s_id, e_id))
    
    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    
    if args.roberta:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    
    graph_constructor = GraphConstructor(args)
    with open('{}_{}.json'.format(args.output, batch), 'w') as f_out:
        for instance in tqdm(input_data[s_id:e_id]):
            claim = instance['claim']
            evi_list = list()
            for evidence in instance['evidence']:
                item = [process_sent(claim), process_wiki_title(evidence[0]),
                                 process_sent(evidence[2])]

                # append claim concepts and evi concepts
                item.extend([instance['claim_concepts'], evidence[4]]) 
                evi_list.append(item)
            #label = label_map[instance['label']]
            evi_list = evi_list[:args.evi_num]

            evi_head_indices = []
            evi_tail_indices = []
            evi_rel_ids = []
            for evi in evi_list:
                head_indices, tail_indices, rel_ids = tok2int_sent(evi, tokenizer, args.max_len, graph_constructor, roberta=args.roberta)
                evi_head_indices.append(head_indices)
                evi_tail_indices.append(tail_indices)
                evi_rel_ids.append(rel_ids)

            instance['evi_head_indices'] = evi_head_indices
            instance['evi_tail_indices'] = evi_tail_indices
            instance['evi_rel_ids'] = evi_rel_ids

            f_out.write('{}\n'.format(json.dumps(instance)))

def main():
    parser = argparse.ArgumentParser()
    parser = add_concept_args(parser)
    parser = add_span_gat_args(parser)
    parser.add_argument('--max_len', default=130, type=int)
    parser.add_argument('--evi_num', default=500, type=int)
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--bert_pretrain', default='../../bert_base')
    parser.add_argument('--roberta', action='store_true', default=False)
    
    args = parser.parse_args()
#     worker(args, 0,8)
    for batch in range(8):
        p = multiprocessing.Process(target=worker, args=(args, batch, 8,))
        p.start()

if __name__=='__main__':
    main()
