import requests

import argparse
import configparser
import json
import math
import multiprocessing
from tqdm import tqdm
import re
import time
import numpy as np
import difflib

from pytorch_pretrained_bert.tokenization import BertTokenizer
from prepare_concept import bert_concept_alignment

tokenizer = BertTokenizer.from_pretrained('../../bert_base', do_lower_case=False)
MAX_SEQ_LENGTH = 130

import numpy as np

TAGS = ['ARG', 'V', 'TMP', 'LOC', 'ADV', 'MNR', 'PRD', 'DIR', 'CAU', 'PNC', 'DIS', # from SRL
        'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
        'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
        'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']


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

            
def add_ner_srl_args(parser):
    parser.add_argument('--ner_srl_train_path', default='../data/srl_and_ner/train.json', type=str)
    parser.add_argument('--ner_srl_eval_path', default='../data/srl_and_ner/eval.json', type=str)
    parser.add_argument('--ner_srl_test_path', default='../data/srl_and_ner/test.json', type=str)
    
    return parser

def get_bounds(tokens):
    """
    every tokens should have 3 bounds
    [CLS] CLAIM [SEP] TITLE [SEP] EVI [SEP]
    """
    start = 0
    bounds = []
    for id, token in enumerate(tokens):
        if token == '[CLS]':
            start = id + 1
        if token == '[SEP]':
            bounds.append((start, id)) # exclusive
            start = id + 1
    return bounds
        

def convert_to_word_level(tokens):
    n_token = len(tokens)
    token_ids = [0] * n_token
    words = []
    
    cur_word = ''
    for i, token in enumerate(tokens):
        if token.startswith('##'):
            cur_word += token[2:]
            token_ids[i] = len(words)
            continue
        elif cur_word != '':
            words.append(cur_word)
        cur_word = token
        token_ids[i] = len(words)
    words.append(cur_word)
    return words, token_ids


def get_ner_token_and_tags(ner):
    n_words = len(ner['words'])
    ner_tokens = []
    ner_tags = []
    for i in range(n_words):
        tag = ner['tags'][i]
        if tag.startswith('B-'):
            tag = tag[2:]
            for j in range(i + 1, n_words):
                end_tag = ner['tags'][j]
                if end_tag.startswith('L-'):
                    ner_tokens.append((i, j))
                    ner_tags.append(tag)
                    break
            i = j + 1
    
    return ner_tokens, ner_tags


def clean_tag(tag):
    WHITELIST = ['ARG', 'V', 'TMP', 'LOC', 'ADV', 'MNR', 'PRD', 'DIR', 'CAU', 'PNC', 'DIS']
    tag = tag[2:]
    if tag in WHITELIST: # filter V
        return tag
    if tag.startswith('ARGM-') and tag[5:] in WHITELIST: # filter TMP, LOC, ADV, MNR
        return tag[5:]
    if tag.startswith('ARG') and tag[3:].isnumeric():
        return 'ARG'
    return None


def get_srl_token_and_tags(srl):
    """
    get span which potentially become entity from SRL
    extract relation between srl_token as well
    """
    items = []
    item_id = 0
    rel = dict()
    for verb in srl['verbs']:
        start_id = 0
        has_add = False
        for id, tag in enumerate(verb['tags']):
            if tag == 'O':
                start_id = id + 1
                continue
            if (id == len(verb['tags']) - 1 or not verb['tags'][id + 1].startswith('I-')):
                tag = clean_tag(verb['tags'][id])
                if tag is not None:
                    items.append((start_id, id, tag, item_id))
                    if has_add:
                        rel[item_id-1] = item_id
                    has_add = True
                    item_id += 1
                start_id = id + 1

    items.sort(key=lambda x: (x[1], x[0]))
    srl_tokens = []
    srl_tags = []
    cur_end = -1
    
    parent = list(range(len(items))) # since we may merged some ids, this var to track which id merge to which id
    for item in items:
        if item[0] <= cur_end:
            for srl in srl_tokens:
                if srl[0] >= item[0] or srl[1] >= item[0]:
                    parent[item[3]] = srl[2]
                    break
            continue
        srl_tokens.append((item[0], item[1], item[3]))
        srl_tags.append(item[2])
        cur_end = item[1]

    seen_edges = {}
    edges = []
    for key, val in rel.items():
        head = parent[key]
        tail = parent[val]
        if (head, tail) not in seen_edges:
            edges.append((head, tail))
            seen_edges[(head, tail)] = True
    return srl_tokens, srl_tags, edges  


def combine_srl_and_ner(ner_token_ids, ner_tags, srl_token_ids, srl_tags, words):
    items = [] # (start_id, end_id, tag, prior)
    for token_id, tag in zip(ner_token_ids, ner_tags):
        items.append((token_id[0], token_id[1], tag, 0))
    
    for token_id, tag in zip(srl_token_ids, srl_tags):
        items.append((token_id[0], token_id[1], tag, 1))
    
    items.sort(key=lambda x: (x[0], x[1], x[3]))
    
    tokens, tags = [], []
    cur_end = -1
    for item in items:
        if item[0] <= cur_end:
            continue
        tokens.append(' '.join(words[item[0]:item[1] + 1]))
        tags.append(item[2])
        cur_end = item[1]
    return tokens, tags


def match_append(span, new_token, target):
    new_span_0 = (span + ' ' + new_token).lstrip()
    new_span_1 = (span + new_token).lstrip()
    if target.startswith(new_span_0):
        return True, new_span_0
    if target.startswith(new_span_1):
        return True, new_span_1
    
    is_same = True
    new_span = ''
    for i, s in enumerate(difflib.ndiff(new_span_0, target)):
        s_split = s.split(' ')
        op, char = s_split[0], s_split[-1]
        char = ' ' if char == '' else char
        if (op == '-' or op == '+') and char != ' ':
            is_same = False
        if op == '-':
            continue
        else:
            new_span += char
    if is_same and new_span == target:
        return True, new_span
    
    is_same = True
    new_span = ''
    for i, s in enumerate(difflib.ndiff(new_span_1, target)):
        s_split = s.split(' ')
        op, char = s_split[0], s_split[-1]
        char = ' ' if char == '' else char
        if (op == '-' or op == '+') and char != ' ':
            is_same = False
        if op == '-':
            continue
        else:
            new_span += char
    if is_same and new_span == target:
        return True, new_span
    
     # in case the source was tokenized into 2 tokens, but in SRL/NER, it was not
    len_new_tok = len(new_token)
    if span.endswith(new_token) and (-len_new_tok-1 < 0 or span[-len_new_tok-1] != ''):
        temp = target.lstrip(span).strip()
        if not temp.startswith(new_token):
            return True, span
    
    return False, None


def align_two_tokens(source_tokens, source_token_ids, query_tokens, bounds):
    
    def update_flag(arr, start, end, val):
        for i in range(start, end):
            if arr[i] != -1:
                continue
            arr[i] = val
    
    n_token = len(source_tokens)
    
    flag = [-1] * n_token
    current_span = ''
    start_id = bounds[0]
    query_idx = 0 # to match query_tokens[query_idx]
    while query_idx < len(query_tokens):
        for token_idx in range(start_id, bounds[1]):
            if query_idx == len(query_tokens):
                    break
            tokens = source_tokens[token_idx].split(' ')
            for sub_tok_id, sub_tok in enumerate(tokens):
                if sub_tok == '[UNK]' and query_tokens[query_idx].startswith(current_span):
                    splitted_tokens = query_tokens[query_idx][len(current_span):].split(' ')
                    unk_word = splitted_tokens[0]
                    if unk_word == '':
                        unk_word = splitted_tokens[1]
                    sub_tok = unk_word
                if query_idx == len(query_tokens):
                    break
                    
                # very special case, where in the source, 2 tokens were merged into 1, while in the NER/SRL, it was 2 tokens
                if (query_idx < len(query_tokens) - 1) and sub_tok == query_tokens[query_idx] + query_tokens[query_idx + 1]:
                    update_flag(flag, start_id, token_idx+1, query_idx)
                    query_idx += 2
                    current_span = ''
                    start_id = token_idx
                    start_id += 1 if sub_tok_id == len(tokens) - 1 else 0
                    continue
                # another case, like sub_tok = Don, query = Do, or sub_tok = Wed, query = We
                if (query_idx < len(query_tokens) - 1) and sub_tok.startswith(query_tokens[query_idx]) and len(sub_tok) - len(query_tokens[query_idx]) == 1:
                    update_flag(flag, start_id, token_idx+1, query_idx)
                    query_idx += 1
                    current_span = ''
                    start_id = token_idx
                    start_id += 1 if sub_tok_id == len(tokens) - 1 else 0
                    continue
                matches, next_span = match_append(current_span, sub_tok, query_tokens[query_idx])
                if not matches:
                    # previous span is not correct. reset and try again
                    current_span = ''
                    start_id = token_idx
#                     start_id += 1 if sub_tok_id == len(tokens) - 1 else 0
                matches, next_span = match_append(current_span, sub_tok, query_tokens[query_idx])
                if matches:
                    current_span = next_span
                    if current_span.lstrip() == query_tokens[query_idx]:
                        update_flag(flag, start_id, token_idx+1, query_idx)

                        # update state
                        query_idx += 1
                        current_span = ''
                        start_id = token_idx
                        start_id += 1 if sub_tok_id == len(tokens) - 1 else 0        
        break
    
    aligned_masks = [-1] * len(source_token_ids)
    for id, tok_id in enumerate(source_token_ids):
        try:
            aligned_masks[id] = flag[tok_id]
        except:
            pass

    return aligned_masks


def align_two_tokens_1(source_tokens, source_token_ids, query_tokens, bounds, tokenizer):
    
    def update_flag(arr, start, end, val):
        for i in range(start, end):
            if arr[i] != -1:
                continue
            arr[i] = val
    
    def find_match(query_idx, current_span, start_id, flag, bounds):
        cur_start_id = start_id
        current_target = ' '.join(convert_to_word_level(tokenizer.tokenize(query_tokens[query_idx]))[0])
#         print(current_target)
        next_target = None
        if query_idx < len(query_tokens) - 1:
            next_target = ' '.join(convert_to_word_level(tokenizer.tokenize(query_tokens[query_idx + 1]))[0])

        for token_idx in range(start_id, bounds[1]):
            tokens = source_tokens[token_idx].split(' ')
            for sub_tok_id, sub_tok in enumerate(tokens):
                # very special case, where in the source, 2 tokens were merged into 1, while in the NER/SRL, it was 2 tokens
                if (next_target is not None) and sub_tok == current_target + next_target:
                    update_flag(flag, start_id, token_idx+1, query_idx)
                    query_idx += 2
                    current_span = ''
                    start_id = token_idx
                    start_id += 1 if sub_tok_id == len(tokens) - 1 else 0
                    return query_idx, current_span, start_id

                # another case, like sub_tok = Don, query = Do, or sub_tok = Wed, query = We
                if (next_target is not None) and sub_tok.startswith(current_target) and len(sub_tok) - len(current_target) == 1:
                    update_flag(flag, start_id, token_idx+1, query_idx)
                    query_idx += 1
                    current_span = ''
                    start_id = token_idx
                    start_id += 1 if sub_tok_id == len(tokens) - 1 else 0
                    return query_idx, current_span, start_id
                
                matches, next_span = match_append(current_span, sub_tok, current_target)
                if not matches:
                    # previous span is not correct. reset and try again
                    current_span = ''
                    start_id = token_idx
#                     start_id += 1 if sub_tok_id == len(tokens) - 1 else 0

                matches, next_span = match_append(current_span, sub_tok, current_target)
                if matches:
                    current_span = next_span
                    if current_span.lstrip() == current_target:
                        update_flag(flag, start_id, token_idx+1, query_idx)

                        # update state
                        query_idx += 1
                        current_span = ''
                        start_id = token_idx
                        start_id += 1 if sub_tok_id == len(tokens) - 1 else 0        
                        return query_idx, current_span, start_id

        if current_target.startswith(current_span):
            update_flag(flag, start_id, token_idx+1, query_idx)
            
            query_idx += 1
            current_span = ''
            start_id = token_idx
            start_id += 1 if sub_tok_id == len(tokens) - 1 else 0        
            return query_idx, current_span, start_id

        start_id = cur_start_id
        # can't find one. so the token would be different from original tokenizer
        # try find with the same prefix or suffix, and it contains > 50% of the character
        for token_idx in range(start_id, bounds[1]):
            tokens = source_tokens[token_idx].split(' ')
            for sub_tok_id, sub_tok in enumerate(tokens):
                if (sub_tok.startswith(current_target) or sub_tok.endswith(current_target)) and \
                    2 * len(current_target) >= len(sub_tok):
                    update_flag(flag, start_id, token_idx+1, query_idx)
                    query_idx += 1
                    current_span = ''
                    start_id = token_idx
                    start_id += 1 if sub_tok_id == len(tokens) - 1 else 0        
                    return query_idx, current_span, start_id

    n_token = len(source_tokens)
    
    flag = [-1] * n_token
    current_span = ''
    start_id = bounds[0]
    query_idx = 0 # to match query_tokens[query_idx]

    while query_idx < len(query_tokens) and start_id < bounds[1]:
        query_idx, current_span, start_id = find_match(query_idx, current_span, start_id, flag, bounds)
    
    aligned_masks = [-1] * len(source_token_ids)
    for id, tok_id in enumerate(source_token_ids):
        try:
            aligned_masks[id] = flag[tok_id]
        except:
            pass

    return aligned_masks
                    

def convert_tags_to_ids(tags):
    # https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf
    tag_ids = []
    for tag in tags:
        assert tag in TAGS
        tag_ids.append(TAGS.index(tag))
    return tag_ids

def refine_srl(srl_token_ids, srl_tags, words, edges):
    items = [] # (start_id, end_id, tag, token_id) 
    for token_id, tag in zip(srl_token_ids, srl_tags):
        items.append((token_id[0], token_id[1], tag, token_id[2]))
    items.sort(key=lambda x: (x[0], x[1], x[3]))

    parent = [-1] * 130
    id = 0
    for item in items:
        if parent[item[3]] == -1:
            parent[item[3]] = id
            id += 1

    tokens, tags = [], []
    cur = (-1, -1)
    for item in items:
        if item[0] >= cur[0] and item[0] < cur[1]:
            continue
        tokens.append(' '.join(words[item[0]:item[1] + 1]))
        tags.append(item[2])
        cur_end = (item[0], item[1])

    seen_edges = {}
    new_edges = []
    for edge in edges:
        head = parent[edge[0]]
        tail = parent[edge[1]]
        if (head, tail) not in seen_edges:
            new_edges.append((head, tail))
            seen_edges[(head, tail)] = True
    return tokens, tags, new_edges


def generate_all_paths(tokens, edges):
    n_token = len(tokens)
    in_dir = [0] * (n_token)
    for edge in edges:
        in_dir[edge[1]] += 1
    
    queue = []
    for i in range(n_token):
        if in_dir[i] == 0:
            queue.append((i, [i]))
    
    paths = []
    while(len(queue) > 0):
        cur_id, cur_path = queue.pop()
        
        can_append = False
        for edge in edges:
            if edge[0] == cur_id:
                queue.append((edge[1], cur_path + [edge[1]]))
                can_append = True
        if not can_append and len(cur_path) > 1:
            paths.append(cur_path)

    return paths


def get_paths_from_input_ids(paths, masks):
    res = []
    msks = []
    for path in paths:
        ids = []
        msk = []
        for p in path:
            if p == -1:
                continue
            for index, value in enumerate(masks):
                if value == p:
                    ids.append(index)
                    msk.append(p)
        res.append(ids)
        msks.append(msk)
    return res, msks


def extract_ner(tokens, ner, srl, tokenizer):
    assert len(ner['words']) == len(srl['words'])
    
    ner_token_ids, ner_tags = get_ner_token_and_tags(ner)
    srl_token_ids, srl_tags, _ = get_srl_token_and_tags(srl)
    sn_tokens, tags = combine_srl_and_ner(ner_token_ids, ner_tags, srl_token_ids, srl_tags, srl['words'])
    
    words, token_ids = convert_to_word_level(tokens)
    aligned_masks = align_two_tokens_1(words, token_ids, sn_tokens, get_bounds(words)[0], tokenizer)
    
    tag_ids = convert_tags_to_ids(tags)
    ner_masks = []
    for id in range(len(tags)):
        temp = (np.array(aligned_masks) == id).astype(int)
        temp = np.nonzero(temp)[0].tolist()
        if temp != []:
            ner_masks.append(temp)
        elif len(ner_masks) > 0:
            ner_masks.append([ner_masks[-1][-1]])
        else:
            ner_masks.append([])
    return ner_masks, tag_ids


def construct_srl_graph(tokens, claim_srl, evi_srl, tokenizer):
    words, token_ids = convert_to_word_level(tokens)
    bounds = get_bounds(words)
    
    claim_token_ids, claim_tags, claim_edges = get_srl_token_and_tags(claim_srl)
    claim_tokens, claim_tags, claim_edges = refine_srl(claim_token_ids, claim_tags, claim_srl['words'], claim_edges)
    claim_aligned_masks = align_two_tokens_1(words, token_ids, claim_tokens, bounds[0], tokenizer)
    claim_paths = generate_all_paths(claim_tokens, claim_edges)
    claim_paths, claim_path_masks = get_paths_from_input_ids(claim_paths, claim_aligned_masks)
    
    claim_tags = convert_tags_to_ids(claim_tags)
    claim_masks = []
    for id in range(len(claim_tags)):
        temp = (np.array(claim_aligned_masks) == id).astype(int)
        temp = np.nonzero(temp)[0].tolist()
        if temp != []:
            claim_masks.append(temp)
        elif len(claim_masks) > 0:
            claim_masks.append([claim_masks[-1][-1]])
        else:
            claim_masks.append([])
    claim_input = (claim_tags, claim_masks, claim_paths, claim_path_masks)
    
    evi_token_ids, evi_tags, evi_edges = get_srl_token_and_tags(evi_srl)
    evi_tokens, evi_tags, evi_edges = refine_srl(evi_token_ids, evi_tags, evi_srl['words'], evi_edges)
    evi_aligned_masks = align_two_tokens_1(words, token_ids, evi_tokens, bounds[2], tokenizer)
    evi_paths = generate_all_paths(evi_tokens, evi_edges)
    evi_paths, evi_path_masks = get_paths_from_input_ids(evi_paths, evi_aligned_masks)
    
    evi_tags = convert_tags_to_ids(evi_tags)
    evi_masks = []
    for id in range(len(evi_tags)):
        temp = (np.array(evi_aligned_masks) == id).astype(int)
        temp = np.nonzero(temp)[0].tolist()
        if temp != []:
            evi_masks.append(temp)
        elif len(evi_masks) > 0:
            evi_masks.append([evi_masks[-1][-1]])
        else:
            evi_masks.append([])
    assert len(evi_masks) == len(evi_tags)
    evi_input = (evi_tags, evi_masks, evi_paths, evi_path_masks)

    return claim_input, evi_input

def get_merged_tokens(sentence, tokenizer, max_seq_length):
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
        
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    
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
    
    return merged_tokens


def read_data(path):
    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    examples = list()
    with open(path, 'r') as f:
        for step, line in enumerate(f):
            instance = json.loads(line.strip())
            claim = instance['claim']
            evi_list = list()
            for evidence in instance['evidence']:
                item = [process_sent(claim), process_wiki_title(evidence[0]),
                                 process_sent(evidence[2])]

                # append claim concepts and evi concepts
                item.extend([instance['claim_concepts'], evidence[4]]) 
                evi_list.append(item)
            if 'label' in instance:
                label = label_map[instance['label']]
            else:
                label = -1
            evi_labels = instance['evidence_labels'] if 'evidence_labels' in instance else [0.0] * len(instance['evidence'])
            if len(evi_labels) != 5:
                evi_labels += [-1.0] * (5 - len(evi_labels))
            evi_list = evi_list[:5]
            id = instance['id']
            examples.append([evi_list, (label, evi_labels), id])
    return examples


def process(examples, srls_and_ners, ids, tokenizer, target_path, MAX_SEQ_LENGTH=130):
    cur_data = {}
#     with open(target_path, 'r') as f:
#         for l in f:
#             l = json.loads(l)
#             cur_data[l['id']] = l
#     print('cur data len', len(cur_data))
    
    with open(target_path, 'w') as f_out:
         for id, example, srl_and_ner in tqdm(zip(ids, examples, srls_and_ners)):
            if id in cur_data:
                f_out.write('{}\n'.format(json.dumps(cur_data[id])))
            else:
                claim_srl = srl_and_ner['claim_srl']
                claim_ner = srl_and_ner['claim_ner']
                evis_srl = srl_and_ner['evis_srl']
                
                ner_masks = []
                ner_tag_ids = []

                claim_masks = []
                claim_tags = []
                claim_paths = []
                claim_path_masks = []

                evi_masks = []
                evi_tags = []
                evi_paths = []
                evi_path_masks = []

                if len(example) > 0:
                    merged_tokens = get_merged_tokens(example[0], tokenizer, MAX_SEQ_LENGTH)
                    ner_masks, ner_tag_ids = extract_ner(merged_tokens, claim_ner, claim_srl, tokenizer)

                    for ex, evi_srl in zip(example, evis_srl):
                        merged_tokens = get_merged_tokens(ex, tokenizer, MAX_SEQ_LENGTH)
                        claim_graph, evi_graph = construct_srl_graph(merged_tokens, claim_srl, evi_srl, tokenizer)

                        claim_tags.append(claim_graph[0])
                        claim_masks.append(claim_graph[1])
                        claim_paths.append(claim_graph[2])
                        claim_path_masks.append(claim_graph[3])

                        evi_tags.append(evi_graph[0])
                        evi_masks.append(evi_graph[1])
                        evi_paths.append(evi_graph[2])
                        evi_path_masks.append(evi_graph[3])

                res = {
                    'id': id,
                    'ner_masks': ner_masks,
                    'ner_tag_ids': ner_tag_ids,
                    'claim_masks': claim_masks,
                    'claim_tags': claim_tags,
                    'claim_paths': claim_paths,
                    'claim_path_masks': claim_path_masks,
                    'evi_masks': evi_masks,
                    'evi_tags': evi_tags,
                    'evi_paths': evi_paths,
                    'evi_path_masks': evi_path_masks,
                }

                f_out.write('{}\n'.format(json.dumps(res)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str)
    parser.add_argument('--source', dest='source', type=str)
    parser.add_argument('--output', dest='output', type=str)
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained('../../bert_base', do_lower_case=False)
    examples = read_data(args.input)
    inputs, labels, ids = list(zip(* examples))
    
    srls_and_ners = []
    with open(args.source, 'r') as f:
        for step, x in enumerate(f):
            x = json.loads(x)
            srls_and_ners.append(x)

    process(inputs, srls_and_ners, ids, tokenizer, args.output)

if __name__=='__main__':
    main()

# python populate_srl_and_ner_data_1.py --input ../fever_with_concepts/bert_train_concept.json --source ../srl_and_ner/train.json --output ../fever_srl_and_ner_1/train.json
# python populate_srl_and_ner_data.py --input ../fever_with_concepts/bert_eval_concept.json --source ../srl_and_ner/eval.json --output ../fever_srl_and_ner/eval.json
