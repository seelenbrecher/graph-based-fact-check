import csv
import numpy as np
import torch

CONCEPT_DUMMY_IDX = 799273

def add_concept_args(parser):
    parser.add_argument('--concept_emb', default='../checkpoint/transe/glove.transe.sgd.ent.npy', type=str)
    parser.add_argument('--concept_num', default=799274, type=int)
    parser.add_argument('--concept_dim', default=100, type=int)
    
    parser.add_argument('--relation_emb', default='../checkpoint/transe/glove.transe.sgd.rel.npy', type=str)
    parser.add_argument('--relation_num', default=17, type=int)
    parser.add_argument('--relation_dim', default=100, type=int)
    
    parser.add_argument('--node_dim', default=768, type=int)
    
    parser.add_argument('--use_concept', action='store_true', default=False)
    
    parser.add_argument('--concept_vocab_path', default='preprocess_concept/data/concept.txt')
    parser.add_argument('--relation_vocab_path', default='preprocess_concept/data/relation.txt')
    parser.add_argument('--c2r_vocab_path', default='preprocess_concept/data/conceptnet.en.csv')
    
    return parser

def add_span_gat_args(parser):
    parser.add_argument('--span_use_gat', default=False, action='store_true')
    parser.add_argument('--span_gat_n_layers', default=3, type=int)
    #last elem act as classification layers on the original imp. I don't use the classification layer, but to make the changes as little as possible, i put this as dummy, and stop the model until last-layer - 1
    parser.add_argument('--span_gat_n_heads', default=[4, 4, 1], type=int, nargs="*") 
    parser.add_argument('--span_gat_n_features', default=[100, 192, 192, 1], type=int, nargs="*")
    parser.add_argument('--span_gat_add_skip_conn', default=False, action='store_true')
    parser.add_argument('--span_gat_bias', default=True)
    parser.add_argument('--span_gat_dropout', default=0.6, type=float)
    parser.add_argument('--span_gat_log_attention_weights', default=False, action='store_true')

    return parser

def load_transe_emb(args):
    c_path = args.concept_emb
    r_path = args.relation_emb
    
    cp_emb = np.load(c_path)
    cp_emb = torch.tensor(cp_emb)
    
    # add 1 more emb for dummy concept
    cp_emb_mean = cp_emb.mean(dim=0)
    cp_emb_mean = cp_emb_mean.view(1, -1)
    cp_emb = torch.cat((cp_emb, cp_emb_mean), dim=0)
    assert cp_emb.shape[0] == args.concept_num
    assert cp_emb.shape[1] == args.concept_dim
    
    rel_emb = np.load(r_path)
    rel_emb = torch.tensor(rel_emb)
    assert rel_emb.shape[0] == args.relation_num
    assert rel_emb.shape[1] == args.relation_dim
    
    return cp_emb, rel_emb


class GraphConstructor():
    OCCURENCE = 'occurence'
    EXACT = 'exact'
    
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

    def load_c2r_vocab(self):
        self.connections = {}
        with open(self.c2r_vocab_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for indice, row in enumerate(csv_reader):
                rel, head, tail, weight = row
                self.connections[head] = tail
                if (head, tail) not in self.connections:
                    self.connections[(head, tail)] = []
                self.connections[(head, tail)].append(rel)
        self.connections
    
    def _inside_bound(self, idx, bound):
        return idx >= bound[0] and idx < bound[1]

    def _make_connection(self, head_info, tail_info, bounds):
        """
        check if head and tail create an edge to the graph
        3 criteria:
            1. has relation w.r.t their concepts
            2. order of occurence in the sentences
            3. concept exact matching (TODO: can employ better matching)
        """
        blacklisted = ['[SEP]', '[CLS]']

        idx_head, head, c_head = head_info
        idx_tail, tail, c_tail = tail_info
        claim_bound, title_bound, evi_bound = bounds

        if head in blacklisted or tail in blacklisted:
            return False, None

        # based on concepts
        if (c_head, c_tail) in self.connections:
            return True, self.connections[(c_head, c_tail)][0]

        # based on occurence
        all_claims = self._inside_bound(idx_head, claim_bound) and self._inside_bound(idx_tail, claim_bound)
        all_titles = self._inside_bound(idx_head, title_bound) and self._inside_bound(idx_tail, title_bound)
        all_evis = self._inside_bound(idx_head, evi_bound) and self._inside_bound(idx_tail, evi_bound)
        if (idx_head == idx_tail - 1) and (all_claims or all_titles or all_evis):
            return True, self.OCCURENCE

        if c_head and c_head == c_tail:
            return True, self.EXACT

        return False, None

    def construct_concept_graph(self, tokens, concepts, input_mask):
        n_token = len(tokens)

        bounds = [] # supposed to create 3 items. claim_bound, title_bound, evi_bound 
        s_id = 1
        e_id = 1
        while(e_id != n_token):
            if tokens[e_id] == '[SEP]':
                bounds.append((s_id, e_id))
                s_id = e_id + 1
            e_id += 1
        
        if len(bounds) != 3:
            import pdb
            pdb.set_trace()
        assert len(bounds) == 3

        head_indices = []
        tail_indices = []
        rel_ids = []
        for idx_head, head in enumerate(tokens):
            for idx_tail, tail in enumerate(tokens):
                if idx_head == idx_tail:
                    continue
                c_head = None if concepts[idx_head] == -1 else self.id2concept[concepts[idx_head]]
                c_tail = None if concepts[idx_tail] == -1 else self.id2concept[concepts[idx_tail]]
                is_connect, conn = self._make_connection((idx_head, head, c_head), (idx_tail, tail, c_tail), bounds)
                if is_connect:
                    head_indices.append(idx_head)
                    tail_indices.append(idx_tail)
                    rel_ids.append(self.rel2id[conn] if conn in self.rel2id else -1)

        return head_indices, tail_indices, rel_ids
    
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