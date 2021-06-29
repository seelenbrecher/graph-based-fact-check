import numpy as np

TAGS = ['ARG', 'V', 'TMP', 'LOC', 'ADV', 'MNR', 'PRD', 'DIR', 'CAU', 'PNC', 'DIS', # from SRL
        'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
        'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
        'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']


def add_ner_srl_args(parser):
    parser.add_argument('--ner_srl_train_path', default='../data/fever_srl_and_ner/train.json', type=str)
    parser.add_argument('--ner_srl_eval_path', default='../data/fever_srl_and_ner/eval.json', type=str)
    parser.add_argument('--ner_srl_test_path', default='../data/fever_srl_and_ner/test.json', type=str)
    parser.add_argument('--srl_n_node_types', default=29, type=int)
    
    # lstm
    parser.add_argument('--srl_lstm_hidden_dim', default=128, type=int)
    parser.add_argument('--srl_lstm_num_layers', default=4, type=int)
    parser.add_argument('--srl_lstm_droput', default=0.1, type=int)
    parser.add_argument('--srl_lstm_bidirectional', default=True)
    
    return parser

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
                if srl[0] >= item[0]:
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
    if new_span_0 in target:
        return True, new_span_0
    if new_span_1 in target:
        return True, new_span_1
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
                if query_idx == len(query_tokens):
                    break
                matches, next_span = match_append(current_span, sub_tok, query_tokens[query_idx])
                if not matches:
                    # previous span is not correct. reset and try again
                    current_span = ''
                    start_id = token_idx
                    start_id += 1 if sub_tok_id == len(tokens) - 1 else 0
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


def extract_ner(tokens, ner, srl):
    assert len(ner['words']) == len(srl['words'])
    
    ner_token_ids, ner_tags = get_ner_token_and_tags(ner)
    srl_token_ids, srl_tags, _ = get_srl_token_and_tags(srl)
    sn_tokens, tags = combine_srl_and_ner(ner_token_ids, ner_tags, srl_token_ids, srl_tags, srl['words'])
    
    words, token_ids = convert_to_word_level(tokens)
    aligned_masks = align_two_tokens(words, token_ids, sn_tokens, get_bounds(words)[0])
    
    tag_ids = convert_tags_to_ids(tags)
    ner_masks = []
    for id in range(len(tags)):
        temp = (np.array(aligned_masks) == id).astype(int)
        ner_masks.append(temp.tolist())
    
    return ner_masks, tag_ids


def construct_srl_graph(tokens, claim_srl, evi_srl):
    words, token_ids = convert_to_word_level(tokens)
    bounds = get_bounds(words)
    
    claim_token_ids, claim_tags, claim_edges = get_srl_token_and_tags(claim_srl)
    claim_tokens, claim_tags, claim_edges = refine_srl(claim_token_ids, claim_tags, claim_srl['words'], claim_edges)
    claim_aligned_masks = align_two_tokens(words, token_ids, claim_tokens, bounds[0])
    claim_paths = generate_all_paths(claim_tokens, claim_edges)
    
    claim_tags = convert_tags_to_ids(claim_tags)
    claim_masks = []
    for id in range(len(claim_tags)):
        temp = (np.array(claim_aligned_masks) == id).astype(int)
        claim_masks.append(temp.tolist())
    claim_input = (claim_tags, claim_masks, claim_paths)

    evi_token_ids, evi_tags, evi_edges = get_srl_token_and_tags(evi_srl)
    evi_tokens, evi_tags, evi_edges = refine_srl(evi_token_ids, evi_tags, evi_srl['words'], evi_edges)
    evi_aligned_masks = align_two_tokens(words, token_ids, evi_tokens, bounds[2])
    evi_paths = generate_all_paths(evi_tokens, evi_edges)
    
    evi_tags = convert_tags_to_ids(evi_tags)
    evi_masks = []
    for id in range(len(evi_tags)):
        temp = (np.array(evi_aligned_masks) == id).astype(int)
        evi_masks.append(temp.tolist())
    evi_input = (evi_tags, evi_masks, evi_paths)
    
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


def pad_ner_masks(ner_masks):
    """
    (batch_size, n_ner, n_node)
    """
    max_n_ner = 1
    max_n_node = 1
    for ner_mask in ner_masks:
        max_n_ner = max(max_n_ner, len(ner_mask))
        for mask_per_node in ner_mask:
            max_n_node = max(max_n_node, len(mask_per_node))
        
    for i, ner_mask in enumerate(ner_masks):
        for j, nm in enumerate(ner_mask):
            ner_mask[j] = nm[:max_n_node]
            ner_mask[j] += [0] * (max_n_node - len(nm))
        
        if len(ner_mask) < max_n_ner:
            ner_masks[i] += np.zeros((max_n_ner - len(ner_mask), max_n_node), dtype=int).tolist() 

            
def pad_ner_tagids(ner_tagids):
    """
    (batch_size, n_ner)
    """
    max_n_ner = 0
    for ner_tag in ner_tagids:
        max_n_ner = max(max_n_ner, len(ner_tag))
        
    for i, ner_tag in enumerate(ner_tagids):
        if len(ner_tag) < max_n_ner:
            ner_tagids[i] += np.zeros((max_n_ner - len(ner_tag)), dtype=int).tolist()
            

def pad_srl_masks(items, EVI_LEN=5):
    """
    (batch_size, EVI_LEN, n_node, n_token)
    """
    max_n_node = 1
    max_n_token = 1
    for item in items:
        for item_per_evi in item:
            max_n_node = max(max_n_node, len(item_per_evi))
            for item_per_node in item_per_evi:
                max_n_token = max(max_n_token, len(item_per_node))
    
    for i, item in enumerate(items):
        for j, item_per_evi in enumerate(item):
            for k, item_per_node in enumerate(item_per_evi):
                item_per_evi[k] = item_per_node[:max_n_token]
                item_per_evi[k] += [0] * (max_n_token - len(item_per_node))
            if len(item_per_evi) < max_n_node:
                item[j] += np.zeros((max_n_node - len(item_per_evi), max_n_token), dtype=int).tolist()
        if len(item) < EVI_LEN:
            items[i] += np.zeros((EVI_LEN - len(item), max_n_node, max_n_token), dtype=int).tolist()
            

def pad_srl_tags(items, EVI_LEN=5):
    """
    (batch_size, EVI_LEN, n_node)
    """
    max_n_node = 1
    for item in items:
        for item_per_evi in item:
            max_n_node = max(max_n_node, len(item_per_evi))
        
    for i, item in enumerate(items):
        for j, item_per_evi in enumerate(item):
            if len(item_per_evi) < max_n_node:
                item[j] += np.zeros((max_n_node - len(item_per_evi)), dtype=int).tolist()
        if len(item) < EVI_LEN:
            items[i] += np.zeros((EVI_LEN - len(item), max_n_node), dtype=int).tolist()
            

def pad_srl_paths(items, PAD_VAL=-1, EVI_LEN=5, MAX_PATH_LEN=100):
    """
    (batch_size, EVI_LEN, n_path, path_len)
    """
    max_n_path = 0
    max_path_len = 0
    for item in items:
        for item_per_evi in item:
            max_n_path = max(max_n_path, len(item_per_evi))
            for item_per_path in item_per_evi:
                max_path_len = max(max_path_len, len(item_per_path))
    if max_path_len > MAX_PATH_LEN:
        max_path_len = MAX_PATH_LEN
    for i, item in enumerate(items):
        for j, item_per_evi in enumerate(item):
            for k, item_per_path in enumerate(item_per_evi):
                item_per_evi[k] = item_per_path[:max_path_len]
                item_per_evi[k] += [PAD_VAL] * (max_path_len - len(item_per_path))
            if len(item_per_evi) < max_n_path:
                item[j] += (np.zeros((max_n_path - len(item_per_evi), max_path_len), dtype=int)+PAD_VAL).tolist()
        if len(item) < EVI_LEN:
            items[i] += (np.zeros((EVI_LEN - len(item), max_n_path, max_path_len), dtype=int)+PAD_VAL).tolist()

            
# def generate_srl_ner_input(examples, srls_and_ners, tokenizer, EVI_LEN=5, MAX_SEQ_LENGTH=130):
#     batch_size = len(examples)
    
#     batch_ner_masks = [] # supposed to be (batch_size, n_ner, MAX_TOKEN)
#     batch_ner_tag_ids = [] # (batch_size, n_ner)
    
#     batch_claim_masks = [] # supposed to be (batch_size, n_evi, n_node, MAX_TOKEN)
#     batch_claim_tags = [] # (batch_size, n_evi, n_node)
#     batch_claim_paths = [] # (batch_size, n_evi, n_path, path_len)
    
#     batch_evi_masks = [] # supposed to be (batch_size, n_evi, n_node, MAX_TOKEN)
#     batch_evi_tags = [] # (batch_size, n_evi, n_node)
#     batch_evi_paths = [] # (batch_size, n_evi, n_path, path_len)
    
#     for example, srl_and_ner in zip(examples, srls_and_ners):
#         claim_srl = srl_and_ner['claim_srl']
#         claim_ner = srl_and_ner['claim_ner']
#         evis_srl = srl_and_ner['evis_srl']

#         merged_tokens = get_merged_tokens(example[0], tokenizer, MAX_SEQ_LENGTH)
#         ner_masks, ner_tag_ids = extract_ner(merged_tokens, claim_ner, claim_srl)
        
#         claim_graphs = []
#         evi_graphs = []
#         for ex, evi_srl in zip(example, evis_srl):
#             merged_tokens = get_merged_tokens(ex, tokenizer, MAX_SEQ_LENGTH)
#             claim_graph, evi_graph = construct_srl_graph(merged_tokens, claim_srl, evi_srl)
#             claim_graphs.append(claim_graph)
#             evi_graphs.append(evi_graph)
#         batch_ner_masks.append(ner_masks)
#         batch_ner_tag_ids.append(ner_tag_ids)
        
#         batch_claim_tags.append([a[0] for a in claim_graphs])
#         batch_claim_masks.append([a[1] for a in claim_graphs])
#         batch_claim_paths.append([a[2] for a in claim_graphs])
        
#         batch_evi_tags.append([a[0] for a in evi_graphs])
#         batch_evi_masks.append([a[1] for a in evi_graphs])
#         batch_evi_paths.append([a[2] for a in evi_graphs])
        
#     pad_ner_masks(batch_ner_masks, MAX_SEQ_LENGTH)
#     pad_ner_tagids(batch_ner_tag_ids)
    
#     pad_srl_masks(batch_claim_masks, EVI_LEN, MAX_SEQ_LENGTH)
#     pad_srl_tags(batch_claim_tags, EVI_LEN)
#     pad_srl_paths(batch_claim_paths, EVI_LEN)
    
#     pad_srl_masks(batch_evi_masks, EVI_LEN, MAX_SEQ_LENGTH)
#     pad_srl_tags(batch_evi_tags, EVI_LEN)
#     pad_srl_paths(batch_evi_paths, EVI_LEN)
    
#     ner_input = (batch_ner_masks, batch_ner_tag_ids)
#     claim_srl_input = (batch_claim_masks, batch_claim_tags, batch_claim_paths)
#     evi_srl_input = (batch_evi_masks, batch_evi_tags, batch_evi_paths)
#     return ner_input, claim_srl_input, evi_srl_input

def generate_srl_ner_input(srls_and_ners, EVI_LEN=5, MAX_SEQ_LENGTH=130):
    batch_size = len(srls_and_ners)
    
    batch_ner_masks = [] # supposed to be (batch_size, n_ner, n_node)
    batch_ner_tag_ids = [] # (batch_size, n_ner)
    
    batch_claim_masks = [] # supposed to be (batch_size, n_evi, n_node, n_token)
    batch_claim_tags = [] # (batch_size, n_evi, n_node)
    batch_claim_paths = [] # (batch_size, n_evi, n_path, path_len)
    
    batch_evi_masks = [] # supposed to be (batch_size, n_evi, n_node, n_token)
    batch_evi_tags = [] # (batch_size, n_evi, n_node)
    batch_evi_paths = [] # (batch_size, n_evi, n_path, path_len)
    
    for srl_and_ner in srls_and_ners:
        batch_ner_masks.append(srl_and_ner['ner_masks'])
        batch_ner_tag_ids.append(srl_and_ner['ner_tag_ids'])
        
        batch_claim_tags.append(srl_and_ner['claim_tags'])
        batch_claim_masks.append(srl_and_ner['claim_masks'])
        batch_claim_paths.append(srl_and_ner['claim_paths'])
        
        batch_evi_tags.append(srl_and_ner['evi_tags'])
        batch_evi_masks.append(srl_and_ner['evi_masks'])
        batch_evi_paths.append(srl_and_ner['evi_paths'])
    
    pad_ner_masks(batch_ner_masks)
    pad_ner_tagids(batch_ner_tag_ids)
    
    pad_srl_masks(batch_claim_masks, EVI_LEN)
    pad_srl_tags(batch_claim_tags, EVI_LEN)
    pad_srl_paths(batch_claim_paths, len(batch_claim_masks[0][0]), EVI_LEN)
    
    pad_srl_masks(batch_evi_masks, EVI_LEN)
    pad_srl_tags(batch_evi_tags, EVI_LEN)
    pad_srl_paths(batch_evi_paths, len(batch_evi_masks[0][0]), EVI_LEN)
    
    ner_input = (batch_ner_masks, batch_ner_tag_ids)
    claim_srl_input = (batch_claim_masks, batch_claim_tags, batch_claim_paths)
    evi_srl_input = (batch_evi_masks, batch_evi_tags, batch_evi_paths)
    return ner_input, claim_srl_input, evi_srl_input