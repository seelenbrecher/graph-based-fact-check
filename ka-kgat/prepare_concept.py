import numpy as np
import torch

def add_concept_args(parser):
    parser.add_argument('--concept_emb', default='../checkpoint/transe/glove.transe.sgd.ent.npy', type=str)
    parser.add_argument('--concept_num', default=799274, type=int)
    parser.add_argument('--concept_dim', default=100, type=int)
    
    parser.add_argument('--relation_emb', default='../checkpoint/transe/glove.transe.sgd.rel.npy', type=str)
    parser.add_argument('--relation_num', default=17, type=int)
    parser.add_argument('--relation_dim', default=100, type=int)
    
    parser.add_argument('--node_dim', default=768, type=int)
    
    parser.add_argument('--use_concept', action='store_true', default=False)
    
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