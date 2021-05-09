import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder
from torch.autograd import Variable
import numpy as np



def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma


class ConceptEmbeddingModel(nn.Module):
    def __init__(self, pretrained_concept, pretrained_relation, args):
        super(ConceptEmbeddingModel, self).__init__()
        self.concept_num = args.concept_num
        self.relation_num = args.relation_num
        self.concept_dim = args.concept_dim
        self.relation_dim = args.relation_dim
        
        self.concept_emb = nn.Embedding(self.concept_num, self.concept_dim)
        self.relation_emb = nn.Embedding(self.relation_num, self.relation_dim)
        
        if pretrained_concept is not None:
            self.concept_emb.weight.data.copy_(pretrained_concept)
        else:
            bias = np.sqrt(6.0 / self.concept_dim)
            nn.init.uniform_(self.concept_emb.weight, -bias, bias)

        if pretrained_relation is not None:
            self.relation_emb.weight.data.copy_(pretrained_relation)
        else:
            bias = np.sqrt(6.0 / self.relation_dim)
            nn.init.uniform_(self.relation_emb.weight, -bias, bias)
    
    def forward(self, concept_inp, relation_inp):
        concept_output = None
        relation_output = None
        
        if concept_inp is not None:
            concept_output = self.concept_emb(concept_inp)
        
        if relation_inp is not None:
            relation_output = self.relation_emb(relation_inp)
        
        return concept_output, relation_output
        

class inference_model(nn.Module):
    def __init__(self, bert_model, concept_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        self.evi_num = args.evi_num
        self.nlayer = args.layer
        self.kernel = args.kernel
        
        self.node_dim = args.node_dim
        
        # concept embedding
        self.use_concept = args.use_concept
        self.concept_dim = args.concept_dim
        if self.use_concept:
            self.concept_model = concept_model
            self.bert2concept_alignment = nn.Sequential(
                Linear(self.bert_hidden_dim + self.concept_dim, self.bert_hidden_dim * 2),
                ReLU(True),
                Linear(self.bert_hidden_dim * 2, self.node_dim)
            )
        
        self.proj_inference_de = nn.Linear(self.node_dim * 2, self.num_labels)
        self.proj_att = nn.Linear(self.kernel, 1)
        self.proj_input_de = nn.Linear(self.node_dim, self.node_dim)
        self.proj_gat = nn.Sequential(
            Linear(self.bert_hidden_dim * 2, 128),
            ReLU(True),
            Linear(128, 1)
        )
        self.proj_select = nn.Linear(self.kernel, 1)
        self.mu = Variable(torch.FloatTensor(kernal_mus(self.kernel)), requires_grad = False).view(1, 1, 1, 21).cuda()
        self.sigma = Variable(torch.FloatTensor(kernel_sigmas(self.kernel)), requires_grad = False).view(1, 1, 1, 21).cuda()


    def self_attention(self, inputs, inputs_hiddens, mask, mask_evidence, index):
        idx = torch.LongTensor([index]).cuda()
        mask = mask.view([-1, self.evi_num, self.max_len])
        mask_evidence = mask_evidence.view([-1, self.evi_num, self.max_len])
        own_hidden = torch.index_select(inputs_hiddens, 1, idx)
        own_mask = torch.index_select(mask, 1, idx)
        own_input = torch.index_select(inputs, 1, idx)
        own_hidden = own_hidden.repeat(1, self.evi_num, 1, 1)
        own_mask = own_mask.repeat(1, self.evi_num, 1)
        own_input = own_input.repeat(1, self.evi_num, 1)

        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)
        own_norm = F.normalize(own_hidden, p=2, dim=-1)

        att_score = self.get_intersect_matrix_att(hiddens_norm.view(-1, self.max_len, self.node_dim), own_norm.view(-1, self.max_len, self.node_dim),
                                                  mask_evidence.view(-1, self.max_len), own_mask.view(-1, self.max_len))
        att_score = att_score.view(-1, self.evi_num, self.max_len, 1)
        #if index == 1:
        #    for i in range(self.evi_num):
        #print (att_score.view(-1, self.evi_num, self.max_len)[0, 1, :])
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)
        weight_inp = torch.cat([own_input, inputs], -1)
        weight_inp = self.proj_gat(weight_inp)
        weight_inp = F.softmax(weight_inp, dim=1)
        outputs = (inputs * weight_inp).sum(dim=1)
        weight_de = torch.cat([own_input, denoise_inputs], -1)
        weight_de = self.proj_gat(weight_de)
        weight_de = F.softmax(weight_de, dim=1)
        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs, outputs_de

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1], 1)
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu.cuda()) ** 2) / (self.sigma.cuda() ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1) / (torch.sum(attn_q, 1) + 1e-10)
        log_pooling_sum = self.proj_select(log_pooling_sum).view([-1, 1])
        return log_pooling_sum

    def get_intersect_matrix_att(self, q_embed, d_embed, attn_q, attn_d):
        """
        Calculate Node Kernel
        """
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1])
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu.cuda()) ** 2) / (self.sigma.cuda() ** 2) / 2)) * attn_d
        log_pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(log_pooling_sum, min=1e-10))
        log_pooling_sum = self.proj_att(log_pooling_sum).squeeze(-1)
        log_pooling_sum = log_pooling_sum.masked_fill_((1 - attn_q).bool(), -1e4)
        log_pooling_sum = F.softmax(log_pooling_sum, dim=1)
        return log_pooling_sum
    
    def initialize_node(self, token_level_bert, t2c_tensor, tpool_tensor):
        """
        initialize word-level node representation
        Add concept embedding into word-level node representation
        TODO: do self-attention by using GCN
        """
        # pool subword level to word level
        token_level_bert = token_level_bert.transpose(2, 1)
        token_level_bert = token_level_bert.bmm(tpool_tensor)
        token_level_bert = token_level_bert.transpose(2, 1)
        
        if self.use_concept:
            # get concept representation
            token_level_concept, _ = self.concept_model(t2c_tensor, None)

            # align bert and concept representation
            token_level_bert = torch.cat((token_level_bert, token_level_concept), dim=2)
            token_level_bert = self.bert2concept_alignment(token_level_bert)
        return token_level_bert
        

    def forward(self, inputs):
        bert_tensor, comb_tensor, tpool_tensor = inputs
        inp_tensor, msk_tensor, seg_tensor = bert_tensor
        comb_inp_tensor, comb_msk_tensor, comb_seg_tensor = comb_tensor
        
        msk_tensor = msk_tensor.view(-1, self.max_len)
        inp_tensor = inp_tensor.view(-1, self.max_len)
        seg_tensor = seg_tensor.view(-1, self.max_len)
        
        comb_inp_tensor = comb_inp_tensor.view(-1, self.max_len)
        comb_msk_tensor = comb_msk_tensor.view(-1, self.max_len)
        comb_seg_tensor = comb_seg_tensor.view(-1, self.max_len)
        
        tpool_tensor = tpool_tensor.view(-1, self.max_len, self.max_len)
        
        # initialize node's representations
        inputs_hiddens, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)
        inputs_hiddens = self.initialize_node(inputs_hiddens, comb_inp_tensor, tpool_tensor)
        
        # now we are using comb_msk_tensor and comb_seg_tensor
        # because we merge some sub-word levels from BERT tokenizer to span level
#         mask_text = msk_tensor.view(-1, self.max_len).float()
#         mask_text[:, 0] = 0.0
#         mask_claim = (1 - seg_tensor.float()) * mask_text
#         mask_evidence = seg_tensor.float() * mask_text
        mask_text = comb_msk_tensor.view(-1, self.max_len).float()
        mask_text[:, 0] = 0.0
        mask_claim = (1 - comb_seg_tensor.float()) * mask_text
        mask_evidence = comb_seg_tensor.float() * mask_text
        
        # calculate Node Kernel
        inputs_hiddens = inputs_hiddens.view(-1, self.max_len, self.node_dim)
        inputs_hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=2)
        log_pooling_sum = self.get_intersect_matrix(inputs_hiddens_norm, inputs_hiddens_norm, mask_claim, mask_evidence)
        log_pooling_sum = log_pooling_sum.view([-1, self.evi_num, 1])
        select_prob = F.softmax(log_pooling_sum, dim=1)
        
        # calculate Edge kernel
        inputs = inputs.view([-1, self.evi_num, self.bert_hidden_dim])
        inputs_hiddens = inputs_hiddens.view([-1, self.evi_num, self.max_len, self.node_dim])
        inputs_att_de = []
        for i in range(self.evi_num):
            outputs, outputs_de = self.self_attention(inputs, inputs_hiddens, mask_text, mask_text, i)
            inputs_att_de.append(outputs_de)
        
        # calculate prob
        inputs_att = inputs.view([-1, self.evi_num, self.node_dim])
        inputs_att_de = torch.cat(inputs_att_de, dim=1)
        inputs_att_de = inputs_att_de.view([-1, self.evi_num, self.node_dim])
        inputs_att = torch.cat([inputs_att, inputs_att_de], -1)
        inference_feature = self.proj_inference_de(inputs_att)
        class_prob = F.softmax(inference_feature, dim=2)
        prob = torch.sum(select_prob * class_prob, 1)
        prob = torch.log(prob)
        return prob





















