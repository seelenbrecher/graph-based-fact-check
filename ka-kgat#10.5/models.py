import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder
from torch.autograd import Variable
import numpy as np

from prepare_concept import CONCEPT_DUMMY_IDX, REL_DUMMY_IDX



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
        
        # GAT on span-level representation
        self.span_use_gat = args.span_use_gat
        if self.span_use_gat:
            self.span_gat = GAT(args)
        
        self.concept_dim = args.concept_dim
        if self.span_use_gat:
            # because we feed concept features into GAT, the dimension = dim of GAT last features layers
            self.concept_dim = args.span_gat_n_heads[-2] * args.span_gat_n_features[-2]
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
        
        self.roberta = False


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
    
    def get_relation_features(self, rel_inp_ids, rel_segments, rel_indices):
        # rel_inp_ids = input to BERT
        # rel_segments = segmentize which part of relation each token belongs to
        # rel_indices = the relation index for each edge
        bs, inp_len = rel_inp_ids.shape
        _, n_rel, _ = rel_segments.shape
        masks = torch.ones((1, inp_len), dtype=torch.long).cuda()
        segments = torch.zeros((1, inp_len), dtype=torch.long).cuda()
        
        rel_inp = rel_inp_ids[0].view(1, -1)
        rel_seg = rel_segments[0].unsqueeze(0).transpose(2, 1)
        
        if self.roberta:
            out = self.pred_model(rel_inp, masks)
            rel_emb = out.last_hidden_state
        else:
            rel_emb, _ = self.pred_model(rel_inp, masks, segments)
        rel_emb = F.normalize(rel_emb, p=2, dim=2)
        rel_emb = rel_emb.transpose(2, 1)
        
        rel_features = rel_emb.bmm(rel_seg).transpose(2, 1)
        
        rel_seg = rel_seg.transpose(2, 1)
        n_tokens_per_rel = rel_seg.sum(dim=2).repeat(1, rel_features.shape[-1]).view(1, rel_features.shape[-1], -1).transpose(2, 1)
        rel_features = rel_features / n_tokens_per_rel
        rel_features = rel_features.squeeze(0)
        
        rel_features = rel_features[rel_indices]
        
        return rel_features
        
    
    def initialize_node(self, token_level_bert, t2c_tensor, tpool_tensor, graph_tensor):
        """
        initialize word-level node representation
        Add concept embedding into word-level node representation
        TODO: do self-attention by using GAT
        """
        # pool subword level to word level
        token_level_bert = token_level_bert.transpose(2, 1)
        token_level_bert = token_level_bert.bmm(tpool_tensor)
        token_level_bert = token_level_bert.transpose(2, 1)
        
        edge_indices, rel_indices, graph_inp_mask, rel_inp_ids, rel_segments = graph_tensor
        batch_size, evi_num, _, n_rel = edge_indices.shape # _ = 2, source and target dim
        edge_indices = edge_indices.view(-1, 2, n_rel)
        rel_indices = rel_indices.view(-1, n_rel)
        graph_inp_mask = graph_inp_mask.view(-1, n_rel)
        
        rel_indices = rel_indices.masked_fill_((1-graph_inp_mask).bool(), REL_DUMMY_IDX)
        
        rel_features = self.get_relation_features(rel_inp_ids, rel_segments, rel_indices)
            
        token_level_bert, _, _ = self.span_gat((token_level_bert, edge_indices, rel_features))
            
        return token_level_bert
        

    def forward(self, inputs, test=False, roberta=False):
        self.roberta = roberta
        
        bert_tensor, comb_tensor, graph_tensor, tpool_tensor = inputs
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
        if self.roberta:
            out = self.pred_model(inp_tensor, msk_tensor)
            inputs_hiddens = out.last_hidden_state
            inputs = out.pooler_output
        else:
            inputs_hiddens, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)

        inputs_hiddens = self.initialize_node(inputs_hiddens, comb_inp_tensor, tpool_tensor, graph_tensor)
        
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

#######################################################################################################################################
# implement GAT
# taken from https://github.com/gordicaleksa/pytorch-GAT
class GAT(torch.nn.Module):
    """
    The most interesting and hardest implementation is implementation #3.
    Imp1 and imp2 differ in subtle details but are basically the same thing.

    So I'll focus on imp #3 in this notebook.

    """

    def __init__(self, args):
        super().__init__()
        self.num_of_layers = args.span_gat_n_layers
        self.num_heads_per_layer = args.span_gat_n_heads
        self.num_features_per_layer = args.span_gat_n_features
        self.add_skip_connection = args.span_gat_add_skip_conn
        self.bias = args.span_gat_bias
        self.dropout = args.span_gat_dropout
        self.log_attention_weights = args.span_gat_log_attention_weights
        assert self.num_of_layers == len(self.num_heads_per_layer) == len(self.num_features_per_layer) - 1, f'Enter valid arch params.'

        self.num_heads_per_layer = [1] + self.num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(self.num_of_layers - 1): #until -1 because I don't want to include the prediction layers. only the extractor layers
            layer = GATLayer(
                num_in_features=self.num_features_per_layer[i] * self.num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=self.num_features_per_layer[i+1],
                num_of_heads=self.num_heads_per_layer[i+1],
                concat=True if i < self.num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < self.num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=self.dropout,
                add_skip_connection=self.add_skip_connection,
                bias=self.bias,
                log_attention_weights=self.log_attention_weights,
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, edge_index) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    bs_dim = 0
    nodes_dim = 1      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 2       # attention head dim

    def __init__(self, num_in_features, num_out_features, num_of_heads, relation_num=0, relation_dim=0, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()
        
    def forward(self, data):
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features, edge_index, rel_features = data  # unpack data
        batch_size = in_nodes_features.shape[self.bs_dim]
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[1] == 2, f'Expected edge index with shape=(BATCH_SIZE, 2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (bs, N, FIN) * (FIN, NH*FOUT) -> (bs, N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(batch_size, -1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (bs, N, NH, FOUT) * (1, NH, FOUT) -> (bs, N, NH, 1) -> (bs, N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (bs, E, NH), nodes_features_proj_lifted shape = (bs, E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (bs, E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[:, self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (bs, E, NH, FOUT) * (bs, E, NH, 1) -> (bs, E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        
        rel_features_proj = self.linear_proj(rel_features).view(batch_size, -1, self.num_of_heads, self.num_out_features)
        rel_features_proj = self.dropout(rel_features_proj)
        
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted_weighted + rel_features_proj

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (bs, N, NH, FOUT)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, rel_features)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        N, n_edges, n_features = scores_per_edge.shape
        scores_per_edge = scores_per_edge.view(scores_per_edge.size(0), -1)
        scores_per_edge = scores_per_edge - scores_per_edge.max(dim=-1)[0].view(N, 1)
        scores_per_edge = scores_per_edge.view(N, n_edges, n_features)
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (bs, E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        N, n_edges, n_features = exp_scores_per_edge.shape
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (bs, N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (bs, N, NH) -> (bs, E, NH)
        return torch.cat([torch.index_select(neighborhood_sums[i], 0, trg_index[i]) for i in range(N)]).view(N, n_edges, n_features)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (bs, N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (bs, E) -> (bs, E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[:, self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (bs, E, NH, FOUT) -> (bs, N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        N, seq_len, n_features = scores_source.shape
        _, _, n_edges = edge_index.shape
        src_nodes_index = edge_index[:, self.src_nodes_dim]
        trg_nodes_index = edge_index[:, self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = torch.cat([torch.index_select(scores_source[i], 0, src_nodes_index[i]) for i in range(N)])
        scores_source = scores_source.view(N, n_edges, n_features)
        
        scores_target = torch.cat([torch.index_select(scores_target[i], 0, trg_nodes_index[i]) for i in range(N)])
        scores_target = scores_target.view(N, n_edges, n_features)
        
        nodes_features_matrix_proj_lifted = torch.cat(
            [torch.index_select(nodes_features_matrix_proj[i], 0, src_nodes_index[i]) for i in range(N)]
        )
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj_lifted.view(N, n_edges, n_features, -1)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        batch_size, n_edges, n_features, _ = attention_coefficients.shape
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients
        
        if self.add_skip_connection:  # add skip or residual connection
            # this line is not originated from GAT. I did this so if the node does not have an edge, it will take the initial repr
            in_nodes_features = in_nodes_features.view(batch_size, -1, self.num_of_heads, self.num_out_features)
            ###
            
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
#                 out_nodes_features += in_nodes_features.unsqueeze(1)
                # i changed to this
                out_nodes_features += in_nodes_features
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(batch_size, -1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (bs, N, NH, FOUT) -> (bs, N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(batch_size, -1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            assert False
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    
# end of GAT implementation
#######################################################################################################################################










