from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import obj_edge_vectors, load_word_vectors
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

from .CaptionModel import CaptionModel
import models.lib.gcn_backbone as GBackbone
import models.lib.gpn as GPN

import models.lib.transformer_encoder as transformer_encoder
from models.lib.attention import ScaledDotProductAttention, ScaledDotProductWithBoxAttention, MutualBiAffineAttention

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    """
    for batch computation, pack sequences with different lenghth with explicit setting the batch size at each time step
    """
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

"""
Captioning model using image scene graph
"""
class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size 
        self.num_layers = opt.num_layers  
        self.drop_prob_lm = opt.drop_prob_lm 
        self.seq_length = opt.max_length or opt.seq_length 
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size 
        self.att_hid_size = opt.att_hid_size 
        self.use_bn = opt.use_bn 
        self.ss_prob = opt.sampling_prob 

        # MyAdd: use_box : positional encoding
        self.use_box = True if opt.use_box == 1 else False
        if self.use_box:
            print("use box.")
        self.use_biAffine = True if opt.use_biAffine == 1 else False

        self.gpn = True if opt.use_gpn == 1 else False 
        self.embed_dim = opt.embed_dim 
        self.GCN_dim = opt.gcn_dim
        # MyAdd: transformer d_model
        self.d_model = 1024
        self.noun_fuse = True if opt.noun_fuse == 1 else False  
        self.pred_emb_type = opt.pred_emb_type 
        self.GCN_layers = opt.gcn_layers 
        self.GCN_residual = opt.gcn_residual  
        self.GCN_use_bn = False if opt.gcn_bn == 0 else True   

        self.test_LSTM = False if getattr(opt, 'test_LSTM', 0) == 0 else True 
        self.topk_sampling = False if getattr(opt, 'use_topk_sampling', 0) == 0 else True
        self.topk_temp = getattr(opt, 'topk_temp', 0.6)
        self.the_k = getattr(opt, 'the_k', 3)
        self.sct = False if getattr(opt, 'sct', 0) == 0 else True # show-control-tell testing mode

        # feature fusion layer
        self.obj_v_proj = nn.Linear(self.att_feat_size, self.GCN_dim)
        object_names = np.load(opt.obj_name_path,encoding='latin1') # [0] is 'background'
        self.sg_obj_cnt = object_names.shape[0]
        if self.noun_fuse:
            embed_vecs = obj_edge_vectors(list(object_names), wv_dim=self.embed_dim)
            self.sg_obj_embed = nn.Embedding(self.sg_obj_cnt, self.embed_dim)
            self.sg_obj_embed.weight.data = embed_vecs.clone()
            self.obj_emb_proj = nn.Linear(self.embed_dim, self.GCN_dim)
            self.relu = nn.ReLU(inplace=True)
        predicate_names = np.load(opt.rel_name_path,encoding='latin1') # [0] is 'background'
        self.sg_pred_cnt = predicate_names.shape[0]
        p_embed_vecs = obj_edge_vectors(list(predicate_names), wv_dim=self.embed_dim)
        self.sg_pred_embed = nn.Embedding(predicate_names.shape[0], self.embed_dim)
        self.sg_pred_embed.weight.data = p_embed_vecs.clone()
        self.pred_emb_prj = nn.Linear(self.embed_dim, self.GCN_dim)

        # GCN backbone
        self.gcn_backbone = GBackbone.gcn_backbone(GCN_layers=self.GCN_layers, GCN_dim=self.GCN_dim, \
                                                   GCN_residual=self.GCN_residual, GCN_use_bn=self.GCN_use_bn)


        # MyAdd: Transformer encoder backbone
        self.transformer_encoder = transformer_encoder.TransformerEncoder(N=self.GCN_layers, padding_idx=0, attention_module=ScaledDotProductAttention)
        # MyAdd: Transformer encoder with positional encoding (a boxes input more)
        self.transformer_encoder_with_box = transformer_encoder.TransformerEncoderWithBox(N=self.GCN_layers, padding_idx=0, attention_module=ScaledDotProductWithBoxAttention)
        self.box_embedding = nn.Linear(4, 1024)

        # GPN (sGPN)
        if self.gpn:
            self.gpn_layer = GPN.gpn_layer(GCN_dim=self.GCN_dim, hid_dim=self.att_hid_size, \
                                           test_LSTM=self.test_LSTM, use_nms=False if self.sct else True, \
                                           iou_thres=getattr(opt, 'gpn_nms_thres', 0.75), \
                                           max_subgraphs=getattr(opt, 'gpn_max_subg', 1), \
                                           use_sGPN_score=True if getattr(opt, 'use_gt_subg', 0) == 0 else False)
        else:
            self.read_out_proj = nn.Sequential(nn.Linear(self.GCN_dim, self.att_hid_size), nn.Linear(self.att_hid_size,self.GCN_dim*2))
            nn.init.constant_(self.read_out_proj[0].bias, 0)
            nn.init.constant_(self.read_out_proj[1].bias, 0)
        
        # projection layers in attention-based LSTM
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.fc_feat_size),
                                    nn.ReLU(),
                                    nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.GCN_dim, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_2 = nn.Linear(self.rnn_size, self.att_hid_size)

        if self.use_biAffine:
            print("use biAffine.")
            self.biAffineAttention = MutualBiAffineAttention(hidden_size=1024)
            self.layer_norm = nn.LayerNorm(self.d_model)

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, trip_pred=None, obj_dist=None, obj_box=None, rel_ind=None, \
                 pred_fmap=None, pred_dist=None, gpn_obj_ind=None, gpn_pred_ind=None, gpn_nrel_ind=None,gpn_pool_mtx=None):
        """
        Model feedforward: input scene graph features and sub-graph indices, output token probabilities
        fusion layers --> GCN backbone --> GPN (sGPN) --> attention-based LSTM
        """
        # fuse features (visual, embedding) for each node in graph
        att_feats, pred_fmap = self.feat_fusion(obj_dist, att_feats, pred_dist)
        b = att_feats.size(0); N = att_feats.size(1); K = rel_ind.size(1); L = self.GCN_dim
        # print("b = ", b) # 32
        # MyAdd: Transformer encoder backbone
        if not self.use_box:
            self_att_feats, self_att_masks = self.transformer_encoder(att_feats)
        else:
            region_embed = self.box_embedding(obj_box)
            self_att_feats, self_att_masks = self.transformer_encoder_with_box(att_feats, obj_box,
                                                                               region_embed=region_embed)
        # repeat to 5 counterparts
        self_att_feats = self_att_feats.view(b, 1, N, self.d_model).expand(b, 5, N, self.d_model).contiguous().view(-1, N, self.d_model)
        self_att_masks = self_att_masks.squeeze().view(b, 1, N).expand(b, 5, N).contiguous().view(-1, N)



        # GCN backbone (will expand feats to 5 counterparts)
        att_feats, x_pred = self.gcn_backbone(b,N,K,L,att_feats, obj_dist, pred_fmap, rel_ind)
        b = att_feats.size(0) # has expanded to 5 counterparts

        # MyAdd: look gcn output
        # print("[AttModel gcn_backbone out] att_feats size = ", att_feats.size())
        # print("[AttModel gcn_backbone out] att_masks size = ", att_masks.size())

        # sGPN
        if self.gpn:
            gpn_loss, subgraph_score, att_feats, fc_feats, att_masks = \
                self.gpn_layer(b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks)
        else: # no gpn module, baseline model with full scene graph
            gpn_loss = None
            subgraph_score = None

            # mean pooling, wo global img feats
            read_out = torch.mean(att_feats,1).detach()  # mean pool over full scene graph
            fc_feats = self.read_out_proj(read_out) 

            att_masks = att_masks[:,0,0]
            att_masks[:,:36].fill_(1.0).float()  # MyNote: like max_len = 36, only look top 36 objects
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        # MyAdd: prepare 2 kind of features
        # MyAdd: look gcn output
        # print("[AttModel gpn out] att_feats size = ", att_feats.size())
        # print("[AttModel gpn out] fc_feats size = ", fc_feats.size())
        # print("[AttModel gpn out] att_masks size = ", att_masks.size())

        # MyAdd: use BiAffine attention
        if self.use_biAffine:
            att_feats1 = self.biAffineAttention(self_att_feats, att_feats, att_masks)
            self_att_feats1 = self.biAffineAttention(att_feats, self_att_feats, self_att_masks)
            att_feats = self.layer_norm(att_feats + att_feats1)
            self_att_feats = self.layer_norm(self_att_feats + self_att_feats1)

        # MyAdd: concat 2 encoder's output
        # print("att_feats size = ", att_feats.size())
        # print("self_att_feats size = ", self_att_feats.size())
        att_feats = torch.cat([att_feats, self_att_feats], dim=1)
        self_att_masks = self_att_masks.float()
        # print("cat att_feats size = ", att_feats.size()) # [160, 74, 1024]
        # print("att_masks size = ", att_masks.size())
        # print("self_att_masks size = ", self_att_masks.size())
        att_masks = torch.cat([att_masks, self_att_masks], dim=1)

        # Prepare the features for attention-based LSTM
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # print("features prepared.")
        # MyAdd: look prepared att_feats
        # print("[AttModel] p_fc_feats size = ", p_fc_feats.size())
        # print("[AttModel] p_att_feats size = ", p_att_feats.size())
        # print("[AttModel] pp_att_feats size = ", pp_att_feats.size())
        # print("[AttModel] p_att_masks size = ", p_att_masks.size())


        # MyNote: decoder part
        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output  # output is probability after log_softmax at current time step, sized [batch, self.vocab_size+1]
        # print("*************** AttModel end ***************")
        return outputs, gpn_loss, subgraph_score

    def _sample_sentences(self, fc_feats, att_feats, att_masks=None, trip_pred=None, obj_dist=None, obj_box=None, rel_ind=None, \
                                pred_fmap=None, pred_dist=None, gpn_obj_ind=None, gpn_pred_ind=None, gpn_nrel_ind=None,gpn_pool_mtx=None, opt={}):
        """
        Model inference / sentence decoding: generate captions with beam size > 1
        """
        # fuse features (visual, embedding) for each node in graph
        att_feats, pred_fmap = self.feat_fusion(obj_dist, att_feats, pred_dist)
        b = att_feats.size(0); N = att_feats.size(1); K = rel_ind.size(1); L = self.GCN_dim

        # MyAdd: Transformer encoder backbone
        if not self.use_box:
            self_att_feats, self_att_masks = self.transformer_encoder(att_feats)
        else:
            region_embed = self.box_embedding(obj_box)
            self_att_feats, self_att_masks = self.transformer_encoder_with_box(att_feats, obj_box,
                                                                               region_embed=region_embed)


        # GCN backbone
        att_feats, x_pred = self.gcn_backbone(b,N,K,L,att_feats, obj_dist, pred_fmap, rel_ind)
        b = att_feats.size(0) # has expanded to 5 counterparts

        # print("[sample gcnbackbone] att_feats size = ", att_feats.size())

        # GPN
        if self.gpn:
            gpn_loss, subgraph_score, att_feats, fc_feats, att_masks, keep_ind = \
                self.gpn_layer(b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks)
        else: # no gpn module, baseline model that use full graph
            gpn_loss = None
            att_feats = att_feats[0:1] # use one of 5 counterparts

            read_out = torch.mean(att_feats,1)  # mean pool over full scene graph
            fc_feats = self.read_out_proj(read_out) 

            att_masks = att_masks[0:1,0,0] 
            att_masks[:,:36].fill_(1.0).float()
            keep_ind = torch.arange(att_feats.size(0)).type_as(gpn_obj_ind)  
            subgraph_score = torch.arange(att_feats.size(0)).fill_(1.0).type_as(att_feats)

        # MyAdd: prepare 2 kind of features
        # repeat to equal counterparts
        bs = att_feats.size(0)
        self_att_feats = self_att_feats.expand(bs, N, self.d_model).contiguous()
        self_att_masks = self_att_masks.squeeze().unsqueeze(0).expand(bs, N).contiguous()

        # print("[sample_sentence] att_feats size = ", att_feats.size())  # [10, 37, 1024]
        # print("[sample_sentence] att_masks size = ", att_masks.size())  # [10, 37]

        # MyAdd: use BiAffine attention
        if self.use_biAffine:
            att_feats1 = self.biAffineAttention(self_att_feats, att_feats, att_masks)
            self_att_feats1 = self.biAffineAttention(att_feats, self_att_feats, self_att_masks)
            att_feats = self.layer_norm(att_feats + att_feats1)
            self_att_feats = self.layer_norm(self_att_feats + self_att_feats1)

        #
        # print("[sample_sentence] self_att_feats size = ", self_att_feats.size())  # [5, 37, 1024]
        # print("[sample_sentence] self_att_masks size = ", self_att_masks.size())  # [5, 37]
        # self_att_feats = self_att_feats[0:1]
        # self_att_masks = self_att_masks[0:1]
        att_feats = torch.cat([att_feats, self_att_feats], dim=1)
        self_att_masks = self_att_masks.float()
        att_masks = torch.cat([att_masks, self_att_masks], dim=1)

        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)


        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)
            
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, None, None, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), subgraph_score, keep_ind

    def _sample(self, fc_feats, att_feats, att_masks=None, trip_pred=None, obj_dist=None, obj_box=None, rel_ind=None, \
                pred_fmap=None, pred_dist=None, gpn_obj_ind=None, gpn_pred_ind=None, gpn_nrel_ind=None,gpn_pool_mtx=None, opt={}):
        """
        Model inference / sentence decoding: generate captions with beam size == 1 (disabling beam search)
        """
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        return_att = True if opt.get('return_att', 0) == 1 else False

        if beam_size > 1:
            return self._sample_sentences(fc_feats, att_feats, att_masks, trip_pred, obj_dist, obj_box, rel_ind, pred_fmap, pred_dist, \
                                      gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx, opt)
        
        # fuse features (visual, embedding) for each node in graph
        att_feats, pred_fmap = self.feat_fusion(obj_dist, att_feats, pred_dist)
        b = att_feats.size(0); N = att_feats.size(1); K = rel_ind.size(1); L = self.GCN_dim

        # MyAdd: Transformer encoder backbone
        if not self.use_box:
            self_att_feats, self_att_masks = self.transformer_encoder(att_feats)
        else:
            region_embed = self.box_embedding(obj_box)
            self_att_feats, self_att_masks = self.transformer_encoder_with_box(att_feats, obj_box,
                                                                               region_embed=region_embed)
        # repeat to 5 counterparts
        self_att_feats = self_att_feats.view(b, 1, N, self.d_model).expand(b, 5, N, self.d_model).contiguous().view(-1, N, self.d_model)
        self_att_masks = self_att_masks.squeeze().view(b, 1, N).expand(b, 5, N).contiguous().view(-1, N)

        # GCN backbone
        att_feats, x_pred = self.gcn_backbone(b,N,K,L,att_feats, obj_dist, pred_fmap, rel_ind)
        b = att_feats.size(0) # has expanded to 5 counterparts
        
        # GPN
        if self.gpn:
            gpn_loss, subgraph_score, att_feats, fc_feats, att_masks, keep_ind = \
                self.gpn_layer(b,N,K,L,gpn_obj_ind, gpn_pred_ind, gpn_nrel_ind,gpn_pool_mtx,att_feats,x_pred,fc_feats,att_masks)
        else: # no gpn module, baseline model that use full graph
            gpn_loss = None
            att_feats = att_feats[0:1] # use one of 5 counterparts

            read_out = torch.mean(att_feats,1)  # mean pool over full scene graph
            fc_feats = self.read_out_proj(read_out) 

            att_masks = att_masks[0:1,0,0] 
            att_masks[:,:36].fill_(1.0).float()
            keep_ind = torch.arange(att_feats.size(0)).type_as(gpn_obj_ind)  
            subgraph_score = torch.arange(att_feats.size(0)).fill_(1.0).type_as(att_feats)

        # MyAdd: use BiAffine attention
        if self.use_biAffine:
            att_feats1 = self.biAffineAttention(self_att_feats, att_feats, att_masks)
            self_att_feats1 = self.biAffineAttention(att_feats, self_att_feats, self_att_masks)
            att_feats = self.layer_norm(att_feats + att_feats1)
            self_att_feats = self.layer_norm(self_att_feats + self_att_feats1)


        # MyAdd: prepare 2 kind of features
        att_feats = torch.cat([att_feats, self_att_feats], dim=1)
        self_att_masks = self_att_masks.float()
        att_masks = torch.cat([att_masks, self_att_masks], dim=1)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        att2_weights = []
        
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            if return_att:
                logprobs, state, att_weight = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, return_att=False)
                att2_weights.append(att_weight)
            else:
                logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break

            if self.topk_sampling:  # sample top-k word from a re-normalized probability distribution
                logprobs = F.log_softmax(logprobs / float(self.topk_temp), dim=1)
                tmp = torch.empty_like(logprobs).fill_(float('-inf'))
                topk, indices = torch.topk(logprobs, self.the_k, dim=1)
                tmp = tmp.scatter(1, indices, topk)
                logprobs = tmp
                # sample the word index according to log probability (negative values)
                it = torch.distributions.Categorical(logits=logprobs.data).sample() # logits: log(probability) are negative values
                sampleLogprobs = logprobs.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions 
            else:                
                if sample_max: # True (greedy decoding)
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()               

            # stop when all finished, unfinished: 0 or 1
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # early quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
      
        if return_att:
            # attention weights [b,20+1,N]
            att2_weights = torch.cat([_.unsqueeze(1) for _ in att2_weights], 1)
            return seq, seqLogprobs, subgraph_score, keep_ind, att2_weights
        else:
            return seq, seqLogprobs, subgraph_score, keep_ind 

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, sg_emb=None, p_sg_emb=None,return_att=False):
        """
        Attention-based LSTM feedforward
        """
        xt = self.embed(it) # 'it' contains a word index
        
        if return_att:
            output, state, att_weight = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks, return_att=return_att)
            logprobs = F.log_softmax(self.logit(output), dim=1)
            return logprobs, state, att_weight
        else:
            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
            logprobs = F.log_softmax(self.logit(output), dim=1)
            return logprobs, state

    def init_hidden(self, bsz):
        weight = self.logit.weight if hasattr(self.logit, "weight") else self.logit[0].weight
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks, sg_emb=None):
        """
        Project features and prepare for the inputs of attention-based LSTM
        """
        # MyAdd : prepare two kind of features(gcn and self-attn)
        bs, N, _ = att_feats.shape
        # print("att_feats shape = ", att_feats.shape) # [320, 74, 1024]
        self_att_feats = att_feats[:, N // 2:, :]
        att_feats = att_feats[:, : N // 2, :]
        # print("att_feats size = ", att_feats.size()) # [500,37,1024]

        # print("self_att_feats size = ", self_att_feats.size()) #  [500,37,1024]
        self_att_masks = att_masks[:, N // 2:]
        att_masks = att_masks[:, :N // 2]
        # MyNote: In self attention mask of transformer, 1 means masking it out, while in attn_lstm 1 means keeping it
        self_att_masks = (self_att_masks == 0.0).float()
        # print("self_att_masks size = ", self_att_masks.size())
        # print("att_masks size = ", att_masks.size())
        att_feats, att_masks = self.clip_att(att_feats, att_masks)


        # print("sum = ", self_att_masks.sum())
        # s = self_att_masks.data.long().sum(1)
        # print("s = ", s)
        # print("self_att_masks = ", self_att_masks.dtype)
        # print("att_masks = ", att_masks.dtype)
        # exit(1)
        self_att_feats, self_att_masks = self.clip_att(self_att_feats, self_att_masks)


        # print("self_att_mask == att_mask? ", self_att_masks.sum(), " ", att_masks.sum())
        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) # pack sequences with different length
        # print("self_att_feats size = ", self_att_feats.size())
        # print("[prepare]att_feats size = ", att_feats.size()) #[3, 36, 1000]
        self_att_feats = pack_wrapper(self.att_embed, self_att_feats, self_att_masks)

        # delete the last dummy padded node
        self_att_feats = self_att_feats[:, :-1, :]
        self_att_masks = self_att_masks[:, :-1]

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)
        p_self_att_feats = self.ctx2att_2(self_att_feats)

        # MyAdd: concat 2 kind of features

        # print("before cat: att_feats size = ", att_feats.size())
        # print("before cat: self_att_feats size = ", self_att_feats.size())
        # print("before cat: att_masks size = ", att_masks.size())
        # print("before cat: self_att_masks size = ", self_att_masks.size())
        att_feats = torch.cat([att_feats, self_att_feats], dim=1)
        att_masks = torch.cat([att_masks, self_att_masks], dim=1)
        p_att_feats = torch.cat([p_att_feats, p_self_att_feats], dim=1)

        
        return fc_feats, att_feats, p_att_feats, att_masks

    def feat_fusion(self, obj_dist, att_feats, pred_dist):
        """
        Fuse visual and word embedding features for nodes and edges
        """
        # fuse features (visual, embedding) for each node in graph
        if self.noun_fuse: # Sub-GC
            obj_emb = self.obj_emb_proj(self.sg_obj_embed(obj_dist.view(-1, self.sg_obj_cnt)[:,1:].max(1)[1] + 1)).view(obj_dist.size(0), obj_dist.size(1), self.GCN_dim)
            att_feats = self.obj_v_proj(att_feats)
            att_feats = self.relu(att_feats + obj_emb)
        else: # GCN-LSTM baseline that use full graph
            att_feats = self.obj_v_proj(att_feats)
        
        if self.pred_emb_type == 1: # hard emb, not including background
            pred_emb = self.sg_pred_embed(pred_dist.view(-1, self.sg_pred_cnt)[:,1:].max(1)[1] + 1)
        elif self.pred_emb_type == 2: # hard emb, including background
            pred_emb = self.sg_pred_embed(pred_dist.view(-1, self.sg_pred_cnt).max(1)[1])
        pred_fmap = self.pred_emb_prj(pred_emb).view(pred_dist.size(0), pred_dist.size(1), self.GCN_dim) 
        return att_feats, pred_fmap


"""
Attention-based LSTM
"""
class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.attention = Attention(opt)
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) 
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.gated_weight_fc1 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.gated_weight_fc2 = nn.Linear(opt.rnn_size * 2, opt.rnn_size)
        self.init_weights()

    def init_weights(self):
        '''
        Init the weight of gated_weight_fc
        '''
        nn.init.xavier_uniform_(self.gated_weight_fc1.weight)
        nn.init.xavier_uniform_(self.gated_weight_fc2.weight)
        nn.init.constant_(self.gated_weight_fc1.bias, 0)
        nn.init.constant_(self.gated_weight_fc1.bias, 0)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None, sg_emb=None, p_sg_emb=None,return_att=False):
        """
        prev_h: h_lang output of previous language LSTM
        fc_feats: vector after pooling over K regions, one vector per image 
        xt: embedding of previous word
        att_feats: packed region features 
        p_att_feats: projected [packed region features]
        h_att, c_att: hidden state and cell state of attention LSTM
        h_lang, c_lang: hidden state and cell state of language LSTM
        """
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0])) # the 2nd arg is from previous att_lstm
        
        # attended region features
        # MyAdd: detach 2 kind of features
        b, N, _ = att_feats.shape
        self_att_feats = att_feats[:, N // 2:, :]
        att_feats = att_feats[:, :N // 2, :]
        self_att_masks = att_masks[:, N // 2:]
        att_masks = att_masks[:, :N // 2]
        p_self_att_feats = p_att_feats[:, N // 2:, :]
        p_att_feats = p_att_feats[:, :N // 2, :]
        # print("self_att_feats[shape] = ", self_att_feats.size()) # [500, 36, 1000]
        # print("att_feats[shape] = ", att_feats.size()) # [500, 36, 1000]
        # print("[TopDownCore] self_att_feats size = ", self_att_feats.size())
        # print("[TopDownCore] self_att_masks size = ", self_att_masks.size())
        # print("[TopDownCore] p_self_att_feats size = ", p_self_att_feats.size())
        if return_att:
            self_att, self_att_weight = self.attention(h_att, self_att_feats, p_self_att_feats, self_att_masks, return_att=return_att)
            att, att_weight = self.attention(h_att, att_feats, p_att_feats, att_masks, return_att=return_att)
        else:
            self_att = self.attention(h_att, self_att_feats, p_self_att_feats, self_att_masks)
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        # print("self_att[shape] = ", self_att.size()) #[500, 1000]
        # print("att[shape] = ", att.size()) #[500, 1000]
        # print("self_att_weight[shape] = ", self_att_weight.size()) #[500, 36]
        # print("self_att_weight = ", self_att_weight)
        # print("att_weight[shape] = ", att_weight.size()) # [500, 36]
        # print("att_weight = ", att_weight) #[500, 36]
        # print("self_att_weight == att_weight ? ==> ", self_att_weight == att_weight) # False        # MyAdd: gated fusion of two att_feats
        alpha1 = torch.sigmoid(self.gated_weight_fc1(torch.cat([h_att, att], -1)))
        alpha2 = torch.sigmoid(self.gated_weight_fc2(torch.cat([h_att, self_att], -1)))
        # print("att size = ", att.size()) #[500, 1000]
        # a1:
        # att = att
        # a2:
        # att = self_att
        # a3:
        att = (att * alpha1 + self_att * alpha2) / np.sqrt(2)
        # a4: concat
        # att = torch.cat([att, self_att], -1)
        # print("alpha1 size = ", alpha1.size()) # [500, 1000]

        # print(alpha1)
        # print(alpha2)
        # a1 = torch.mean(alpha1, dim=1, keepdim=True)
        # a2 = torch.mean(alpha2, dim=1, keepdim=True)
        # print("a1 = ", a1.size())
        # print("a2 = ", a2.size())
        # print("att_weight sum = ", torch.sum(att_weight, dim=1)) # sum=1
        # print("self_att_weight sum = ", torch.sum(self_att_weight, dim=1))  # sum=1

        # print("att_weight size = ", att_weight.size()) #[500, 36]
        # fuse_att_weight = a1 * att_weight + a2 * self_att_weight
        # print("fuse_att_weight sum = ", torch.sum(fuse_att_weight, dim=1))  # sum==1+=0.01
        # print("att_weight_fuse[shape] = ", att_weight_fuse.size()) # [500, 36]
        # print("att_weight_fuse = ", att_weight_fuse)

        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1])) # the 2nd arg is from previous lang_lstm

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang])) 
        
        if return_att:
            return output, state, att_weight
        else:
            return output, state

"""
Attention module in attention-based LSTM
"""
class Attention(nn.Module):

    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size 
        self.att_hid_size = opt.att_hid_size 
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None, return_att=False):
        """
        Input hidden state and region features, output the attended visual features
        """
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_masks = att_masks[:, 0:att_size]
        # MyAdd : look att_size(K)
        # print("att_size (K) = ", att_size)
        # print("att_feats size = ", att_feats.size())
        # print("att_mask size = ", att_masks.size())
        
        att_h = self.h2att(h)                        # [batch,512]
        att_h = att_h.unsqueeze(1).expand_as(att)            # [batch, K, 512]
        dot = att + att_h                                   # [batch, K, 512]
        dot = torch.tanh(dot) #F.tanh(dot)                  # [batch, K, 512]
        dot = dot.view(-1, self.att_hid_size)               # [(batch * K), 512]
        dot = self.alpha_net(dot)                           # [(batch * K), 1]
        dot = dot.view(-1, att_size)                        # [batch, K]
        
        weight = F.softmax(dot, dim=1)                             # [batch, K]
        if att_masks is not None:  # necessary since empty box proposals (att_mask) may exist
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # [batch, K, 1000]
        # MyNote: att_res is kind like attented version of representation of an example(image),
        # which then can be map into an word with a rnn neuron
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # [batch, 1000]

        if return_att:
            return att_res, weight
        else:
            return att_res

"""
Captioning model wrapper
"""
class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)
