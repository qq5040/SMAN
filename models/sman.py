import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from models.lab_unit import LAB
from models.abl_unit import ABL

from models import sampling
from models import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

class SMAN(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    def __init__(self, config: BertConfig, cls_token: int, sep_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, encoder_layers: int = 2,
                 encoder_heads: int = 8, dropout=0.1, max_pairs: int = 100, device: str = 'cuda:0'):
        super(SMAN, self).__init__(config)

        self.device = device
        self.hidden = config.hidden_size
        self.en_layer = encoder_layers
        self.en_head = encoder_heads
        self.d_model = config.hidden_size

        # BERT model
        self.bert = BertModel(config)

        self.entity_span_emb_tr = LAB(num_layers=self.en_layer, d_model=self.d_model, n_head=self.en_head, dropout=dropout,
                                      after_norm=True, scale=True, dropout_attn=None, pos_embed=None, device=device)

        self.entity_span_linear = nn.Linear(self.d_model, self.d_model-size_embedding)

        self.entity_span_classifier = nn.Linear(config.hidden_size*2, entity_types)

        self.entity_ctx_emb_tr = LAB(num_layers=self.en_layer, d_model=self.d_model, n_head=self.en_head, dropout=dropout,
                                     after_norm=True, scale=True, dropout_attn=None, pos_embed=None, device=device)

        self.entity_ctx_classifier = nn.Linear(self.d_model, entity_types)

        #rel
        self.rel_lstm = nn.LSTM(config.hidden_size*2, config.hidden_size, 2, batch_first=True, bidirectional=True, dropout=dropout)

        self.rel_tr = ABL(num_layers=self.en_layer, d_model=self.d_model, n_head=self.en_head, dropout=dropout,
                          after_norm=True, scale=True, dropout_attn=None, pos_embed=None, device=device)

        self.rel_span_classifier = nn.Linear(2*config.hidden_size, config.hidden_size)

        self.rel_tpye_classifier = nn.Linear(config.hidden_size*2, config.hidden_size)

        self.rel_classifier = nn.Linear(config.hidden_size*2, relation_types)

        self.softmax = nn.Softmax(dim=2)

        self.unlinear = nn.Sigmoid()

        self.size_embeddings = nn.Embedding(50, size_embedding)
        self.ner_tpye_embeddings = nn.Embedding(entity_types, config.hidden_size)

        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._sep_token = sep_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor, eup_masks: torch.tensor, edown_masks: torch.tensor, entity_types: torch.tensor,
                       entity_sizes: torch.tensor, entity_start: torch.tensor, entity_end: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor, rup_masks: torch.tensor, rdown_masks: torch.tensor):

        torch.cuda.empty_cache()

        context_mask = context_mask.float()
        context_len = context_mask.sum(dim=-1).int()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes [b,span_num, size_embedding]
        entity_clf = torch.zeros([batch_size, entity_start.shape[1], self._entity_types]).to(self.device)
        entity_spans_pool = torch.zeros([batch_size, entity_start.shape[1], self.hidden]).to(self.device)

        # obtain entitiy logits
        # chunk processing to reduce memory usage
        for i in range(0, entity_start.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_entity_clf, chunk_entity_spans_pool = self._classify_entities(encodings, h, context_len, eup_masks,
                                                                                edown_masks,
                                                                                entity_start, entity_end,
                                                                                size_embeddings, i)

            entity_clf[:, i:i + self._max_pairs, :] = chunk_entity_clf
            entity_spans_pool[:, i:i + self._max_pairs, :] = chunk_entity_spans_pool

        # classify relations---------------------------------------------------------------------------------------
        rel_masks = rel_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)

        rup_masks = rup_masks.float().unsqueeze(-1)
        rdown_masks = rdown_masks.float().unsqueeze(-1)

        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(encodings, h, entity_spans_pool, size_embeddings, relations, rel_masks, rup_masks, rdown_masks, h_large, i, entity_types)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        torch.cuda.empty_cache()

        return entity_clf, rel_clf

    def _forward_eval(self, encodings: torch.tensor, context_mask: torch.tensor, entity_masks: torch.tensor, eup_masks: torch.tensor, edown_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_start: torch.tensor, entity_end: torch.tensor, entity_spans: torch.tensor = None,
                      entity_sample_mask: torch.tensor = None):

        torch.cuda.empty_cache()

        # get contextualized token embeddings from last transformer layer
        context_mask = context_mask.float()
        context_len = context_mask.sum(dim=-1).int()
        h = self.bert(input_ids=encodings, attention_mask=context_mask)[0]

        entity_masks = entity_masks.float()
        batch_size = encodings.shape[0]
        ctx_size = context_mask.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf = torch.zeros([batch_size, entity_start.shape[1], self._entity_types]).to(self.device)
        entity_spans_pool = torch.zeros([batch_size, entity_start.shape[1], self.hidden]).to(self.device)

        # obtain entitiy logits
        # chunk processing to reduce memory usage
        for i in range(0, entity_start.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_entity_clf, chunk_entity_spans_pool = self._classify_entities(encodings, h, context_len, eup_masks, edown_masks,
                                                                    entity_start, entity_end, size_embeddings, i)

            entity_clf[:, i:i + self._max_pairs, :] = chunk_entity_clf
            entity_spans_pool[:, i:i + self._max_pairs, :] = chunk_entity_spans_pool

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rup_masks, rdown_masks, rel_sample_masks, entity_types = self._filter_spans(entity_clf, entity_spans, entity_sample_mask, ctx_size, context_len)

        rel_masks = rel_masks.float()
        rup_masks = rup_masks.float()
        rdown_masks = rdown_masks.float()

        rel_sample_masks = rel_sample_masks.float()
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(encodings, h, entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, rup_masks, rdown_masks, h_large, i, entity_types)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        torch.cuda.empty_cache()

        return entity_clf, rel_clf, relations


    def _classify_relations(self, encodings, bert_h, entity_spans, size_embeddings, relations, rel_masks, rup_masks, rdown_masks, h, chunk_start, entity_types):

        torch.cuda.empty_cache()

        batch_size = relations.shape[0]
        rel_num = relations.shape[1]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            rup_masks = rup_masks[:, chunk_start:chunk_start + self._max_pairs]
            rdown_masks = rdown_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]
            rel_num = relations.shape[1]

        # get cls token as candidate context representation
        entity_globe = get_token(bert_h, encodings, self._cls_token)  # batch, hidden
        entity_globe = entity_globe.unsqueeze(1).repeat(1, rel_num, 1)  # [batch*rel_num, hidden]
        entity_globe = entity_globe.view(batch_size*rel_num, self.hidden)

        ner_type = util.batch_index(entity_types, relations)
        entity_types_embeddings = self.ner_tpye_embeddings(ner_type)     #batch, rel_num, 2, hidden
        entity_types_embeddings = entity_types_embeddings.view(batch_size*rel_num, 2, -1)    #batch*rel_num, 2, hidden
        entity_types_embeddings = entity_types_embeddings.transpose(0, 1)   #2, batch*rel_num, hidden

        rel_tpye_repr = self.rel_tpye_classifier(torch.cat([entity_types_embeddings[0], entity_types_embeddings[1]], dim=-1))
        rel_tpye_repr = rel_tpye_repr.view(batch_size, rel_num, -1)

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size*rel_num, 2, -1)
        entity_pairs = entity_pairs.transpose(0, 1)   #2, batch*rel_num, hidden

        rel_ctx = rel_masks * h
        rel_ctx = rel_ctx.max(dim=2)[0]         #batch, rel_num, hidden
        rel_ctx = rel_ctx.view(batch_size*rel_num, -1)   #batch*rel_num, hidden

        rup_ctx = rup_masks * h
        rup_ctx = rup_ctx.max(dim=2)[0]
        rup_ctx = rup_ctx.view(batch_size * rel_num, -1)

        rdown_ctx = rdown_masks * h
        rdown_ctx = rdown_ctx.max(dim=2)[0]
        rdown_ctx = rdown_ctx.view(batch_size * rel_num, -1)

        # get rel_input to tr
        # batch*rel_num, 6, hidden
        rel_input1 = torch.cat([rup_ctx.unsqueeze(1), entity_pairs[0].unsqueeze(1), rel_ctx.unsqueeze(1), entity_pairs[1].unsqueeze(1), rdown_ctx.unsqueeze(1)], dim=1)

        none_tpyes = torch.zeros([batch_size*rel_num, 1]).to(self.device)
        none_embeddings = self.ner_tpye_embeddings(none_tpyes.long())   #batch*rel_num, 1, hidden
        rel_input2 = torch.cat([none_embeddings, entity_types_embeddings[0].unsqueeze(1), none_embeddings, entity_types_embeddings[1].unsqueeze(1), none_embeddings], dim=1)

        # batch*rel_num, 5, 2*hidden
        rel_input3 = torch.cat([rel_input1, rel_input2], dim=-1)

        rel_input1 = self.dropout(rel_input1)
        rel_input2 = self.dropout(rel_input2)
        rel_input3 = self.dropout(rel_input3)


        out, _ = self.rel_lstm(rel_input3)

        _, hc = self.rel_tr(rel_input1, rel_input2, out)     # [batch*rel_num, 5, hidden]

        chunk_rel_logits = torch.cat([hc[0][0], hc[0][1]], dim=-1).view(batch_size, rel_num, -1)     # batch, rel_num, hidden
        chunk_rel_logits = self.rel_span_classifier(chunk_rel_logits)

        chunk_rel_logits = torch.cat([rel_tpye_repr, chunk_rel_logits], dim=-1)

        chunk_rel_logits = self.rel_classifier(chunk_rel_logits)

        torch.cuda.empty_cache()

        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_mask, ctx_size, context_leng):

        torch.cuda.empty_cache()

        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_mask.long()  # get entity type (including none)
        batch_relations = []

        batch_rel_masks = []
        batch_rup_masks = []
        batch_rdown_masks = []

        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []

            rel_masks = []
            rup_masks = []
            rdown_masks = []

            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)

            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        mask1, mask2, mask3 = sampling.create_rel_mask(s1, s2, ctx_size, context_leng)
                        rel_masks.append(mask1)
                        rup_masks.append(mask2)
                        rdown_masks.append(mask3)
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rup_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rdown_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rup_masks.append(torch.stack(rup_masks))
                batch_rdown_masks.append(torch.stack(rdown_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device).unsqueeze(-1)
        batch_rup_masks = util.padded_stack(batch_rup_masks).to(device).unsqueeze(-1)
        batch_rdown_masks = util.padded_stack(batch_rdown_masks).to(device).unsqueeze(-1)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device).unsqueeze(-1)

        torch.cuda.empty_cache()

        return batch_relations, batch_rel_masks, batch_rup_masks, batch_rdown_masks, batch_rel_sample_masks, entity_logits_max


    def _classify_entities(self, encodings, h, context_len, eup_masks, edown_masks, entity_start, entity_end, size_embeddings, chunk_start):

        torch.cuda.empty_cache()

        batch_size = encodings.shape[0]
        hidden_size = h.shape[2]
        span_num = eup_masks.shape[1]

        # ----------------------------
        # create chunks if necessary
        if eup_masks.shape[1] > self._max_pairs:
            eup_masks = eup_masks[:, chunk_start:chunk_start + self._max_pairs]
            edown_masks = edown_masks[:, chunk_start:chunk_start + self._max_pairs]
            entity_start = entity_start[:, chunk_start:chunk_start + self._max_pairs]
            entity_end = entity_end[:, chunk_start:chunk_start + self._max_pairs]
            size_embeddings = size_embeddings[:, chunk_start:chunk_start + self._max_pairs]
            span_num = eup_masks.shape[1]
        # ----------------------------------------

        eup_masks = eup_masks.float()
        edown_masks = edown_masks.float()

        # max pool entity candidate spans
        eup_spans_pool = eup_masks.unsqueeze(-1) * h.unsqueeze(1).repeat(1, eup_masks.shape[1], 1, 1)
        eup_spans_pool = eup_spans_pool.max(dim=2)[0]  # [batch, span_num, hidden]

        edown_spans_pool = edown_masks.unsqueeze(-1) * h.unsqueeze(1).repeat(1, edown_masks.shape[1], 1, 1)
        edown_spans_pool = edown_spans_pool.max(dim=2)[0]  # [batch, span_num, hidden]

        entity_sep = get_token(h, encodings, self._sep_token)  # [batch, hidden]
        pad = torch.zeros([batch_size, hidden_size]).to(self.device)  # [batch, hidden]
        # entity_pad = get_token(h, encodings, 0)

       # -----------------------------------------------------------------------------------------------------
        entity_input_length = []
        for b in range(batch_size):
            for s in range(span_num):
                start = entity_start[b][s]
                end = entity_end[b][s]
                entity_input_length.append(end - start + 2)

        entity_input_length = torch.stack(entity_input_length, dim=0)
        max_entity_length = entity_input_length.max()

        entity_input = []
        for b in range(batch_size):
            for s in range(span_num):
                start = entity_start[b][s]
                end = entity_end[b][s]
                span = h[b][start:end]
                input = torch.cat([eup_spans_pool[b][s].unsqueeze(0), span, edown_spans_pool[b][s].unsqueeze(0)], dim=0)    #去掉全局信息，仅用上下文

                fill_length = max_entity_length - input.shape[0]
                if fill_length != 0:
                    padd = pad[b].unsqueeze(0).repeat(fill_length, 1)
                    input = torch.cat([input, padd], dim=0)
                entity_input.append(input)

        entity_input = torch.stack(entity_input)  # [b*span_num, maxleng, hidden]

        entity_input_mask = torch.zeros([entity_input.shape[0], entity_input.shape[1]], dtype=torch.float).to(self.device)
        for i, l in enumerate(entity_input_length):
            entity_input_mask[i, :l] = 1.0

        entity_input_length, entity_input_length_id = entity_input_length.sort(descending=True)
        entity_input = entity_input[entity_input_length_id]
        entity_input_mask = entity_input_mask[entity_input_length_id]

        _, raw_id = entity_input_length_id.sort()

        entity_input = self.dropout(entity_input)

        # embedding span
        _, hc = self.entity_span_emb_tr(entity_input, padding_mask=entity_input_mask, lstm_len=entity_input_length)   # batch_size*num_span x max_len x d_model; attn: batch_size x max_len x max_len

        span_emb = torch.cat([hc[0][0], hc[0][1]], dim=-1)[raw_id].view(batch_size, span_num, hidden_size)     # batch, num_span, hidden
        span_emb = self.entity_span_linear(span_emb)
        span_emb = torch.cat([span_emb, size_embeddings], dim=2)     # batch, num_span, hidden

        # --------------------------------------------------------------------------------------------------------------------
        ctx_input_length = []
        sep_id = []
        for b in range(batch_size):
            for s in range(span_num):
                start = entity_start[b][s]
                end = entity_end[b][s]
                ctx_input_length.append(start + context_len[b] - end + 1)
                sep_id.append(start)

        ctx_input_length = torch.stack(ctx_input_length, dim=0)
        sep_id = torch.stack(sep_id, dim=0)
        max_ctx_length = ctx_input_length.max()

        ctx_input = []
        for b in range(batch_size):
            for s in range(span_num):
                start = entity_start[b][s]
                end = entity_end[b][s]
                endd = context_len[b]
                up = h[b][:start]
                down = h[b][end:endd]
                input = torch.cat([up, entity_sep[b].unsqueeze(0), down], dim=0)

                fill_length = max_ctx_length - input.shape[0]
                if fill_length != 0:
                    padd = pad[b].unsqueeze(0).repeat(fill_length, 1)
                    input = torch.cat([input, padd], dim=0)
                ctx_input.append(input)

        ctx_input = torch.stack(ctx_input)  # [b*span_num, maxleng, hidden]

        ctx_input_mask = torch.zeros([ctx_input.shape[0], ctx_input.shape[1]], dtype=torch.float).to(self.device)
        for i, l in enumerate(ctx_input_length):
            ctx_input_mask[i, :l] = 1.0

        ctx_input_length, ctx_input_length_id = ctx_input_length.sort(descending=True)
        ctx_input = ctx_input[ctx_input_length_id]
        ctx_input_mask = ctx_input_mask[ctx_input_length_id]

        _, raw_id = ctx_input_length_id.sort()

        ctx_input = self.dropout(ctx_input)

        # embedding span
        ctx_emb = []
        out, hc = self.entity_ctx_emb_tr(ctx_input, padding_mask=ctx_input_mask, lstm_len=ctx_input_length)   # batch_size x max_len x d_model; attn: batch_size x max_len x max_len
        out = out[raw_id]

        for b in range(batch_size*span_num):
            ctx_emb.append(out[b][sep_id[b]])
        ctx_emb = torch.stack(ctx_emb, dim=0).view(batch_size, span_num, hidden_size)

        ctx_emb = torch.cat([span_emb, ctx_emb], dim=-1)
        entity_clf = self.entity_span_classifier(ctx_emb)

        torch.cuda.empty_cache()

        return entity_clf, span_emb

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'sman': SMAN,
}

def get_model(name):
    return _MODELS[name]
