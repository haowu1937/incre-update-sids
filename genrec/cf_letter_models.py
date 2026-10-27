"""Task-scoped CF-LETTER models, adapted from semantic-id-alignment (MIT).

Reference: 133692ecd4ebf27f4d925a261952f8f6dec5d41c.
Copyright (c) 2026 Iskander Bagautdinov; see third_party/semantic_id_alignment_LICENSE.
Adaptations: causal explicit-query supervision, safe BOS carrier, native GPT2
initialization and explicit-position cached decoding. No pretrained downloads.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SASRec(nn.Module):
    def __init__(self, items, history, config):
        super().__init__();dim=config['dim'];self.history=history
        self.items=nn.Embedding(items+1,dim,padding_idx=0)
        self.positions=nn.Embedding(history+1,dim)
        self.bos=nn.Parameter(torch.empty(dim))
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=config['heads'],dim_feedforward=config['ffn'],
            dropout=config['dropout'],activation='relu',batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,num_layers=config['layers'],enable_nested_tensor=False)
        for name,value in self.named_parameters():
            if 'norm' in name and name.endswith('weight'):nn.init.ones_(value)
            elif name.endswith('bias'):nn.init.zeros_(value)
            else:nn.init.trunc_normal_(value,std=.02,a=-.04,b=.04)
        with torch.no_grad():self.items.weight[0].zero_()

    def forward(self, history):
        b,h=history.shape
        embedded=torch.cat([self.bos.expand(b,1,-1),self.items(history)],dim=1)
        mask=torch.cat([torch.ones(b,1,dtype=torch.bool,device=history.device),history!=0],dim=1)
        embedded=embedded+self.positions(torch.arange(h+1,device=history.device))[None]
        causal=torch.ones(h+1,h+1,dtype=torch.bool,device=history.device).triu(1)
        return self.encoder(embedded,mask=causal,src_key_padding_mask=~mask)[:,-1]

    def loss(self, history, positive, negative):
        value=self(history)
        a=(value*self.items(positive)).sum(-1);b=(value*self.items(negative)).sum(-1)
        return F.binary_cross_entropy_with_logits(a,torch.ones_like(a))+F.binary_cross_entropy_with_logits(b,torch.zeros_like(b))


class CollaborativeRQ(nn.Module):
    def __init__(self, features, config):
        super().__init__();self.register_buffer('features',features.detach().clone())
        self.books=nn.ParameterList([nn.Parameter(torch.zeros(config['capacity'],config['dim'])) for _ in range(config['books'])])
        self.config=dict(config)

    @staticmethod
    def nearest(x,book):
        distance=x.square().sum(-1,keepdim=True)+book.square().sum(-1)[None]-2*x@book.T
        return distance.argmin(-1)

    def forward(self, x):
        remainder=x;restored=torch.zeros_like(x);qloss=x.new_zeros(());codes=[]
        for book in self.books:
            index=self.nearest(remainder,book);codes.append(index);selected=book[index]
            qloss=qloss+self.config['commitment']*F.mse_loss(remainder,selected.detach())+F.mse_loss(selected,remainder.detach())
            straight=remainder+(selected-remainder).detach()
            restored=restored+straight;remainder=remainder-straight
        recon=F.mse_loss(restored,x)
        return recon+self.config['outer_weight']*qloss, recon, qloss, torch.stack(codes,dim=1)

    @torch.no_grad()
    def initialize(self):
        for level,book in enumerate(self.books):
            rows=torch.randperm(len(self.features),device=self.features.device)[:len(book)]
            remainder=self.features[rows].clone()
            for previous in self.books[:level]:remainder-=previous[self.nearest(remainder,previous)]
            book.copy_(remainder)

    @torch.no_grad()
    def assignments(self,chunk=1024):
        values=[]
        for x in self.features.split(chunk):
            r=x;codes=[]
            for book in self.books:
                idx=self.nearest(r,book);codes.append(idx);r=r-book[idx]
            values.append(torch.stack(codes,dim=1))
        return torch.cat(values)

    @torch.no_grad()
    def repair(self):
        counts=[]
        for level,book in enumerate(self.books):
            used=torch.zeros(len(book),device=book.device,dtype=torch.long)
            for batch in self.features.split(1024):
                r=batch.clone()
                for prev in self.books[:level]:r-=prev[self.nearest(r,prev)]
                used+=torch.bincount(self.nearest(r,book),minlength=len(book))
            dead=torch.where(used==0)[0];counts.append(len(dead))
            if len(dead):
                r=self.features[torch.randperm(len(self.features),device=book.device)[:len(dead)]].clone()
                for prev in self.books[:level]:r-=prev[self.nearest(r,prev)]
                book[dead]=r
        return counts

    @torch.no_grad()
    def audit(self):
        codes=self.assignments();restored=sum(book[codes[:,i]] for i,book in enumerate(self.books))
        reconstruction=F.mse_loss(restored,self.features)
        native=quant=0.
        for x in self.features.split(1024):
            loss,_,q,_=self(x);native+=float(loss)*len(x);quant+=float(q)*len(x)
        return dict(total_loss=native/len(self.features),quantization_loss=quant/len(self.features),
            reconstruction_mse=float(reconstruction),codebooks_used=[int(codes[:,i].unique().numel()) for i in range(len(self.books))]),codes


def legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache,'to_legacy_cache') else cache


def select_cache(cache,index):
    return tuple(tuple(t.index_select(0,index) for t in layer) for layer in legacy(cache))


class GenerativeRetriever(nn.Module):
    def __init__(self, history, capacity, config):
        super().__init__()
        from transformers import GPT2Config,GPT2LMHeadModel
        self.capacity=capacity;self.vocab=4*capacity+config['user_buckets']+10
        self.pad=self.vocab-1;self.prefix_length=1+4*history
        c=GPT2Config(vocab_size=self.vocab,n_positions=self.prefix_length+10,
            n_embd=config['d_model'],n_layer=config['layers'],n_head=config['heads'],n_inner=config['ffn'],
            activation_function='relu',resid_pdrop=config['dropout'],embd_pdrop=config['dropout'],attn_pdrop=config['dropout'],
            layer_norm_epsilon=1e-6,initializer_range=.02,tie_word_embeddings=False,pad_token_id=self.pad,
            bos_token_id=self.vocab-3,eos_token_id=self.vocab-2)
        c._attn_implementation='eager'
        self.model=GPT2LMHeadModel(c)

    def teacher_logits(self, prefix, mask, targets, inputs_embeds=None):
        ids=torch.cat([prefix,targets],dim=1)
        attention=torch.cat([mask,torch.ones_like(targets,dtype=torch.bool)],dim=1)
        args=dict(attention_mask=attention,position_ids=torch.arange(ids.shape[1],device=ids.device)[None].expand(len(ids),-1),use_cache=False)
        if inputs_embeds is None:args['input_ids']=ids
        else:args['inputs_embeds']=inputs_embeds
        logits=self.model(**args).logits
        start=prefix.shape[1]-1
        return logits[:,start:start+4]

    def nll(self,prefix,mask,targets,inputs_embeds=None):
        logits=self.teacher_logits(prefix,mask,targets,inputs_embeds)
        return F.cross_entropy(logits.transpose(1,2),targets,reduction='none').sum(-1)

    @torch.no_grad()
    def beam(self,prefix,mask,trie,width):
        """Full-vocabulary log probabilities with catalog constraints; explicit KV positions."""
        b,t=prefix.shape;device=prefix.device
        out=self.model(input_ids=prefix,attention_mask=mask,
            position_ids=torch.arange(t,device=device)[None].expand(b,-1),use_cache=True)
        logits=out.logits[:,-1];cache=out.past_key_values
        scores=torch.zeros(b,1,device=device);seq=torch.empty(b,1,0,dtype=torch.long,device=device)
        attention=mask;current=1
        for level in range(4):
            logp=F.log_softmax(logits,dim=-1).reshape(b,current,-1)
            allowed=torch.zeros_like(logp,dtype=torch.bool)
            seqs=seq.cpu().tolist()
            placeholders=[]
            for i in range(b):
                for j in range(current):
                    options=trie.get(tuple(seqs[i][j]),())
                    if not options:raise ValueError('Invalid active beam prefix')
                    allowed[i,j,list(options)]=True;placeholders.append(options[0])
            logp=logp.masked_fill(~allowed,-torch.inf)
            candidate=(logp+scores[:,:,None]).reshape(b,-1)
            scores,index=candidate.topk(width,dim=-1)
            parent=index//self.vocab;token=index%self.vocab
            valid=torch.isfinite(scores)
            # Keep inactive beam prefixes legal without giving them finite scores.
            fallback=torch.tensor(placeholders,device=device).reshape(b,current).gather(1,parent)
            token=torch.where(valid,token,fallback)
            seq=torch.cat([seq.gather(1,parent[:,:,None].expand(-1,-1,level)),token[:,:,None]],dim=-1)
            if level<3:
                flat=(parent+torch.arange(b,device=device)[:,None]*current).reshape(-1)
                cache=select_cache(cache,flat)
                attention=attention.index_select(0,flat)
                attention=torch.cat([attention,torch.ones(b*width,1,dtype=torch.bool,device=device)],dim=1)
                out=self.model(input_ids=token.reshape(-1,1),attention_mask=attention,
                    position_ids=torch.full((b*width,1),t+level,dtype=torch.long,device=device),
                    past_key_values=cache,use_cache=True)
                logits=out.logits[:,-1];cache=out.past_key_values;current=width
        if not torch.isfinite(scores).all():raise ValueError('Catalog has insufficient finite final beams')
        return seq,scores


def make_trie(mapping):
    trie={}
    for row in mapping:
        for pos in range(4):trie.setdefault(tuple(row[:pos]),set()).add(int(row[pos]))
    return {key:tuple(sorted(v)) for key,v in trie.items()}
