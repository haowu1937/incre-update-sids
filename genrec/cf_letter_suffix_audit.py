"""Vectorized read-only audit of the existing persistent suffix ledger."""
import numpy as np


class SuffixAudit:
    def __init__(self,previous,capacity):
        previous=np.asarray(previous,dtype=np.int64);self.capacity=capacity;self.n=len(previous)
        if previous.shape!=(self.n,4):raise ValueError('Invalid old full SIDs')
        self.raw=previous-np.arange(4)*capacity
        if np.any(self.raw<0) or np.any(self.raw>=capacity) or len(np.unique(previous,axis=0))!=self.n:raise ValueError('Invalid old full SIDs')
        prefix=self.key(self.raw[:,:3]);self.keys,inverse=np.unique(prefix,return_inverse=True)
        self.width=np.zeros(len(self.keys),dtype=np.int64);np.maximum.at(self.width,inverse,self.raw[:,3]+1)

    def key(self,codes):return (codes[:,0]*self.capacity+codes[:,1])*self.capacity+codes[:,2]

    def reserved(self,keys):
        positions=np.searchsorted(self.keys,keys);valid=positions<len(self.keys);safe=np.minimum(positions,len(self.keys)-1)
        return np.where(valid & (self.keys[safe]==keys),self.width[safe],0)

    def __call__(self,codes):
        codes=np.asarray(codes,dtype=np.int64)
        if codes.shape!=(self.n,3) or np.any(codes<0) or np.any(codes>=self.capacity):raise ValueError('Invalid semantic codes')
        prefix=self.key(codes);movers=np.any(codes!=self.raw[:,:3],axis=1);suffix=self.raw[:,3].copy()
        values=prefix[movers];order=np.argsort(values,kind='stable');sorted_values=values[order]
        if len(values):
            starts=np.r_[0,np.flatnonzero(np.diff(sorted_values))+1]
            sorted_ranks=np.arange(len(values))-np.repeat(starts,np.diff(np.r_[starts,len(values)]))
            ranks=np.empty(len(values),dtype=np.int64);ranks[order]=sorted_ranks
            suffix[movers]=self.reserved(values)+ranks
        keys,first,inverse,counts=np.unique(prefix,return_index=True,return_inverse=True,return_counts=True)
        required=np.zeros(len(keys),dtype=np.int64);np.maximum.at(required,inverse,suffix+1)
        bad=np.flatnonzero(required>self.capacity);bad=bad[np.argsort(first[bad])]
        overflow=[dict(semantic_code=codes[first[i]].tolist(),current_items=int(counts[i]),reserved_width=int(self.reserved(keys[i:i+1])[0]),required_capacity=int(required[i])) for i in bad[:20]]
        feasible=len(bad)==0
        stats=dict(items=self.n,unique_semantic_codes=len(keys),maximum_current_collision_bucket=int(counts.max()),maximum_required_suffix_capacity=int(required.max()),suffix_capacity=self.capacity,full_sid_allocation_feasible=feasible,overflow_buckets=overflow,overflow_bucket_count=len(bad))
        mapping=np.column_stack([codes,suffix])+np.arange(4)*self.capacity if feasible else None
        return mapping,stats
