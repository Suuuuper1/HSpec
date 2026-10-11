"""I4 legal envelope and exact DP for a declared, compact cost surrogate.

The real measured recipe is retained separately in the control plane. Costs
here are an explicit [0..Nsafe] vector for one supported hardware/batch/domain.
The DP is exact for this vector, never claimed exact for actual NPU schedules.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit


def _solve(gains, caps, entries, terminals, costs, budget, base_batch, prev, nxt, back,
           best_entry, best_gain, lengths, selected, expected):
    B,K,W=gains.shape; D=W-1
    best_entry[:,:]=-1; best_gain[:,:]=-1e300
    for b in range(B):
        best_gain[b,0]=0.
        for k in range(K):
            if entries[b,k]<0: continue
            cap=min(int(caps[b,k]),D)
            for l in range(1,cap+1):
                value=gains[b,k,l]
                t=int(terminals[b,k])
                if t>0 and l>=t: value-=gains[b,k,t]-gains[b,k,t-1]
                old=best_entry[b,l]
                if value>best_gain[b,l]+1e-12 or (abs(value-best_gain[b,l])<=1e-12 and (old<0 or entries[b,k]<entries[b,old])):
                    best_gain[b,l]=value;best_entry[b,l]=k
    prev[:]=-1e300;prev[0]=0.
    back[:,:]=-1
    for b in range(B):
        nxt[:]=-1e300
        for pn in range(budget+1):
            if prev[pn]<-1e299: continue
            for l in range(D+1):
                n=pn+l
                if n>budget or best_gain[b,l]<-1e299: continue
                v=prev[pn]+best_gain[b,l]
                if v>nxt[n]+1e-12 or (abs(v-nxt[n])<=1e-12 and (back[b,n]<0 or l<back[b,n])):
                    nxt[n]=v;back[b,n]=l
        prev[:]=nxt
    best_n=-1;ratio=-1e300
    for n in range(budget+1):
        if prev[n]<-1e299 or not math.isfinite(costs[n]) or costs[n]<=0: continue
        r=(base_batch+prev[n])/costs[n]
        if r>ratio+1e-12:ratio=r;best_n=n
    lengths[:]=0;selected[:]=-1;expected[:]=0.
    if best_n<0:return -1,0.,0.
    n=best_n
    for b in range(B-1,-1,-1):
        l=int(back[b,n]);lengths[b]=l
        if l>0:
            k=int(best_entry[b,l]);selected[b]=k;expected[b]=best_gain[b,l]
        n-=l
    return best_n,ratio,costs[best_n]

solve_kernel=njit(cache=True,nogil=True)(_solve)


def _menu(gains,caps,entries,terminals,costs,budget,base_batch,selected,lengths,expected,tmp_slot,tmp_len,tmp_gain):
    B,K,W=gains.shape;limit=min(15,budget//max(B,1))
    ratio=-1e300;best_n=-1;selected[:]=-1;lengths[:]=0;expected[:]=0.
    # Uniform resource envelope with the safety endpoint included. This is a
    # restricted schedule menu, not an exact solution of the full DP problem.
    for L in (0,1,2,4,8,15,limit):
        if L>limit:continue
        n=0;total=0.
        for b in range(B):
            bk=-1;bl=0;bg=0.
            for k in range(K):
                if entries[b,k]<0:continue
                l=min(L,int(caps[b,k]));v=gains[b,k,l];t=int(terminals[b,k])
                if t>0 and l>=t:v-=gains[b,k,t]-gains[b,k,t-1]
                if v>bg+1e-12 or (v>0 and abs(v-bg)<=1e-12 and (l<bl or (l==bl and (bk<0 or entries[b,k]<entries[b,bk])))):
                    bk=k;bl=l;bg=v
            tmp_slot[b]=bk;tmp_len[b]=bl;tmp_gain[b]=bg;n+=bl;total+=bg
        if not math.isfinite(costs[n]) or costs[n]<=0:continue
        r=(base_batch+total)/costs[n]
        if r>ratio+1e-12 or (abs(r-ratio)<=1e-12 and n<best_n):
            ratio=r;best_n=n;selected[:]=tmp_slot;lengths[:]=tmp_len;expected[:]=tmp_gain
    return best_n,ratio,costs[best_n] if best_n>=0 else 0.

menu_kernel=njit(cache=True,nogil=True)(_menu)


class BudgetSolver:
    def __init__(self,max_batch=64,max_depth=15,budget=384):
        if not 0<=budget<=384: raise ValueError('unsafe budget')
        self.budget=budget;self.max_batch=max_batch;self.max_depth=max_depth
        self.prev=np.empty(budget+1);self.nxt=np.empty(budget+1)
        self.back=np.empty((max_batch,budget+1),np.int16)
        self.best_entry=np.empty((max_batch,max_depth+1),np.int16)
        self.best_gain=np.empty((max_batch,max_depth+1))
        self.lengths=np.zeros(max_batch,np.int16);self.selected=np.full(max_batch,-1,np.int16)
        self.expected=np.zeros(max_batch)
        self.tmp_slot=np.empty(max_batch,np.int16);self.tmp_len=np.empty(max_batch,np.int16);self.tmp_gain=np.empty(max_batch)

    @property
    def scratch_bytes(self): return sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))

    def solve(self,gains,caps,entries,terminals,costs,compiled=True,base_batch=None):
        B,K,W=gains.shape
        if B>self.max_batch or W!=self.max_depth+1 or caps.shape!=(B,K) or entries.shape!=(B,K) or terminals.shape!=(B,K):
            raise ValueError('budget ABI shape mismatch')
        if costs.shape!=(self.budget+1,):raise ValueError('cost vector ABI mismatch')
        if not np.isfinite(gains).all():raise ValueError('nonfinite gain must use row fallback before solving')
        if np.any(terminals<0) or np.any(terminals>self.max_depth):raise ValueError('invalid terminal depth')
        fn=solve_kernel if compiled else _solve
        n,ratio,cost=fn(gains,caps,entries,terminals,costs,self.budget,B if base_batch is None else base_batch,self.prev,self.nxt,self.back[:B],
            self.best_entry[:B],self.best_gain[:B],self.lengths[:B],self.selected[:B],self.expected[:B])
        return {'status':'PASS' if n>=0 else 'UNKNOWN_COST','n':n,'ratio':ratio,'cost_ms':cost,
                'slot':self.selected[:B], 'length':self.lengths[:B], 'gain':self.expected[:B]}

    def solve_menu(self,gains,caps,entries,terminals,costs,compiled=True,base_batch=None):
        B=gains.shape[0]
        fn=menu_kernel if compiled else _menu
        n,ratio,cost=fn(gains,caps,entries,terminals,costs,self.budget,B if base_batch is None else base_batch,self.selected[:B],self.lengths[:B],self.expected[:B],
            self.tmp_slot[:B],self.tmp_len[:B],self.tmp_gain[:B])
        return {'status':'PASS' if n>=0 else 'UNKNOWN_COST','n':n,'ratio':ratio,'cost_ms':cost,
            'slot':self.selected[:B],'length':self.lengths[:B],'gain':self.expected[:B]}

    def warmup(self):
        self.solve_menu(np.zeros((1,8,16)),np.zeros((1,8),np.int16),np.zeros((1,8),np.int32),
            np.zeros((1,8),np.int16),np.ones(self.budget+1))
        self.solve(np.zeros((1,8,16)),np.zeros((1,8),np.int16),np.zeros((1,8),np.int32),
            np.zeros((1,8),np.int16),np.ones(self.budget+1))
