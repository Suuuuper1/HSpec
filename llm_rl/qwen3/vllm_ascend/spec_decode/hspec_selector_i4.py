"""Opt-in I4 bridge. Default production profiles do not enable this module.

Initialization/rollout-boundary does artifact I/O; prepare()/finish() never do.
TP leader calls prepare; existing proposer broadcasts the final draft IDs.
"""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
from collections import Counter
import numpy as np
try:
    from .hspec_selector_prefix import PrefixScorer
    from .hspec_selector_budget import BudgetSolver
    from .hspec_selector_survival import extract_utility_features_numba
except ImportError:  # pure CPU control-plane loader, same source and kernel
    from hspec_selector_prefix import PrefixScorer
    from hspec_selector_budget import BudgetSolver
    from hspec_selector_survival import extract_utility_features_numba

from numba import njit

@njit(cache=True,nogil=True)
def interpolate_cost(surface,overhead,selector,anchors,batch,context,mode,out):
    """A declared B/depth/context surrogate, never an exact bucket lookup.

    Natural overhead and CPU selector deltas are additive MODEL terms; no
    overlapping measured host/device spans are added. All 1..64 batches are
    supported without fictitious requests or target rows.
    """
    hi=0
    while hi<len(anchors)-1 and anchors[hi]<batch:hi+=1
    lo=max(0,hi-1) if anchors[hi]!=batch else hi
    w=0. if lo==hi else (batch-anchors[lo])/(anchors[hi]-anchors[lo])
    cw=min((17408.-64.)/(8192.-64.),max(0.,(context-64.)/(8192.-64.)))
    for n in range(len(out)):
        depth=n/batch
        if depth>15.:out[n]=np.nan;continue
        dl=int(depth);du=min(15,dl+1);dw=depth-dl
        v=0.;ok=True
        for b,bw in ((lo,1.-w),(hi,w)):
            if bw==0.:continue
            for c,cweight in ((0,1.-cw),(1,cw)):
                if cweight==0.:continue
                # N=0 and a nonempty verification use different sampler
                # paths. Never smooth the AR point into the positive branch.
                a=surface[b,c,1 if n>0 and dl==0 else dl];z=surface[b,c,du]
                if not math.isfinite(a) or not math.isfinite(z) or min(a,z)<=0:ok=False
                v+=bw*cweight*((1.-dw)*a+dw*z)
        extra=(1.-cw)*overhead[batch,0]+cw*overhead[batch,1]+(selector[batch,mode] if mode>=0 else 0.)
        out[n]=v+extra if ok and math.isfinite(extra) and v+extra>0 else np.nan

@njit(cache=True,nogil=True)
def mix_cost_context(short,long,context,delta,out):
    w=min((17408.-64.)/(8192.-64.),max(0.,(context-64.)/(8192.-64.)))
    for n in range(len(out)):
        v=(1.-w)*short[n]+w*long[n]+delta
        out[n]=v if math.isfinite(v) and v>0 else np.nan

@njit(cache=True,nogil=True)
def pack_row(indices, er, eo, buffer, ro, rl, bias, hits, wnd, min_wnd, max_wnd, horizon,
             enabled, safe, mid, midcap, farcap, maxdraft, basepos, cap, stops,
             tokens, available, caps, terminal):
    for k in range(len(indices)):
        e=int(indices[k]);roll=int(er[e]);offset=int(eo[e]);start=int(ro[roll])+offset
        n=min(15,int(rl[roll])-offset);available[k]=n;tokens[k,:]=0
        for d in range(n):tokens[k,d]=buffer[start+d]
        window=wnd
        if bias[e]!=0 and hits[e]>0:
            window=max(min_wnd,min(max_wnd,wnd+int(bias[e]*min(hits[e],horizon)/horizon)))
        window=min(window,maxdraft)
        delta=abs(offset-1-basepos)
        if enabled and delta>safe:
            window=min(window,max(min_wnd,min(midcap if delta<=mid else farcap,maxdraft)))
        caps[k]=min(n,window,cap);terminal[k]=0
        for d in range(n):
            matched=False
            for stop in stops:
                if tokens[k,d]==stop:matched=True;break
            if matched:
                terminal[k]=d+1;caps[k]=min(caps[k],d+1);break

@njit(cache=True,nogil=True)
def batch_pack(scores,indices,norms,table_rows,ers,eos,bufs,ros,rls,kns,biases,hits,
               table_params,tails,tail_lens,basepos,decoded,maxdraft,abs_params,rowcaps,stops,stoplens,
               tokens,available,caps,terminal,features,status):
    for r in range(scores.shape[0]):
        if status[r]!=0:
            available[r,:]=0;caps[r,:]=0;features[r,:,:]=0.;continue
        tr=int(table_rows[r]);params=table_params[tr]
        width=min(scores.shape[1],int(params[4]))
        if width==8:
            # Preserve the compiler's exact full-top8 specialization in the
            # common path; masked K is a separate bounded slow path.
            f,_,_,_,st=extract_utility_features_numba(scores[r],indices[r],ers[tr],eos[tr],bufs[tr],
                ros[tr],rls[tr],kns[tr],tails[r,8-int(tail_lens[r]):],norms[r],int(basepos[r]),int(decoded[r]),int(params[4]))
        else:
            f,_,_,_,st=extract_utility_features_numba(scores[r,:width],indices[r,:width],ers[tr],eos[tr],bufs[tr],
                ros[tr],rls[tr],kns[tr],tails[r,8-int(tail_lens[r]):],norms[r],int(basepos[r]),int(decoded[r]),int(params[4]),True)
        if st!=0:status[r]=1;available[r,:]=0;caps[r,:]=0;features[r,:,:]=0.;continue
        if width==8:features[r,:,:]=f
        else:features[r,:,:]=0.;features[r,:width,:]=f
        pack_row(indices[r,:width],ers[tr],eos[tr],bufs[tr],ros[tr],rls[tr],biases[tr],hits[tr],
            int(params[0]),int(params[1]),int(params[2]),int(params[3]),bool(abs_params[0]),
            int(abs_params[1]),int(abs_params[2]),int(abs_params[3]),int(abs_params[4]),maxdraft,
            int(basepos[r]),int(rowcaps[r]),stops[r,:int(stoplens[r])],tokens[r],available[r],caps[r],terminal[r])

class I4Runtime:
    def __init__(self,bundle,mode='shadow',record_limit=4096):
        if mode not in ('shadow','execute'):raise ValueError('invalid I4 mode')
        if bundle.get('schema')!='hspec.i4.bundle.v1' or bundle.get('required_verifier')!='standard':raise ValueError('invalid I4 bundle')
        self.mode=mode;self.depth_cap=15;self.bundle=bundle;self.scorer=PrefixScorer(bundle['model']);self.solver=BudgetSolver()
        self.surface='depth_curve' in bundle['cost']
        self.costs=np.ascontiguousarray(bundle['cost']['depth_curve' if self.surface else 'curve'],dtype=np.float64)
        if self.costs.shape!=((5,2,16) if self.surface else (5,2,385)):raise ValueError('invalid cost ABI')
        self.costs[self.costs<=0]=np.nan
        self.anchors=np.array(bundle['cost'].get('anchors',[1,8,16,32,64]),np.int64)
        self.overhead=np.ascontiguousarray(bundle['cost'].get('native_overhead_ms',np.zeros((65,2))),dtype=np.float64)
        self.selector_cost=np.ascontiguousarray(bundle['cost'].get('selector_delta_ms',np.zeros((65,2))),dtype=np.float64)
        if self.overhead.shape!=(65,2) or self.selector_cost.shape!=(65,2):raise ValueError('invalid native cost terms')
        self.tokens=np.zeros((64,8,15),np.int32);self.available=np.zeros((64,8),np.int16)
        self.caps=np.zeros((64,8),np.int16);self.entries=np.full((64,8),-1,np.int32)
        self.features=np.zeros((64,8,18));self.ages=np.zeros(64);self.temps=np.full(64,.9)
        self.terminals=np.zeros((64,8),np.int16);self.status=np.ones(64,np.uint8)
        self.abs_params=np.zeros(5,np.int64)
        self.tails=np.zeros((64,8),np.int32);self.tail_lens=np.zeros(64,np.int16)
        self.basepos=np.zeros(64,np.int32);self.decoded=np.zeros(64,np.int32)
        self.rowcaps=np.zeros(64,np.int16);self.stops=np.zeros((64,16),np.int32);self.stoplens=np.zeros(64,np.int16)
        self.cost_work=np.empty(385);self.base_curve=np.empty(385);self.cached_curves=np.empty((2,385));self.cached_batch=-1
        self.table_rows=np.zeros(64,np.int32);self.bound_ids={};self.refs=None;self.table_params=None
        self.records=[];self.record_limit=record_limit;self.counts=Counter();self.last=None
        self.scorer.warmup();self.solver.warmup();self.warm_rows()
        if self.surface:self.cost_curve(1,64.)
    def validate_runtime(self,config):
        # Boundary-only compatibility check. Actor weight updates are a
        # declared transfer axis, not a claim of an exact static cost domain.
        d=self.bundle['cost']['domain'];kw=d['kwargs'];mc=config.model_config
        checks=(config.parallel_config.tensor_parallel_size==4,
            not config.parallel_config.enable_expert_parallel,
            int(config.speculative_config.num_speculative_tokens)==15,
            int(mc.max_model_len)==17408,str(mc.dtype) in ('torch.bfloat16','bfloat16'),
            bool(mc.enforce_eager)==bool(kw['enforce_eager']),
            abs(float(config.cache_config.gpu_memory_utilization)-kw['gpu_memory_utilization'])<1e-6,
            int(config.speculative_config.hspec_n_components)==64)
        if not all(checks):raise ValueError('I4 runtime outside declared TP/dtype/graph/cache/rank domain')
        model=Path(mc.model).resolve()
        for name,h in d['checkpoint']['asset_sha256'].items():
            if hashlib.sha256((model/name).read_bytes()).hexdigest()!=h:
                raise ValueError('I4 checkpoint asset mismatch: '+name)
    def warm_rows(self):
        er=np.zeros(8,np.int32);eo=np.arange(1,9,dtype=np.int32)
        buffer=np.arange(96,dtype=np.int32);ro=np.array([0],np.int64);rl=np.array([96],np.int32)
        # Reproduce the strided host-payload views and both one-row/strided
        # variants outside timed decode, not merely contiguous fixture arrays.
        for rows in (1,2):
            scores=np.ones((rows,10),np.float32)[:,:8]
            ids=np.tile(np.arange(9,dtype=np.int64),(rows,1))[:,:8]
            extract_utility_features_numba(scores[0],ids[0],er,eo,buffer,ro,rl,np.ones(8,np.float32),
                np.array([0],np.int32),1.,0,1,8)
            pack_row(ids[0],er,eo,buffer,ro,rl,np.zeros(8,np.int8),np.zeros(8,np.uint16),
                8,2,28,4,True,2,64,8,4,15,0,15,np.empty(0,np.int32),
                self.tokens[0],self.available[0],self.caps[0],self.terminals[0])
    def bind_tables(self,tables):
        # Called by prefetch/cache construction or an RPC boundary, never by
        # prepare(). Typed lists keep only CPU array references, no large copy.
        from numba.typed import List
        fields=('entry_rollout_idx','entry_offset','token_buffer','rollout_token_offset',
            'rollout_token_len','key_norm_cpu','entry_bias','entry_hits')
        if not tables or any(c.key_norm_cpu is None for c in tables):return
        self.refs=tuple(List([getattr(c,f) for c in tables]) for f in fields)
        self.bound_ids={id(c):i for i,c in enumerate(tables)}
        self.table_params=np.array([[c.wnd_size,c.min_wnd,c.max_wnd,c.entry_blend_horizon,c.n_entries] for c in tables],np.int64)
        # Compile the exact ragged ABI outside decode with empty caps.
        # Shape-one payload views are C contiguous; larger payload views are
        # strided (A). Warm BOTH layouts even when the first cache has 64 rows,
        # so a one-request tail cannot trigger a new compilation in decode.
        for B in (1,2):
            scores=np.ones((B,10),np.float32)[:,:8]
            ids=np.zeros((B,9),np.int64)[:,:8];self.status[:B]=1
            batch_pack(scores,ids,np.ones((B,10),np.float32)[:,9],self.table_rows[:B],*self.refs,self.table_params,
                self.tails[:B],self.tail_lens[:B],self.basepos[:B],self.decoded[:B],15,self.abs_params,
                self.rowcaps[:B],self.stops[:B],self.stoplens[:B],self.tokens[:B],self.available[:B],self.caps[:B],
                self.terminals[:B],self.features[:B],self.status[:B])
    @property
    def constants_bytes(self):return self.costs.nbytes+self.scorer.constants_bytes+self.overhead.nbytes+self.selector_cost.nbytes+self.anchors.nbytes
    @property
    def scratch_bytes(self):return self.scorer.scratch_bytes+self.solver.scratch_bytes+sum(x.nbytes for x in (self.tokens,self.available,self.caps,self.entries,self.features,self.ages,self.temps,self.terminals,self.status,self.tails,self.tail_lens,self.basepos,self.decoded,self.rowcaps,self.stops,self.stoplens,self.table_rows,self.cost_work,self.base_curve,self.cached_curves,self.abs_params))+(self.table_params.nbytes if self.table_params is not None else 0)+64*16*8
    def cost_curve(self,batch,context):
        if self.surface:
            if batch!=self.cached_batch:
                for c,ctx in enumerate((64.,8192.)):
                    interpolate_cost(self.costs,self.overhead,self.selector_cost,self.anchors,batch,ctx,-1,self.cached_curves[c])
                self.cached_batch=batch
            mix_cost_context(self.cached_curves[0],self.cached_curves[1],context,
                self.selector_cost[batch,1 if self.mode=='execute' else 0],self.base_curve)
            return self.base_curve
        return self.costs[self.bundle['cost']['batches'].index(batch),int(context)]
    def prepare(self,p,active,tables,scores,indices,norms,req_states,base_positions,decoded,raw_gate,batch_size,prefill_pending=False):
        start=time.perf_counter_ns();B=len(active)
        self.last=None
        if prefill_pending:
            # The cost model declares decode support. Required chunked prefill
            # remains independent R1; report its coverage, never score it as a
            # cheap decode action or classify it as a broken tail batch.
            self.counts['prefill_deferred_queries']+=B;return None
        self.counts['attempted_queries']+=B
        self.counts['inactive_rows']+=batch_size-B
        if not 0<B<=batch_size<=64 or batch_size not in self.bundle['cost']['batches']:
            self.counts['unsupported_batch']+=B;return None
        if not 1<=scores.shape[1]<=8:
            self.counts['unsupported_K']+=B;return None
        if self.refs is None or any(id(c) not in self.bound_ids for c in tables):
            self.counts['table_refs_not_ready']+=B;return None
        context=[];self.status[:B]=0;self.caps[:B]=0;self.available[:B]=0;self.terminals[:B]=0
        candidate_width=scores.shape[1]
        if candidate_width<8:self.entries[:B]=-1
        for r,i in enumerate(active):
            c=tables[r];req=req_states[i];current=getattr(req,'output_token_ids',())
            sp=getattr(req,'sampling_params',None);temp=float(getattr(sp,'temperature',.9));top_p=float(getattr(sp,'top_p',.9))
            self.temps[r]=temp
            ctx=len(getattr(req,'prompt_token_ids',()))+len(current)
            context.append(ctx if self.surface else (0 if ctx<=1024 else (1 if 8192<=ctx<=17408 else -1)))
            if ctx>int(p.runner.max_model_len):self.status[r]=1
            if req is None or temp!=.9 or top_p!=.9:self.status[r]=1
            tr=self.bound_ids[id(c)];self.table_rows[r]=tr;self.table_params[tr,0]=c.wnd_size
            self.basepos[r]=int(base_positions[r]);self.decoded[r]=int(decoded[i])
            n=min(len(current),8);self.tail_lens[r]=n;self.tails[r,:]=0
            if n:self.tails[r,8-n:]=current[-n:]
            response=max(0,int(getattr(sp,'max_tokens',17408))-len(current)-1)
            self.rowcaps[r]=min(response,max(0,int(p.runner.max_model_len)-ctx-1),self.depth_cap) if raw_gate[r] else 0
            stops=list(getattr(sp,'stop_token_ids',()) or ())
            if not getattr(sp,'ignore_eos',False):
                eos_token=getattr(req,'eos_token_id',None)
                if eos_token is not None:stops.append(int(eos_token))
            if len(stops)>16:self.status[r]=1;stops=[]
            self.stoplens[r]=len(stops)
            if stops:self.stops[r,:len(stops)]=stops
            self.ages[r]=0.
            if candidate_width==8:self.entries[r]=indices[r]
            else:self.entries[r,:candidate_width]=indices[r]
            if c.n_entries<candidate_width:self.entries[r,int(c.n_entries):]=-1
        self.counts['unknown_actor_age']+=B
        self.abs_params[:]=(p._abs_delta_cap_enabled,p._abs_delta_safe_threshold,p._abs_delta_mid_threshold,
            p._abs_delta_mid_cap,p._abs_delta_far_cap)
        batch_pack(scores,indices,norms,self.table_rows[:B],*self.refs,self.table_params,
            self.tails[:B],self.tail_lens[:B],self.basepos[:B],self.decoded[:B],int(p.max_draft_tokens),self.abs_params,
            self.rowcaps[:B],self.stops[:B],self.stoplens[:B],self.tokens[:B],self.available[:B],self.caps[:B],
            self.terminals[:B],self.features[:B],self.status[:B])
        if not self.surface and (len(set(context))!=1 or context[0]<0):
            self.counts['unknown_cost_domain']+=B;return None
        if np.any(self.status[:B]):
            # A bad row is kept for independent R1 fallback. Its gain/actions
            # are zero in the DP; successful rows are not reranked by R1.
            for r in range(B):
                if self.status[r]:self.features[r]=0.;self.available[r]=0;self.caps[r]=0
        strategy=self.bundle.get('solver_by_batch',{}).get(str(batch_size),'dp' if batch_size<=32 else 'sparse_menu')
        # Descendants below every legal action cannot change ancestor mass.
        # Keep ALL candidate branches through the deepest possible action,
        # including branches of entries with shorter individual caps.
        action_depth=min(self.depth_cap,384//B if strategy=='sparse_menu' else 15)
        np.minimum(self.available[:B],action_depth,out=self.available[:B])
        g,st=self.scorer.score(self.tokens[:B],self.available[:B],self.features[:B],self.ages[:B],self.temps[:B])
        self.status[:B]|=st
        for r in range(B):
            if self.status[r]:self.caps[r]=0
        # Non-active requests still contribute their AR/bonus token and target
        # cost. They are not invented zero-filled candidate rows.
        mean_context=((sum(context)/B if B==batch_size else sum(len(getattr(req,'prompt_token_ids',()))+len(getattr(req,'output_token_ids',())) for req in req_states[:batch_size])/batch_size)) if self.surface else context[0]
        curve=self.cost_curve(batch_size,mean_context)
        reserve=min(384,15*int(np.count_nonzero(self.status[:B])))
        if reserve:
            self.cost_work[:]=np.nan
            self.cost_work[:385-reserve]=curve[reserve:]
            curve=self.cost_work

        solve=self.solver.solve if strategy=='dp' else self.solver.solve_menu
        try:result=solve(g,self.caps[:B],self.entries[:B],self.terminals[:B],curve,base_batch=batch_size)
        except Exception:
            if strategy!='dp':raise
            # A solver-local failure keeps the same legal model/action/cost
            # ABI and uses the predeclared bounded schedule menu. Failure of
            # this path propagates to the independent R1/P0 fallback.
            self.counts['dp_to_sparse_menu_recovery']+=1
            result=self.solver.solve_menu(g,self.caps[:B],self.entries[:B],self.terminals[:B],curve,base_batch=batch_size)
            strategy='sparse_menu_recovery'
        result['strategy']=strategy
        if result['status']!='PASS':self.counts['unknown_cost_domain']+=B;return None
        result['valid']=self.status[:B]==0
        result['entry']=self.entries[np.arange(B),np.maximum(result['slot'],0)].copy()
        result['entry'][result['slot']<0]=-1
        result['elapsed_ms']=(time.perf_counter_ns()-start)/1e6
        result['context_index']=context[0]
        result['cost_batch']=batch_size;result['mean_context']=mean_context
        self.last=result;self.counts['queries']+=B;self.counts['valid_queries']+=int(np.count_nonzero(result['valid']))
        return result
    def finish(self,p,results,active,req_ids,decoded,tables):
        if self.last is None:return
        x=self.last
        self.counts['executed_queries' if self.mode=='execute' else 'shadow_queries']+=int(np.count_nonzero(x['valid']))
        if self.mode=='execute':
            self.counts['nonempty_execute_queries']+=sum(bool(results[i]) for r,i in enumerate(active) if x['valid'][r])
            self.counts['executed_draft_tokens']+=sum(len(results[i]) for r,i in enumerate(active) if x['valid'][r])
        if self.record_limit==0:return
        if len(self.records)>=self.record_limit:self.counts['dropped_records']+=1;return
        actual_n=sum(len(v) for v in results)
        actual_cost=self.cost_curve(x['cost_batch'],x['mean_context'])[actual_n] if actual_n<=384 else np.nan
        self.records.append({'mode':self.mode, 'query_sequence':self.counts['queries'],
            'proposal_step_id':getattr(p.runner,'_hspec_i4_step_id',None),
            'proposed_lengths':[int(v) for v in x['length']], 'solver_strategy':x['strategy'],
            'cost_batch':x['cost_batch'],'mean_context':x['mean_context'],
            'predicted_actual_cost_ms':float(actual_cost) if math.isfinite(actual_cost) else None,'requests':[str(req_ids[i]) for i in active],
            'all_requests':[str(i) for i in req_ids[:x['cost_batch']]],'all_actual':[list(v) for v in results],
            'decoded':[int(decoded[i]) for i in active],'table_version':int(p._cache_version),
            'prefix_tails':[self.tails[r,8-int(self.tail_lens[r]):].tolist() for r in range(len(active))],
            'cache_generation':int(p._cache_generation),'model_version':self.scorer.model_version,
            'actual':[list(results[i]) for i in active],
            'hypothetical':[tables[r].get_draft_tokens(int(x['entry'][r]),int(x['length'][r])) if x['entry'][r]>=0 else [] for r in range(len(active))],
            'valid':x['valid'].tolist(),'expected':x['gain'].tolist(),
            'selector_ms':float(x['elapsed_ms']),'predicted_cost_ms':float(x['cost_ms']),
            'cost_domain_hash':self.bundle['cost']['domain_hash'],
            'cost_semantics':self.bundle['cost']['semantics'],'counterfactual_label':None})
    def drain(self):
        r={'records':self.records,'counts':dict(self.counts),'constants_bytes':self.constants_bytes,'scratch_bytes':self.scratch_bytes}
        self.records=[];self.counts.clear();return r

def load_i4(path,expected_sha,mode,record_limit=0):
    with open(path,'rb') as f:raw=f.read()
    if not expected_sha or hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('I4 bundle hash mismatch')
    return I4Runtime(json.loads(raw),mode,record_limit)
