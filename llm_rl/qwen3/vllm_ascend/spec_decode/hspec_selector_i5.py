"""Opt-in I5 actions and batch guard; the I4 bundle/default path is unchanged.

No file/RPC/device operations in prepare. A margin is a conservative model
sensitivity envelope, not a statistically calibrated confidence interval.
"""
from __future__ import annotations
import hashlib,json,random
import numpy as np
from numba import njit
from .hspec_selector_i4 import I4Runtime

@njit(cache=True,nogil=True)
def choose_r1(features, entries, caps, out):
    for r in range(len(out)):
        best=-1
        for k in range(entries.shape[1]):
            if entries[r,k]<0 or caps[r,k]<=0 or features[r,k,0]<-1.e-4:continue
            # feature0 is relative cosine drop; feature1 is suffix / 8.
            if best<0 or features[r,k,1]>features[r,best,1] or (features[r,k,1]==features[r,best,1] and features[r,k,0]>features[r,best,0]):best=k
        out[r]=best

@njit(cache=True,nogil=True)
def choose_model_fixed(g,caps,entries,terminal,out,nominal=8):
    for r in range(len(out)):
        best=-1;value=-1.
        for k in range(caps.shape[1]):
            d=min(nominal,int(caps[r,k]))
            if d<=0 or entries[r,k]<0:continue
            v=g[r,k,d];t=int(terminal[r,k])
            if t>0 and d>=t:v-=g[r,k,t]-g[r,k,t-1]
            if v>value+1.e-12 or (abs(v-value)<=1.e-12 and (best<0 or entries[r,k]<entries[r,best])):value=v;best=k
        out[r]=best

@njit(cache=True,nogil=True)
def endpoint_kernel(g,caps,entries,terminal,curve,base,slots,lengths,expected):
    B=len(caps);n=0
    # A common window within the real cap384; per-row legal endpoints can
    # be shorter. No after-the-fact clipping of a scored longer action.
    common=384//max(B,1)
    for r in range(B):
        best=-1;value=-1.;length=0
        for k in range(caps.shape[1]):
            d=min(common,int(caps[r,k]))
            if d<=0 or entries[r,k]<0:continue
            v=g[r,k,d]
            t=int(terminal[r,k])
            if t>0 and d>=t:v-=g[r,k,t]-g[r,k,t-1]
            if v>value:value=v;best=k;length=d
        slots[r]=best;lengths[r]=length;expected[r]=max(value,0.);n+=length
    if n>=len(curve) or not np.isfinite(curve[n]):return -1,0.,0.
    return n,(base+np.sum(expected))/curve[n],curve[n]

@njit(cache=True,nogil=True)
def greedy_fixed_kernel(g,caps,terminal,curve,base,fixed,slots,lengths,expected):
    slots[:]=-1;lengths[:]=0;expected[:]=0.;n=0;total=0.
    if not np.isfinite(curve[0]):return -1,0.,0.
    ratio=base/curve[0]
    while n<384 and n+1<len(curve) and np.isfinite(curve[n+1]):
        best=-1;best_value=0.;best_ratio=ratio
        for r in range(len(caps)):
            k=fixed[r]
            if k<0:continue
            d=int(lengths[r])+1
            if d>caps[r,k]:continue
            value=g[r,k,d];t=int(terminal[r,k])
            if t>0 and d>=t:value-=g[r,k,t]-g[r,k,t-1]
            candidate=(base+total-expected[r]+value)/curve[n+1]
            if candidate>best_ratio+1.e-12:best=r;best_value=value;best_ratio=candidate
        if best<0:break
        total+=best_value-expected[best];expected[best]=best_value;lengths[best]+=1;slots[best]=fixed[best];n+=1;ratio=best_ratio
    return n,ratio,curve[n]

class ActionSolver:
    def __init__(self, native, owner):self.native=native;self.owner=owner
    def __getattr__(self,name):return getattr(self.native,name)
    def _solve(self,method,g,caps,entries,terminals,cost,**kw):
        o=self.owner;B=len(caps);variant=o.i5.get('action','joint')
        if variant=='r1_entry_exact' or o.i5.get('guard')=='margin':
            choose_r1(o.features[:B],entries,caps,o.r1_slots[:B])
        if variant in ('fixed_exact','fixed_greedy'):choose_model_fixed(g,caps,entries,terminals,o.r1_slots[:B],min(8,384//B) if B>32 else 8)
        if o.i5.get('radius_actions'):
            caps[o.features[:B,:,0]<-1.e-4]=0
        if o.i5.get('post_clip'):
            o.original_caps[:B]=caps
            np.minimum(o.available[:B],o.rowcaps[:B,None],out=caps)
            # A07 delays heuristic caps only. EOS/stop is always a hard
            # boundary, including in the hypothetical optimization.
            for r in range(B):
                for k in range(caps.shape[1]):
                    if terminals[r,k]>0:caps[r,k]=min(caps[r,k],terminals[r,k])
        if variant in ('fixed_exact','fixed_greedy','r1_entry_exact'):
            for r in range(B):
                keep=o.r1_slots[r]
                for k in range(caps.shape[1]):
                    if k!=keep:caps[r,k]=0
            if variant=='fixed_greedy':
                n,ratio,cost_ms=greedy_fixed_kernel(g,caps,terminals,cost,kw['base_batch'],o.r1_slots[:B],o.endpoint_slots[:B],o.endpoint_lengths[:B],o.endpoint_gains[:B])
                return {'status':'PASS' if n>=0 else 'UNKNOWN_COST','n':n,'ratio':ratio,'cost_ms':cost_ms,
                    'slot':o.endpoint_slots[:B],'length':o.endpoint_lengths[:B],'gain':o.endpoint_gains[:B]}
        elif variant=='r1_c':
            # Restrict to the exact common legal window and optimize entries.
            # Endpoint-only finite actions are handled by the separate kernel.
            n,ratio,cost_ms=endpoint_kernel(g,caps,entries,terminals,cost,kw['base_batch'],o.endpoint_slots[:B],o.endpoint_lengths[:B],o.endpoint_gains[:B])
            return {'status':'PASS' if n>=0 else 'UNKNOWN_COST','n':n,'ratio':ratio,'cost_ms':cost_ms,
                'slot':o.endpoint_slots[:B],'length':o.endpoint_lengths[:B],'gain':o.endpoint_gains[:B]}
        result=getattr(self.native,method)(g,caps,entries,terminals,cost,**kw)
        if o.i5.get('post_clip') and result['status']=='PASS':
            for r in range(B):
                k=result['slot'][r]
                if k<0:continue
                d=min(int(result['length'][r]),int(o.original_caps[r,k]));result['length'][r]=d
                v=g[r,k,d];t=int(terminals[r,k])
                if t>0 and d>=t:v-=g[r,k,t]-g[r,k,t-1]
                result['gain'][r]=v
            result['n']=int(np.sum(result['length']));result['cost_ms']=float(cost[result['n']])
            result['ratio']=(kw['base_batch']+float(np.sum(result['gain'])))/result['cost_ms']
        return result
    def solve(self,*a,**kw):return self._solve('solve',*a,**kw)
    def solve_menu(self,*a,**kw):return self._solve('solve_menu',*a,**kw)

class I5Runtime(I4Runtime):
    def __init__(self,bundle,mode='execute',record_limit=0):
        self.i5=bundle['i5']
        self.calibration_rng=random.Random(20261012)
        if self.i5.get('guard') not in ('none','fixed','margin'):raise ValueError('invalid I5 guard')
        if self.i5.get('action','joint') not in ('joint','fixed_exact','fixed_greedy','r1_c','r1_entry_exact'):raise ValueError('invalid I5 action')
        super().__init__(bundle,mode,record_limit)
        self.active_model_sha256=bundle['model_sha256'] if 'model_sha256' in bundle else ''
        # Pending versions distinguish coefficients/calibrator, including
        # LOTO models that intentionally share the same architecture name.
        self.scorer.model_version+='-'+self.active_model_sha256[:12]
        if 'entry_model' in self.i5:
            self.scorer=IndependentScorer(self.i5['entry_model'])
            self.active_model_sha256=hashlib.sha256(json.dumps(self.i5['entry_model'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if self.i5.get('no_other'):
            # Numerically zero OTHER (softmax floor exp(-700)); no new model
            # fit, and the same child logits. This is the A03 counterfactual.
            self.scorer.constants[-1][:]=-1.e6
            self.scorer.model_version+='-without-OTHER'
            self.active_model_sha256=hashlib.sha256((self.active_model_sha256+':without-OTHER').encode()).hexdigest()
        self.original_caps=np.zeros((64,8),np.int16);self.i5_curve=np.empty(385)
        self.i5_gate=np.ones(64,np.bool_)
        self.r1_slots=np.full(64,-1,np.int16)
        self.endpoint_slots=np.full(64,-1,np.int16);self.endpoint_lengths=np.zeros(64,np.int16);self.endpoint_gains=np.zeros(64)
        choose_r1(self.features[:1],self.entries[:1],self.caps[:1],self.r1_slots[:1])
        self.solver=ActionSolver(self.solver,self)
    def calibration_sample(self):
        return self.calibration_rng.random()<1./64.
    def finish(self,*args):
        old=len(self.records);super().finish(*args)
        if len(self.records)>old:self.records[-1]['active_model_sha256']=self.active_model_sha256
    @property
    def scratch_bytes(self):
        return super().scratch_bytes+sum(x.nbytes for x in (self.original_caps,self.i5_curve,self.i5_gate,self.r1_slots,self.endpoint_slots,self.endpoint_lengths,self.endpoint_gains))
    def cost_curve(self,batch,context):
        mode=self.i5.get('cost_mode','native')
        curve=super().cost_curve(batch,64. if mode=='nonly' else context)
        if mode=='constant' and hasattr(self,'i5_curve'):
            self.i5_curve[:]=curve[0];self.i5_curve[~np.isfinite(curve)]=np.nan
            return self.i5_curve
        return curve
    def prepare(self,p,active,tables,scores,indices,norms,req_states,base_positions,decoded,raw_gate,batch_size,prefill_pending=False):
        self.last=None
        if prefill_pending:
            return super().prepare(p,active,tables,scores,indices,norms,req_states,base_positions,decoded,raw_gate,batch_size,True)
        # batch_size is the real scheduler cohort, including cache-missing
        # decode rows. len(active) counts retrieval-ready rows, not occupancy.
        self.counts['i5_routed_queries']+=len(active)
        if self.i5['guard']!='none' and batch_size<int(self.i5.get('min_batch',16)):
            self.counts['i5_small_batch_guard_queries']+=len(active)
            p._record_proposer_metric('i5_small_batch_guard_queries',len(active));return None
        if self.i5.get('calibrated_gate',False):
            # Research ablation only: utility/abstain replaces raw similarity
            # admission. Packing still applies every EOS/KV/response cap.
            raw_gate=self.i5_gate[:len(active)]
        result=super().prepare(p,active,tables,scores,indices,norms,req_states,base_positions,decoded,raw_gate,batch_size,prefill_pending)
        if result is not None and self.i5['guard']=='margin':
            baseline_gain=0.;baseline_n=0
            for r in range(len(active)):
                k=self.r1_slots[r]
                if k>=0:
                    d=int(self.caps[r,k]);baseline_n+=d;baseline_gain+=self.scorer.gains[r,k,d]
            curve=self.cost_curve(batch_size,result['mean_context'])
            rho=float(self.i5.get('relative_cost_uncertainty',.35))
            candidate=(batch_size+float(np.sum(result['gain'])))/(result['cost_ms']*(1+rho))
            reference=(batch_size+baseline_gain)/(curve[min(baseline_n,384)]*max(1-rho,1.e-6))
            if baseline_n>384 or not np.isfinite(reference) or candidate<=reference*(1+float(self.i5.get('margin',.01))):
                self.counts['i5_cost_uncertain_queries']+=len(active);self.last=None;result=None
        if result is None:self.counts['i5_baseline_queries']+=len(active)
        else:
            self.counts['i5_learned_queries']+=int(np.count_nonzero(result['valid']))
            p._record_proposer_metric('i5_learned_queries',int(np.count_nonzero(result['valid'])))
        return result

@njit(cache=True,nogil=True)
def independent_kernel(available,features,ages,temps,theta,mean,scale,depth_base,depth_slope,cal_t,cal_b,hazard,gains):
    for r in range(len(available)):
        for k in range(available.shape[1]):
            gains[r,k,:]=0.
            if available[r,k]<=0:continue
            base=0.;slope=0.
            for j in range(24 if hazard else 18):
                x=0.
                if j<18:x=features[r,k,j]
                elif j==22:x=min(max(ages[r]/32.,-8.),8.)
                elif j==23:x=min(max(temps[r],0.),2.)
                limit=8. if hazard else 12.
                z=min(max((x-mean[j])/scale[j],-limit),limit)
                base+=z*theta[j]
                if hazard:slope+=z*theta[24+j]
            mass=1.;total=0.
            for d in range(1,16):
                if d<=available[r,k]:
                    bucket=0 if d==1 else 1 if d<=3 else 2 if d<=7 else 3
                    logit=base+depth_base[bucket]+(slope+depth_slope[bucket])*d/15. if hazard else base+theta[18+d-1]
                    prob=1./(1.+np.exp(-min(max((logit+cal_b[bucket])/cal_t[bucket],-40.),40.)))
                    mass=mass*prob if hazard else prob;total+=mass
                gains[r,k,d]=total

class IndependentScorer:
    def __init__(self,artifact):
        self.model_version='i5-independent-'+artifact['kind']+'-'+hashlib.sha256(json.dumps(artifact,sort_keys=True).encode()).hexdigest()[:12];self.hazard=artifact['kind']=='hazard'
        self.theta=np.asarray(artifact['theta'],np.float64);self.mean=np.asarray(artifact['feature_mean'],np.float64);self.scale=np.asarray(artifact['feature_scale'],np.float64);self.temperature=float(artifact['temperature'])
        if self.theta.shape!=((48,) if self.hazard else (33,)) or self.mean.shape!=(24,) or self.scale.shape!=(24,) or np.any(self.scale<=0) or self.temperature<=0 or not all(np.isfinite(x).all() for x in (self.theta,self.mean,self.scale)):raise ValueError('invalid independent scorer')
        if not self.hazard and np.any(np.diff(self.theta[18:])>1.e-7):raise ValueError('nonmonotone P3A depth biases')
        dc=artifact.get('depth_calibration',{});self.cal_t=np.asarray(dc.get('temperature',[self.temperature]*4),np.float64);self.cal_b=np.asarray(dc.get('bias',[0.]*4),np.float64)
        if self.cal_t.shape!=(4,) or self.cal_b.shape!=(4,) or not np.isfinite(self.cal_t).all() or not np.isfinite(self.cal_b).all() or np.any(self.cal_t<=0):raise ValueError('invalid depth calibration')
        self.depth_base=np.zeros(4);self.depth_slope=np.zeros(4)
        if self.hazard:
            delta=np.clip((1.-self.mean[18:22])/self.scale[18:22],-8.,8.)-np.clip(-self.mean[18:22]/self.scale[18:22],-8.,8.)
            self.depth_base[:]=delta*self.theta[18:22];self.depth_slope[:]=delta*self.theta[42:46]
        self.gains=np.zeros((64,8,16));self.status=np.zeros(64,np.uint8)
        self.score(np.ones((1,8,15),np.int32),np.ones((1,8),np.int16),np.zeros((1,8,18)),np.zeros(1),np.full(1,.9))
    @property
    def constants_bytes(self):return self.theta.nbytes+self.mean.nbytes+self.scale.nbytes+self.cal_t.nbytes+self.cal_b.nbytes+self.depth_base.nbytes+self.depth_slope.nbytes
    @property
    def scratch_bytes(self):return self.gains.nbytes+self.status.nbytes
    def score(self,tokens,available,features,ages,temps):
        B=len(available);self.status[:B]=0
        independent_kernel(available,features,ages,temps,self.theta,self.mean,self.scale,self.depth_base,self.depth_slope,self.cal_t,self.cal_b,self.hazard,self.gains[:B])
        return self.gains[:B],self.status[:B]
