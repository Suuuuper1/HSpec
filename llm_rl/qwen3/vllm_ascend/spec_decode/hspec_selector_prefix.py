"""Frozen I2 prefix-linear ABI, ported to bounded arrays for I4.

Only NumPy/Numba; no device, RPC or I/O in score(). The v1 artifact deliberately
uses the I2 parent representative (first source slot, root slot 0). Changing that
representation is a new I2 model, not an I4 runtime optimization.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit

FEATURE_NAMES = tuple([f'p3a_{i}' for i in range(18)] +
    ['depth_1','depth_2_3','depth_4_7','depth_8_15','actor_age_norm','recipe_temperature'])


def _kernel(tokens, available, features, ages, temperatures, theta, other_theta,
            mean, scale, calibration_t, calibration_b, parent, token, depth,
            representative, paths, mass, first, sibling, logits, used, probabilities, gains, status):
    B, K, D = tokens.shape
    for b in range(B):
        status[b] = 0
        gains[b,:,:] = 0.
        parent[0] = -1; depth[0] = 0; representative[0] = 0; mass[0] = 1.
        count = 1;first[:]=-1;sibling[:]=-1
        for k in range(K):
            p = 0
            for f in range(18):
                if not math.isfinite(features[b,k,f]): status[b] = 1
            for d in range(int(available[b,k])):
                t = tokens[b,k,d]; found = -1
                n=int(first[p])
                while n>=0:
                    if token[n] == t:found=n;break
                    n=int(sibling[n])
                if found < 0:
                    found = count; count += 1
                    parent[found] = p; token[found] = t; depth[found] = d+1
                    representative[found] = k
                    sibling[found]=first[p];first[p]=found
                paths[k,d] = found; p = found
        if status[b] != 0: continue
        used[:]=False
        for p in range(count):
            if first[p]>=0:used[int(representative[p])]=True
        probabilities[:,:,:]=-1.
        for slot in range(K):
            if not used[slot]:continue
            child_base=0.;other_base=0.
            for f in range(24):
                x=0.
                if f<18:x=features[b,slot,f]
                elif f==22:x=min(max(ages[b]/32.,-8.),8.)
                elif f==23:x=min(max(temperatures[b],0.),2.)
                z=(x-mean[f])/scale[f]
                child_base+=z*theta[f];other_base+=z*other_theta[f]
            for bucket in range(4):
                child_logit=child_base+theta[18+bucket]/scale[18+bucket]
                other_logit=other_base+other_theta[18+bucket]/scale[18+bucket]
                denominator=max(calibration_t[bucket],1e-4)
                logits[slot,bucket,0]=child_logit/denominator
                logits[slot,bucket,1]=(other_logit+calibration_b[bucket])/denominator
        for p in range(count):
            children=0;n=int(first[p])
            while n>=0:children+=1;n=int(sibling[n])
            if children==0:continue
            d=int(depth[p])+1
            bucket=0 if d==1 else (1 if d<=3 else (2 if d<=7 else 3))
            slot=int(representative[p])
            probability=probabilities[slot,bucket,children]
            if probability<0.:
                child_logit=logits[slot,bucket,0];other_logit=logits[slot,bucket,1]
                mx = max(child_logit,other_logit)
                c = math.exp(max(child_logit-mx,-700.))
                o = math.exp(max(other_logit-mx,-700.))
                probability = c/(children*c+o)
                probabilities[slot,bucket,children]=probability
            n=int(first[p])
            while n>=0:mass[n]=mass[p]*probability;n=int(sibling[n])
        for k in range(K):
            total = 0.
            for d in range(D):
                if d < available[b,k]: total += mass[int(paths[k,d])]
                gains[b,k,d+1] = total
        if not np.isfinite(gains[b]).all(): status[b] = 1; gains[b,:,:] = 0.

score_kernel = njit(cache=True,nogil=True)(_kernel)


class PrefixScorer:
    def __init__(self, artifact, max_batch=64, max_k=8, max_depth=15):
        if artifact.get('schema_version') != 'hspec.i2.prefix-linear.v1' or tuple(artifact.get('feature_names',())) != FEATURE_NAMES:
            raise ValueError('prefix artifact ABI mismatch')
        self.model_version = str(artifact['model_version'])
        self.constants = tuple(np.ascontiguousarray(artifact[name], dtype=np.float64)
            for name in ('theta','other_theta','feature_mean','feature_scale')) + tuple(
            np.ascontiguousarray(artifact['calibration'][name],dtype=np.float64)
            for name in ('temperature','other_bias'))
        for a, size in zip(self.constants,(24,24,24,24,4,4)):
            if a.shape != (size,) or not np.isfinite(a).all(): raise ValueError('nonfinite or invalid model constants')
        if np.any(self.constants[3] <= 0) or np.any(self.constants[4] <= 0): raise ValueError('scale/temperature must be positive')
        self.max_batch, self.max_k, self.max_depth = max_batch,max_k,max_depth
        N=1+max_k*max_depth
        self.scratch = (np.empty(N,np.int16),np.empty(N,np.int32),np.empty(N,np.int16),
                        np.empty(N,np.int16),np.empty((max_k,max_depth),np.int16),np.empty(N,np.float64),np.empty(N,np.int16),np.empty(N,np.int16),np.empty((max_k,4,2)),np.empty(max_k,np.bool_),np.empty((max_k,4,max_k+1)))
        self.gains=np.zeros((max_batch,max_k,max_depth+1),np.float64)
        self.status=np.zeros(max_batch,np.uint8)

    @property
    def constants_bytes(self): return sum(x.nbytes for x in self.constants)
    @property
    def scratch_bytes(self): return sum(x.nbytes for x in self.scratch)+self.gains.nbytes+self.status.nbytes

    def score(self,tokens,available,features,ages,temperatures,compiled=True):
        B,K,D=tokens.shape
        if B>self.max_batch or K>self.max_k or D!=self.max_depth:
            raise ValueError('prefix input outside bounded ABI')
        if tokens.dtype!=np.int32 or available.dtype!=np.int16 or features.dtype!=np.float64:
            raise ValueError('prefix dtype mismatch')
        if available.shape!=(B,K) or features.shape!=(B,K,18) or ages.shape!=(B,) or temperatures.shape!=(B,):
            raise ValueError('prefix input shape mismatch')
        if np.any(available<0) or np.any(available>D): raise ValueError('invalid available length')
        if not all(x.flags.c_contiguous for x in (tokens,available,features,ages,temperatures)):
            raise ValueError('prefix arrays must be contiguous')
        fn=score_kernel if compiled else _kernel
        fn(tokens,available,features,ages,temperatures,*self.constants,*self.scratch,self.gains[:B,:K],self.status[:B])
        return self.gains[:B,:K],self.status[:B]

    def warmup(self):
        self.score(np.zeros((1,8,15),np.int32),np.ones((1,8),np.int16),
                   np.zeros((1,8,18)),np.zeros(1),np.full(1,.9))
