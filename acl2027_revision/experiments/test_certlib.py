import numpy as np, itertools, time
from certlib import *
# Table 1 / Prop 9 at T=6
inst = Instance(np.array([[7]*6,[6]*6]), np.array([[1]*6,[1,0,0,0,0,0]]), 6)
print("Prop9 T=6 joint", joint_min(inst,0).value, "union", union_min(inst,0), "split", split_min(inst,0))
bf = brute_force(inst,0); print("brute", bf)
# M.5 two-row example caps 2, contrib 1 each, b=1
inst2 = Instance(np.array([[2],[2]]), np.array([[1],[1]]), 1)
print(brute_force(inst2,0), joint_min(inst2,0).value, union_min(inst2,0), split_min(inst2,0))
rng = np.random.default_rng(0); bad=0; n=0
t=time.time()
for trial in range(400):
    m = rng.integers(1,4); T = rng.integers(1,4)
    p = rng.integers(1,4,size=m)
    c = np.repeat(p[:,None],T,axis=1)
    if rng.random()<0.3: c = rng.integers(1,4,size=(m,T))   # weighted caps
    v = rng.integers(0, c+1)
    if rng.random()<0.3: v = np.where(rng.random((m,T))<0.5, 0, c)  # endpoint-only
    obs = rng.random((m,T))<0.25
    b = int(rng.integers(0, c.sum()+1))
    inst = Instance(c,v,b,obs)
    for eps in range(0, m+1):
        bf = brute_force(inst,eps)
        j = joint_min(inst,eps).value; u = union_min(inst,eps); s = split_min(inst,eps)
        n+=1
        if (j,u,s)!=(bf['joint'],bf['union'],bf['split']):
            bad+=1; print("MISMATCH",c.tolist(),v.tolist(),obs.astype(int).tolist(),b,eps,(j,u,s),bf)
        # greedy exclusion vs milp
        for pp in range(1,m+2):
            assert exclusion_min(inst,pp)==exclusion_min_milp(inst,pp), (c,v,b,pp)
print("random checks", n, "mismatches", bad, time.time()-t)
# median closed form vs brute (opposition units, T entries per row, empty history)
bad=0
for trial in range(300):
    m=rng.integers(1,4); T=rng.integers(1,4)
    Y = (rng.random((m,T))<rng.random()).astype(int)
    b = int(rng.integers(0,m*T+1)); eps=int(rng.integers(0,m+1))
    inst = Instance(np.ones((m,T),int), Y, b)
    if median_closed_form(Y.sum(1),T,b,eps)!=brute_force(inst,eps)['joint']: bad+=1
print("median closed form mismatches", bad)
