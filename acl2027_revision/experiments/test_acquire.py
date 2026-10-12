import numpy as np, time
from certlib import *
from acquire import run_policy
from panels import *
rng=np.random.default_rng(1)
# oracle OM-CF consistency on random Boolean (odd-median-like) instances
bad=0;n=0
for trial in range(500):
    m=int(rng.integers(1,7)); T=int(rng.integers(1,6)); h1=int(rng.integers(1,4))
    Y=(rng.random((m,T))<rng.random()).astype(int)
    b=int(rng.integers(0,m*T*h1+1)); 
    inst=Instance(np.full((m,T),h1), Y*h1, b)
    for eps in range(0,m+1):
        q=run_policy(inst,None,"omcf",eps,seed=trial,oracle=True)
        Q=median_closed_form(Y.sum(1),T,b//h1,eps)
        n+=1
        if q!=Q: bad+=1; 
        if q!=Q and bad<5: print("mismatch",Y.tolist(),h1,b,eps,q,Q)
print("oracle omcf vs closed form:",n,"bad",bad)
# panel test
pan=make_panel(n_groups=600,seed=3)
for pool in ["median","mean"]:
    t=time.time()
    task=build_task(pan,0,0.05,0.10,pool)
    inst=task.inst
    print(pool,"N",inst.meta['N'],"S0",task.S0,"m",inst.m,"m0",inst.meta['m0'],"b",inst.b,"k",task.k, "bounds",count_bounds(inst),"L*",true_optimum(inst))
    eps=int(np.ceil(0.02*task.S0))
    for pol in ["fixed","random","width","reference","frontier","omcf"]:
        print(" ",pol,[run_policy(inst,task.pred,pol,e,seed=0) for e in (eps,0)])
    if pool=="median":
        z=(inst.v.sum(1)//inst.c[:,0]); bb=inst.b//inst.c[0,0]
        print("  Qeps closed", median_closed_form(z,inst.T,bb,eps), median_closed_form(z,inst.T,bb,0))
    else:
        print("  joint", joint_min(inst,eps,30).value, joint_min(inst,0,30).value, time.time()-t)
