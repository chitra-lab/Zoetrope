"""Frozen cortical crescent benchmark v1.0. Python -m runnable; numpy/pandas only."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

VERSION = "cortical_crescent_v1.0"
ROTATIONS = {2:-38., 5:57., 8:-65.}
AMPLITUDES = np.array([3.5,3.1,3.3,2.,3.6,3.6,2.8,2.8])

@dataclass(frozen=True)
class Config:
    cells_at_t0:int=1800
    population:str="growing"
    sampling:str="uniform_physical"
    regime:str="B"
    parallel_strength:float=0.
    offaxis_strength:float=0.
    library_log_sd:float=0.
    seed:int=20261008

    def validate(self):
        if self.cells_at_t0<1 or self.population not in {"growing","constant"}:
            raise ValueError("invalid population configuration")
        if self.sampling not in {"uniform_physical","uniform_intrinsic"} or self.regime not in set("ABCD"):
            raise ValueError("invalid sampling or regime")
        if min(self.parallel_strength,self.offaxis_strength,self.library_log_sd)<0:
            raise ValueError("negative perturbation")
        if self.regime!="C" and self.parallel_strength:
            raise ValueError("parallel_strength is allowed only in C")
        if self.regime!="D" and self.offaxis_strength:
            raise ValueError("offaxis_strength is allowed only in D")

def kappa(t):
    return 3.35*np.maximum(np.asarray(t),0.)**1.15

def width(u,t):
    return .17*(1+t*(.82+.70*np.exp(-.5*((np.asarray(u)-.5)/.26)**2)))

def jacobian(u,z,t):
    h=width(u,t)
    return h*(1.25+kappa(t)*np.asarray(z)*h)

def area(t):
    u=np.linspace(0,1,2049)
    h=width(u,t)
    f=1.25*h+.5*kappa(t)*h*h
    return float(np.sum(.5*(f[1:]+f[:-1])*np.diff(u)))

def geometry(u,z,t):
    u,z=np.broadcast_arrays(np.asarray(u,float),np.asarray(z,float))
    s=u-.5; k=kappa(t); th=k*s; h=width(u,t)
    cx=1.25*s*np.sinc(th/np.pi)
    cy=.625*k*s*s*np.sinc(th/(2*np.pi))**2
    nx=np.sin(th);ny=-np.cos(th)
    return cx+z*h*nx,cy+z*h*ny,h*nx,h*ny

def rotate(x,y,deg):
    a=np.deg2rad(deg)
    return np.cos(a)*x-np.sin(a)*y,np.sin(a)*x+np.cos(a)*y

def velocity(z,t):
    return 1.8*z*(1-z)*(1+.45*(2*z-1))*(1-.9*(3*t*t-2*t*t*t))

def velocity_z_derivative(z,t):
    return 1.8*((1-2*z)*(1+.45*(2*z-1))+.9*z*(1-z))*(1-.9*(3*t*t-2*t*t*t))

def flow(a,tend):
    z=np.asarray(a,float).copy(); n=max(1,int(np.ceil(300*tend))); dt=tend/n
    for i in range(n):
        t=i*dt;k1=velocity(z,t);k2=velocity(z+dt*k1/2,t+dt/2)
        k3=velocity(z+dt*k2/2,t+dt/2);k4=velocity(z+dt*k3,t+dt)
        z+=dt*(k1+2*k2+2*k3+k4)/6
    return z

def sample(n,t,rng,mode):
    if mode=="uniform_intrinsic":
        return rng.random(n),rng.random(n)
    max_j=float(np.max(jacobian(np.linspace(0,1,513),1.,t)))
    us=[];zs=[];remaining=n
    while remaining:
        m=max(64,int(remaining*1.7))
        u=rng.random(m);z=rng.random(m)
        good=rng.random(m)*max_j<=jacobian(u,z,t)
        us.append(u[good][:remaining]);zs.append(z[good][:remaining])
        remaining-=len(us[-1])
    return np.concatenate(us),np.concatenate(zs)

def expression_parameters(seed):
    rng=np.random.default_rng(seed+9001)
    raw=np.zeros((48,8))
    p=3*np.eye(8)+.35*rng.normal(size=(8,8))
    for j in range(8):
        raw[j*5:(j+1)*5]=p[j]+.09*rng.normal(size=(5,8))
    load,_=np.linalg.qr(raw)
    for j in range(8):
        if load[j*5:(j+1)*5,j].mean()<0:load[:,j]*=-1
    baseline=8.7+rng.normal(0,.32,48)
    off=rng.normal(size=48);off/=np.linalg.norm(off)
    return load,baseline,off

def spatial(a):
    l=1/(1+np.exp(-(a-.64)/.105))
    return np.stack((1-2*a,2*np.exp(-.5*((a-.43)/.15)**2)-.65,2*l-1,np.sin(2*np.pi*a)),axis=-1)

def spatial_derivative(a):
    e=np.exp(-.5*((a-.43)/.15)**2);l=1/(1+np.exp(-(a-.64)/.105))
    return np.stack((-2*np.ones_like(a),-2*e*(a-.43)/.15**2,2*l*(1-l)/.105,2*np.pi*np.cos(2*np.pi*a)),axis=-1)

def psi(a,u,t,cfg,params):
    a,u,t=np.broadcast_arrays(np.asarray(a,float),np.asarray(u,float),np.asarray(t,float))
    load,base,off=params
    temporal=np.stack((t*np.cos(2*np.pi*a+.35),t*np.sin(2*np.pi*a+.35),
        4*t*(1-t)*np.cos(4*np.pi*a-.65),4*t*(1-t)*np.sin(4*np.pi*a-.65)),axis=-1)
    if cfg.regime=="A":temporal*=0
    out=base+(np.concatenate((spatial(a),temporal),axis=-1)*AMPLITUDES)@load.T
    if cfg.regime=="C":
        out+=cfg.parallel_strength*.15*t[...,None]*((spatial_derivative(a)*AMPLITUDES[:4])@load[:,:4].T)
    if cfg.regime=="D":
        out+=cfg.offaxis_strength*3*(np.sin(2*np.pi*u)*np.cos(np.pi*a)*(1+.35*t))[...,None]*off
    if not np.all(np.isfinite(out)) or np.min(out)<=0:raise ValueError("nonpositive Fisher-flat expression")
    return out

def rate(a,u,t,cfg,params=None):
    return psi(a,u,t,cfg,expression_parameters(cfg.seed) if params is None else params)**2/4

def derivatives(a,u,t,cfg,params):
    h=2e-5
    da=(psi(a+h,u,t,cfg,params)-psi(a-h,u,t,cfg,params))/(2*h)
    q=(psi(a,u,t+h,cfg,params)-psi(a,u,t-h,cfg,params))/(2*h)
    fa=(flow(a+h,t)-flow(a-h,t))/(2*h)
    psi_z=da/fa[:,None]
    vf=velocity(flow(a,t),t)-np.sum(q*psi_z,axis=1)/np.maximum(np.sum(psi_z**2,axis=1),1e-12)
    return q,psi_z,vf

def growth_rate(u,z,t,cfg):
    # Source needed for uniform physical density and given migration/deformation.
    h=width(u,t);k=kappa(t);s=u-.5
    hd=.17*(.82+.70*np.exp(-.5*(s/.26)**2))
    kd=3.35*1.15*t**.15
    J=jacobian(u,z,t)
    local=(hd*(1.25+2*k*z*h)+kd*z*h*h+k*h*h*velocity(z,t))/J+velocity_z_derivative(z,t)
    eps=1e-5;lo=max(0,t-eps);hi=min(1,t+eps)
    adot=(area(hi)-area(lo))/(hi-lo)
    if cfg.sampling=="uniform_intrinsic":
        return velocity_z_derivative(z,t)+(adot/area(t) if cfg.population=="growing" else 0)
    return local-(adot/area(t) if cfg.population=="constant" else 0)

def generate(cfg=Config()):
    cfg.validate();p=expression_parameters(cfg.seed)
    times=np.linspace(0,1,10);areas=np.array([area(t) for t in times])
    obs=[];hidden=[];count=[];rates=[];qarr=[];zarr=[]
    genes=[f"G{j+1:02d}" for j in range(48)]
    for i,t in enumerate(times):
        n=cfg.cells_at_t0 if cfg.population=="constant" else round(cfg.cells_at_t0*areas[i]/areas[0])
        rc=np.random.default_rng(cfg.seed+100+i)
        rn=np.random.default_rng(cfg.seed+10000+i)
        rp=np.random.default_rng(cfg.seed+20000+i)
        u,z=sample(n,t,rc,cfg.sampling)
        ag=np.linspace(0,1,4001);a=np.interp(z,flow(ag,t),ag)
        px,py,gzx,gzy=geometry(u,z,t)
        angle=ROTATIONS.get(i,0.)
        tx,ty=rp.uniform(-.12,.12,2)
        x,y=rotate(px,py,angle);x+=tx;y+=ty
        vz=velocity(z,t);vx,vy=rotate(gzx*vz,gzy*vz,angle)
        lam=rate(a,u,t,cfg,p)
        L=np.exp(rn.normal(-cfg.library_log_sd**2/2,cfg.library_log_sd,n)) if cfg.library_log_sd else np.ones(n)
        Y=rn.poisson(lam*L[:,None]).astype(np.int32)
        ids=[f"tissue{i:02d}_cell{j:05d}" for j in range(n)]
        obs.append(pd.DataFrame(dict(cell_id=ids,tissue_id=i,time=t,x=x,y=y)))
        q,psiz,vf=derivatives(a,u,t,cfg,p)
        hidden.append(pd.DataFrame(dict(cell_id=ids,material_a_true=a,u_true=u,z_true=z,
            vx_migration_true=vx,vy_migration_true=vy,vz_true=vz,v_fisher_true=vf,
            q_norm_true=np.linalg.norm(q,axis=1),fisher_observability_true=np.sum(psiz**2,axis=1),
            growth_rate_continuity_true=growth_rate(u,z,t,cfg),true_library_factor=L,
            rotation_deg=angle,translation_x=tx,translation_y=ty,
            physical_x_true=px,physical_y_true=py)))
        count.append(pd.DataFrame(Y,columns=genes).assign(cell_id=ids))
        rates.append(pd.DataFrame(lam,columns=genes).assign(cell_id=ids))
        qarr.append(q.astype(np.float32));zarr.append(psiz.astype(np.float32))
    return dict(config=cfg,observed=pd.concat(obs,ignore_index=True),
        counts=pd.concat(count,ignore_index=True)[["cell_id"]+genes],
        hidden=pd.concat(hidden,ignore_index=True),rates=pd.concat(rates,ignore_index=True)[["cell_id"]+genes],
        q_true=np.concatenate(qarr),psi_z_true=np.concatenate(zarr),
        genes=genes,areas=areas,times=times)

def save(result,outdir):
    d=Path(outdir);pub=d/"public";hid=d/"hidden"
    pub.mkdir(parents=True,exist_ok=True);hid.mkdir(parents=True,exist_ok=True)
    result["observed"].to_csv(pub/"observed_cells.csv.gz",index=False)
    result["counts"].to_csv(pub/"counts.csv.gz",index=False)
    pd.DataFrame({"gene":result["genes"]}).to_csv(pub/"genes.csv",index=False)
    result["hidden"].to_csv(hid/"cell_truth.csv.gz",index=False)
    result["rates"].to_csv(hid/"expected_rates.csv.gz",index=False)
    np.savez_compressed(hid/"molecular_truth.npz",q_true=result["q_true"],
        psi_z_true=result["psi_z_true"],cell_id=result["hidden"].cell_id.to_numpy())
    info=dict(version=VERSION,config=asdict(result["config"]),tissue_areas=result["areas"].tolist(),
        cells_per_tissue=result["observed"].groupby("tissue_id").size().tolist(),
        velocity_target="migration-only, in rotated sequencer coordinates",
        population_note="Area-proportional sampling is not a realized birth/death genealogy")
    (d/"manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    sha=hashlib.sha256()
    for f in sorted(pub.iterdir()):
        sha.update(f.name.encode());sha.update(f.read_bytes())
    (d/"public_sha256.txt").write_text(sha.hexdigest()+"\n")
    return d

def main():
    import argparse
    p=argparse.ArgumentParser(description="Frozen cortical benchmark v1.0")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--regime",choices=list("ABCD"),default="B")
    p.add_argument("--population",choices=["growing","constant"],default="growing")
    p.add_argument("--sampling",choices=["uniform_physical","uniform_intrinsic"],default="uniform_physical")
    p.add_argument("--cells-at-t0",type=int,default=1800)
    p.add_argument("--parallel-strength",type=float,default=0)
    p.add_argument("--offaxis-strength",type=float,default=0)
    p.add_argument("--library-log-sd",type=float,default=0)
    p.add_argument("--seed",type=int,default=20261008)
    a=p.parse_args()
    cfg=Config(cells_at_t0=a.cells_at_t0,population=a.population,sampling=a.sampling,
        regime=a.regime,parallel_strength=a.parallel_strength,
        offaxis_strength=a.offaxis_strength,library_log_sd=a.library_log_sd,seed=a.seed)
    print("Saved",save(generate(cfg),a.output_dir),VERSION)

if __name__=="__main__":
    main()
