"""Isolated gauge benchmark. Training sees public files only; evaluation loads hidden truth."""
import argparse
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import contourpy
from scipy.optimize import isotonic_regression
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flow_stp.stage1 import Stage1ARunConfig, run_stage1a
from flow_stp.stage2 import integrate_flow_rk4, estimate_local_dynamics
from flow_stp.simulations.cortical_benchmark.simulation import (
    Config, generate, save, flow, geometry, rotate, expression_parameters, derivatives,
)

def calibration(depth, learned, grid, orientation):
    # Equal-depth bins prevent densely populated regions dominating the curve.
    edges=np.linspace(0,1,31)
    x=[];y=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(depth>=lo)&(depth<hi)
        if mask.sum()>=3:
            x.append(np.median(depth[mask]));y.append(np.median(learned[mask])*orientation)
    iso=isotonic_regression(np.asarray(y)).x
    if grid[0]<x[0] or grid[-1]>x[-1]:
        raise ValueError('Calibration grid exceeds observed common support')
    return np.interp(grid,x,iso)

def inverse(curve,grid,value):
    # No silent endpoint clipping: unsupported predictions are NaN and counted.
    unique,indices=np.unique(curve,return_index=True)
    return np.interp(value,unique,grid[indices],left=np.nan,right=np.nan)

def evaluate(result, dataset, out, steps=72):
    previous=out/'gauge_results.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get('evaluation_version')!=2:
            preserved=out/'evaluation_v1';preserved.mkdir(exist_ok=True)
            for name in ['gauge_results.json','evaluation_arrays.npz','calibration.csv','calibration.png']:
                if (out/name).exists():shutil.copy2(out/name,preserved/name)
    hidden=pd.read_csv(dataset/'hidden/cell_truth.csv.gz')
    observed=pd.read_csv(dataset/'public/observed_cells.csv.gz')
    cfg=Config(**json.loads((dataset/'manifest.json').read_text())['config'])
    frame=observed.merge(hidden,on='cell_id',validate='one_to_one')
    frame['learned']=result.cells.z_stage1a.to_numpy()
    splits=np.load(out/'tables/fixed_splits_diagnostics_and_gauge.npz')
    grid=np.linspace(.05,.95,181)
    times=result.normalized_times
    correlations=[spearmanr(g.z_true,g.learned).statistic for _,g in frame.groupby('tissue_id')]
    orientation=1 if np.mean(correlations)>=0 else -1
    curves=[];anatomy=[]
    for s,g in frame.groupby('tissue_id'):
        train=splits[f'train_{s}'] if f'train_{s}' in splits else splits[f'train_idx_{s}']
        val=splits[f'validation_{s}']
        h=calibration(g.z_true.to_numpy()[train],g.learned.to_numpy()[train],grid,orientation)
        curves.append(h)
        anatomy.append((g.iloc[val].z_true.to_numpy(),orientation*g.iloc[val].learned.to_numpy()))
    curves=np.asarray(curves);mean=curves.mean(0);span=mean[-1]-mean[0]
    if span<1e-6: raise ValueError('Collapsed pooled calibration')
    scores=dict(evaluation_version=2,orientation=orientation,calibration_span=float(span),
        gauge_distortion=float(np.mean((curves-mean)**2)/span**2),
        ordering_spearman_mean=float(orientation*np.mean(correlations)),
        ordering_spearman_min=float(orientation*np.min(correlations) if orientation==1 else -np.max(correlations)))
    errors=[];coverage=[]
    for depth,learned in anatomy:
        pred=inverse(mean,grid,learned);m=np.isfinite(pred)&(depth>=grid[0])&(depth<=grid[-1])
        coverage.append(m.mean());errors.append(np.mean((pred[m]-depth[m])**2))
    scores['heldout_anatomy_rmse']=float(np.sqrt(np.mean(errors)))
    scores['heldout_anatomy_coverage']=float(np.mean(coverage))
    # Production summary is cell-weighted. Also report age-balanced likelihood.
    counts=pd.read_csv(dataset/'public/counts.csv.gz').set_index('cell_id')
    import flow_stp.stage1.stage1a as selected
    matrices=[torch.tensor(counts.loc[g.cell_id].to_numpy(),dtype=torch.float32) for _,g in observed.groupby('tissue_id')]
    offsets=selected._build_library_offsets(matrices,True)
    nll=[]
    for s,g in frame.groupby('tissue_id'):
        val=splits[f'validation_{s}'];row=result.xy_norm_params.iloc[s]
        xy=g[['x','y']].to_numpy()[val]
        xy=(xy-np.array([row.center_x,row.center_y]))/np.array([row.scale_x,row.scale_y])
        with torch.no_grad():
            output=result.model(coords=torch.tensor(xy,dtype=torch.float32),slice_idx=torch.full((len(val),),s,dtype=torch.long),
                time=torch.full((len(val),),float(times[s])),log_library_size=offsets[s][val])
            nll.append(float(result.model.poisson_nll(matrices[s][val],output['log_mu'])))
    scores['balanced_validation_poisson_nll']=float(np.mean(nll))
    # Fixed common reference depth; no optimization uses these anchors.
    anchors=np.linspace(.10,.75,51)
    initial=np.interp(anchors,grid,curves[0])*orientation
    transverse=np.array([.1,.3,.5,.7,.9])
    us,zs=np.meshgrid(transverse,anchors,indexing='ij')
    sx,sy,*_=geometry(us.ravel(),zs.ravel(),0.)
    source=frame[frame.tissue_id==0];sx,sy=rotate(sx,sy,source.rotation_deg.iloc[0])
    sx+=source.translation_x.iloc[0];sy+=source.translation_y.iloc[0]
    nr=result.xy_norm_params.iloc[0]
    sourcecoords=np.column_stack(((sx-nr.center_x)/nr.scale_x,(sy-nr.center_y)/nr.scale_y))
    with torch.no_grad():sourcez=result.model.coordinate_maps(torch.tensor(sourcecoords,dtype=torch.float32),0).numpy().ravel()
    full_trajectory=integrate_flow_rk4(result.model.fisher_flat,torch.tensor(np.r_[initial,sourcez],dtype=torch.float32),n_steps=steps)
    trajectory=replace(full_trajectory,time=full_trajectory.time[:,:len(initial)],
        position=full_trajectory.position[:,:len(initial)],velocity=full_trajectory.velocity[:,:len(initial)])
    physical_positions=full_trajectory.position[:,len(initial):].numpy().reshape(steps+1,len(transverse),len(anchors))
    positions=trajectory.position.numpy()*orientation
    node_times=trajectory.time[:,0].numpy()
    truth=np.asarray([flow(anchors,float(t)) for t in node_times])
    fixed=np.asarray([inverse(mean,grid,p) for p in positions])
    # Only observed ages have directly estimated calibration curves.
    indices=np.rint(np.asarray(times)*steps).astype(int)
    correspondence=np.asarray([inverse(h,grid,positions[i]) for h,i in zip(curves,indices)])
    true_at_ages=truth[indices]
    for label,pred,target in [('fixed_gauge_flow',fixed,truth),('material_flow',correspondence,true_at_ages)]:
        valid=np.isfinite(pred)
        scores[label+'_coverage']=float(valid.mean())
        scores[label+'_rmse']=float(np.sqrt(np.mean((pred[valid]-target[valid])**2)))
        denom=np.mean((target-np.broadcast_to(anchors,target.shape))**2)
        scores[label+'_nrmse']=float(np.sqrt(np.mean((pred[valid]-target[valid])**2)/denom))
    # Closest point on the ACTUAL learned target level set in registered physical
    # geometry. Truth is used to build an evaluation mesh, never for fitting.
    contour_error=[];motion=[]
    for j,t in enumerate(times):
        if j==0:continue
        ug=np.linspace(0,1,257);zg=np.linspace(0,1,193)
        uu,zz=np.meshgrid(ug,zg);px,py,*_=geometry(uu,zz,float(t))
        g=frame[frame.tissue_id==j];angle=g.rotation_deg.iloc[0]
        ox,oy=rotate(px,py,angle);ox+=g.translation_x.iloc[0];oy+=g.translation_y.iloc[0]
        norm=result.xy_norm_params.iloc[j]
        coord=np.column_stack(((ox.ravel()-norm.center_x)/norm.scale_x,(oy.ravel()-norm.center_y)/norm.scale_y))
        with torch.no_grad():
            learned=result.model.coordinate_maps(torch.tensor(coord,dtype=torch.float32),j).numpy().reshape(zz.shape)
        generator=contourpy.contour_generator(x=ug,y=zg,z=learned)
        for ui,u in enumerate(transverse):
            contours=[generator.lines(float(p)) for p in physical_positions[indices[j],ui]]
            xt,yt,*_=geometry(u,true_at_ages[j],float(t))
            x0,y0,*_=geometry(u,anchors,0.)
            for k,lines in enumerate(contours):
                if not lines:continue
                distances=[]
                for line in lines:
                    xp,yp,*_=geometry(line[:,0],line[:,1],float(t))
                    pts=np.column_stack((xp,yp));v=np.diff(pts,axis=0);delta=np.array([xt[k],yt[k]])-pts[:-1]
                    fraction=np.clip(np.sum(v*delta,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-15),0,1)
                    distances.append(np.min(np.sum((delta-fraction[:,None]*v)**2,axis=1)))
                contour_error.append(min(distances))
                motion.append((xt[k]-x0[k])**2+(yt[k]-y0[k])**2)
    scores['physical_contour_oracle_nrmse']=float(np.sqrt(np.mean(contour_error)/np.mean(motion)))
    scores['physical_contour_coverage']=len(contour_error)/((len(times)-1)*len(transverse)*len(anchors))
    # rho = integral of the orthogonal molecular speed along frozen RK4 nodes.
    rho=[];true_speed=[];params=expression_parameters(cfg.seed)
    for t,p in zip(node_times,trajectory.position):
        local=estimate_local_dynamics(result.model.fisher_flat,p,torch.full_like(p,float(t)))
        rho.append(local.residual_speed.detach().numpy().ravel())
        q,_,_=derivatives(anchors,np.full_like(anchors,.5),float(t),cfg,params)
        true_speed.append(np.linalg.norm(q,axis=1))
    integral=np.trapezoid(np.asarray(rho),node_times,axis=0)
    integral_true=np.trapezoid(np.asarray(true_speed),node_times,axis=0)
    scores['rho_absolute_rmse']=float(np.sqrt(np.mean((integral-integral_true)**2)))
    scores['rho_relative_rmse']=float(np.sqrt(np.mean((integral-integral_true)**2)/np.mean(integral_true**2))) if np.mean(integral_true**2)>1e-12 else None
    scores['rho_mean']=float(integral.mean());scores['rho_true_mean']=float(integral_true.mean())
    scores['rho_cv']=float(integral.std()/max(integral.mean(),1e-12))
    scores['stage1_summary']=result.summary
    # Calibration uncertainty under within-tissue resampling. Ages are fixed.
    rng=np.random.default_rng(841)
    boot=[]
    for b in range(100):
        hs=[]
        for _,g in frame.groupby('tissue_id'):
            ix=rng.integers(0,len(g),len(g))
            hs.append(calibration(g.z_true.to_numpy()[ix],g.learned.to_numpy()[ix],grid,orientation))
        hs=np.asarray(hs);hm=hs.mean(0);r=hm[-1]-hm[0]
        boot.append(np.mean((hs-hm)**2)/r**2)
    scores['gauge_bootstrap_95ci']=np.quantile(boot,[.025,.975]).tolist()
    pd.DataFrame(curves.T,index=grid,columns=[f't={t:.3f}' for t in times]).to_csv(out/'calibration.csv',index_label='true_depth')
    np.savez_compressed(out/'evaluation_arrays.npz',grid=grid,calibration=curves,times=times,
        anchors=anchors,node_times=node_times,learned_flow=positions,true_flow=truth,
        fixed_gauge_flow=fixed,material_flow=correspondence,rho=integral,rho_true=integral_true)
    fig,ax=plt.subplots(figsize=(7,5))
    for t,h in zip(times,curves): ax.plot(grid,(h-mean[0])/span,label=f'{t:.2f}')
    ax.set(xlabel='True anatomical depth z_true',ylabel='Learned calibration (pooled scale)',title=out.name)
    ax.legend(title='Age',ncol=2);fig.tight_layout();fig.savefig(out/'calibration.png',dpi=160);plt.close(fig)
    (out/'gauge_results.json').write_text(json.dumps(scores,indent=2))
    print(json.dumps({k:v for k,v in scores.items() if k!='stage1_summary'},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260622)
    p.add_argument('--epochs',type=int,default=1500)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--cells',type=int,default=1800)
    p.add_argument('--regime',default='B')
    p.add_argument('--data-seed',type=int,default=20261008)
    p.add_argument('--steps',type=int,default=72)
    p.add_argument('--candidate',default='baseline',choices=['baseline','tissue_range','eikonal','normal_hessian'])
    p.add_argument('--weight',type=float,default=.01)
    p.add_argument('--ages',type=int,default=10,choices=[4,10])
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--selected-tissues',default=None,help='Optional four original tissue indices, including 0 and 9')
    a=p.parse_args();torch.set_num_threads(2)
    if not a.dataset.exists():
        simulated=generate(Config(cells_at_t0=a.cells,regime=a.regime,seed=a.data_seed))
        if a.ages==4:
            selected=[int(x) for x in a.selected_tissues.split(',')] if a.selected_tissues else [0,3,6,9]
            if len(selected)!=4 or selected!=sorted(set(selected)) or selected[0]!=0 or selected[-1]!=9:
                raise ValueError('Choose four distinct increasing original tissues including 0 and 9')
            mask=simulated['observed'].tissue_id.isin(selected).to_numpy()
            for key in ['observed','hidden','counts','rates']:
                simulated[key]=simulated[key].loc[mask].reset_index(drop=True)
            simulated['observed']['tissue_id']=simulated['observed'].tissue_id.map({s:i for i,s in enumerate(selected)})
            for key in ['q_true','psi_z_true']:simulated[key]=simulated[key][mask]
            for key in ['times','areas']:simulated[key]=simulated[key][selected]
        staging=a.dataset.with_name(a.dataset.name+f'.building_{os.getpid()}')
        save(simulated,staging)
        if a.ages==4:
            manifest=json.loads((staging/'manifest.json').read_text());manifest['selected_original_tissues']=selected
            (staging/'manifest.json').write_text(json.dumps(manifest,indent=2))
        staging.rename(a.dataset)
    if not (a.dataset/'public_sha256.txt').exists():
        raise RuntimeError('Dataset is incomplete; finish generation before training')
    if a.generate_only:return
    observed=pd.read_csv(a.dataset/'public/observed_cells.csv.gz')
    counts=pd.read_csv(a.dataset/'public/counts.csv.gz').set_index('cell_id')
    genes=counts.columns.tolist(); cs=[];xy=[];panel=[]
    for s,g in observed.groupby('tissue_id'):
        cs.append(counts.loc[g.cell_id].to_numpy());xy.append(g[['x','y']].to_numpy())
        panel.append(dict(slice_id=s,time=g.time.iloc[0]))
    config=Stage1ARunConfig(output_dir=a.output,epochs=a.epochs,batch_size_per_slice=a.batch,
        seed=a.seed,print_every=100,checkpoint_every=500,resume=True)
    started=time.perf_counter()
    runner=run_stage1a
    if a.candidate!='baseline':
        from experimental_stage1 import build
        runner=build(a.candidate,a.weight).run_stage1a
    result=runner(counts_by_slice=cs,spatial_by_slice=xy,panel_df=pd.DataFrame(panel),gene_names=genes,config=config)
    print('Training seconds',time.perf_counter()-started,flush=True)
    evaluate(result,a.dataset,a.output,a.steps)
    (a.output/'experiment_settings.json').write_text(json.dumps(vars(a),default=str,indent=2))

if __name__=='__main__':main()
