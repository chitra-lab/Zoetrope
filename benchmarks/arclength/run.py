"""Geometry-only tournament; run each variant in a fresh process for peak RSS."""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path as PolygonPath
from geometry import geom, inverse_z, mask_polygon, hfun, hprime, true_flow, L, CURV
from flow_stp.tissue_coordinates import (
    AnatomicalContourField, reconstruct_by_arclength, oracle_nearest_contour_point,
)

HERE = Path(__file__).parent


def ideal_endpoint(u0, depth, t, n_points=8193):
    u = np.linspace(0, 1, n_points)
    z = depth[:, None]
    speed = np.hypot(L + z*hfun(u,t)*CURV*t**1.15, z*hprime(u,t))
    arc = np.column_stack((np.zeros(len(z)), cumulative_trapezoid(speed,u,axis=1)))
    fractions = arc / arc[:, -1:]
    mapped = np.array([np.interp(s, fractions[i], u) for i,s in enumerate(u0)])
    return geom(mapped,depth,t)


def build_field(t, nx, ny, variant):
    poly = mask_polygon(t)
    x = np.linspace(poly[:,0].min()-.018,poly[:,0].max()+.018,nx)
    y = np.linspace(poly[:,1].min()-.018,poly[:,1].max()+.018,ny)
    xx,yy = np.meshgrid(x,y)
    xy = np.stack((xx,yy),axis=-1)
    z,u,residual = inverse_z(xy,t)
    mask = PolygonPath(poly).contains_points(xy.reshape(-1,2)).reshape(xx.shape)
    kwargs = dict(tissue_mask=mask,cache_size=0)
    if variant.startswith('polygon'):
        kwargs = dict(tissue_polygon=poly,cache_size=0)
    if variant == 'corner':
        kwargs['corner_mask'] = True
    if variant == 'polygon_refined':
        # Subdivide the existing bilinear field; this does not access exact
        # anatomy between grid nodes or improve the original node values.
        interpolator = RegularGridInterpolator((y,x),z)
        x = np.linspace(x[0],x[-1],2*nx-1)
        y = np.linspace(y[0],y[-1],2*ny-1)
        xx,yy = np.meshgrid(x,y)
        z = interpolator(np.stack((yy,xx),axis=-1))
    if variant == 'polygon_cached':
        kwargs['cache_size'] = 512
    field = AnatomicalContourField(x,y,z,**kwargs)
    return field, {'time':t,'nx':nx,'ny':ny,'contour_nx':len(x),'contour_ny':len(y),
                   'mask_pixels':int(mask.sum()),
                   'inverse_residual_max_inside':float(residual[mask].max())}


def run(args):
    start = time.perf_counter()
    if args.cohort == 'original':
        cohort = pd.read_csv(args.source_cohort)
    else:
        rng = np.random.default_rng(20261013 if args.cohort == 'development' else 20261014)
        n = 100 if args.cohort == 'development' else 300
        cohort = pd.DataFrame({'source_index':np.arange(n),'source_id':np.arange(n),
                               'initial_a':rng.uniform(0,1,n),'u_true':rng.uniform(0,1,n)})
    a,u = cohort.initial_a.to_numpy(),cohort.u_true.to_numpy()
    source_xy = geom(u,a,0)
    source,meta0 = build_field(0,args.nx,args.ny,args.variant)
    metadata = [meta0]
    build_seconds = time.perf_counter()-start
    records = []
    prediction_seconds = 0.
    for j in range(1,10):
        t = j/9
        tick = time.perf_counter()
        target,meta = build_field(t,args.nx,args.ny,args.variant)
        build_seconds += time.perf_counter()-tick
        metadata.append(meta)
        depths = true_flow(a,t)
        truth = geom(u,depths,t)
        ideal = ideal_endpoint(u,depths,t)
        tick = time.perf_counter()
        for k in range(len(a)):
            row = dict(variant=args.variant,cohort=args.cohort,source_index=k,
                       time=t,initial_a=a[k],u_true=u[k],target_depth=depths[k],
                       motion_squared=float(np.sum((truth[k]-source_xy[k])**2)),
                       ideal_error_squared=float(np.sum((ideal[k]-truth[k])**2)),
                       success=False,failure='')
            try:
                pred = reconstruct_by_arclength(source_xy[k],a[k],depths[k],source,target)
                # Truth first enters the evaluation only AFTER canonical prediction.
                oracle,_ = oracle_nearest_contour_point(truth[k],depths[k],target)
                row.update(success=True,predicted_x=pred.xy[0],predicted_y=pred.xy[1],
                           truth_x=truth[k,0],truth_y=truth[k,1],ideal_x=ideal[k,0],ideal_y=ideal[k,1],
                           arclength_error_squared=float(np.sum((pred.xy-truth[k])**2)),
                           oracle_error_squared=float(np.sum((oracle-truth[k])**2)),
                           numerical_error_squared=float(np.sum((pred.xy-ideal[k])**2)),
                           source_contour_distance=pred.source_contour_distance)
            except (ValueError,IndexError) as exc:
                row['failure'] = str(exc)
            records.append(row)
        prediction_seconds += time.perf_counter()-tick
        print(args.variant,args.cohort,j,sum(r['success'] for r in records[-len(a):]),flush=True)
    frame = pd.DataFrame(records)
    valid = frame[frame.success]
    summary = dict(variant=args.variant,cohort=args.cohort,nx=args.nx,ny=args.ny,
                   n_requested=len(frame),n_success=len(valid),coverage=len(valid)/len(frame),
                   build_seconds=build_seconds,prediction_seconds=prediction_seconds,
                   total_seconds=time.perf_counter()-start,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   failures=frame.loc[~frame.success,'failure'].value_counts().to_dict())
    for name in ['arclength','oracle','ideal','numerical']:
        col = name+'_error_squared'
        summary[name+'_rmse'] = float(np.sqrt(valid[col].mean()))
        summary[name+'_nrmse'] = float(np.sqrt(valid[col].sum()/valid.motion_squared.sum()))
    summary['ideal_full_cohort_nrmse'] = float(np.sqrt(frame.ideal_error_squared.sum()/frame.motion_squared.sum()))
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    prefix = f'{args.cohort}_{args.variant}_{args.nx}x{args.ny}'
    frame.to_csv(out/(prefix+'_endpoints.csv'),index=False)
    pd.DataFrame(metadata).to_csv(out/(prefix+'_grids.csv'),index=False)
    (out/(prefix+'_summary.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='failures'},indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant',default='baseline',choices=['baseline','corner','polygon','polygon_refined','polygon_cached'])
    p.add_argument('--cohort',default='heldout',choices=['original','development','heldout'])
    p.add_argument('--source-cohort',type=Path,default=HERE/'original_cohort.csv',
                   help='Optional recovered original cohort CSV, supplied locally by its owner.')
    p.add_argument('--nx',type=int,default=500)
    p.add_argument('--ny',type=int,default=420)
    p.add_argument('--output',default='results/arclength')
    run(p.parse_args())
