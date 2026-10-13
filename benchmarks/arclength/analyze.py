"""Paired metrics, cell-cluster bootstrap, depth/time plots and failure causes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import geom, mask_polygon, true_flow
from run import build_field, ideal_endpoint


def read_endpoints(path):
    path = Path(path)
    return pd.read_csv(path if path.exists() else path.with_suffix(path.suffix+'.gz'))


def check_ideal_convergence(output):
    rng = np.random.default_rng(20261013)
    a,u = rng.uniform(0,1,100),rng.uniform(0,1,100)
    errors = []
    for j in range(1,10):
        t = j/9
        z = true_flow(a,t)
        for start in range(0,100,20):
            sl = slice(start,start+20)
            p = ideal_endpoint(u[sl],z[sl],t)
            q = ideal_endpoint(u[sl],z[sl],t,n_points=32769)
            errors.extend(np.linalg.norm(p-q,axis=1))
    result = dict(n_points_primary=8193,n_points_check=32769,n_endpoints=len(errors),
                  max_endpoint_difference=float(np.max(errors)),
                  rmse_endpoint_difference=float(np.sqrt(np.mean(np.square(errors)))))
    (output/'ideal_convergence.json').write_text(json.dumps(result,indent=2))


def metric(frame, kind):
    return float(np.sqrt(frame[kind+'_error_squared'].sum()/frame.motion_squared.sum()))


def causes(frame):
    failed = frame.loc[~frame.success,'failure'].fillna('').astype(str)
    labels = np.select([failed.str.contains('has 0 contours'),
                        failed.str.contains(r'has \d+ contours'),
                        failed.str.contains('outside'),
                        failed.str.contains('gradient'),
                        failed.str.contains('orientation')],
                       ['missing contour','fragmented contour','source outside domain',
                        'degenerate gradient','orientation'],default='other')
    return pd.Series(labels).value_counts().to_dict()


def bootstrap(frame, n=3000, seed=732):
    columns = ['motion_squared','arclength_error_squared','oracle_error_squared',
               'ideal_error_squared','numerical_error_squared','success']
    sums = frame.assign(motion_squared=frame.motion_squared.where(frame.success,0)).groupby('source_index')[columns].sum()
    matrix = sums.to_numpy(float)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0,len(matrix),(n,len(matrix)))
    totals = matrix[selected].sum(axis=1)
    result = {}
    for i,kind in enumerate(['arclength','oracle','ideal','numerical'],1):
        draws = np.sqrt(totals[:,i]/totals[:,0])
        result[kind+'_nrmse_95ci'] = np.quantile(draws,[.025,.975]).tolist()
    result['coverage_95ci'] = np.quantile(totals[:,-1]/len(frame),[.025,.975]).tolist()
    return result


def paired_bootstrap(left, right, n=3000, seed=732):
    common = left.success.to_numpy() & right.success.to_numpy()
    l,r = left[common],right[common]
    grouped = pd.DataFrame({'source_index':l.source_index,
                            'motion':l.motion_squared,'left':l.arclength_error_squared,
                            'right':r.arclength_error_squared}).groupby('source_index').sum().to_numpy()
    rng = np.random.default_rng(seed)
    totals = grouped[rng.integers(0,len(grouped),(n,len(grouped)))].sum(axis=1)
    draws = np.sqrt(totals[:,2]/totals[:,0])-np.sqrt(totals[:,1]/totals[:,0])
    return dict(n_paired=int(common.sum()),baseline_nrmse=metric(l,'arclength'),
                improved_nrmse=metric(r,'arclength'),
                ideal_paired_nrmse=metric(r,'ideal'),
                nrmse_difference_95ci=np.quantile(draws,[.025,.975]).tolist())


def conditional_table(frames, variable):
    common = np.logical_and.reduce([f.success.to_numpy() for f in frames.values()])
    rows = []
    bins = [0,.01,.02,.05,.1,.25,.5,.75,.9,.95,.98,.99,1]
    for variant,frame in frames.items():
        groups = frame[variable] if variable=='time' else pd.cut(frame[variable],bins,include_lowest=True)
        for label,indices in frame.groupby(groups,observed=True).groups.items():
            requested = frame.loc[indices]
            paired = requested[common[indices]]
            own = requested[requested.success]
            display = f'{float(label):.2f}' if variable=='time' else f'{max(0,label.left):g}–{label.right:g}'
            row = dict(variant=variant,variable=variable,bin=display,n_requested=len(requested),
                       n_success=len(own),coverage=len(own)/len(requested),n_paired=len(paired))
            for kind in ['arclength','oracle','ideal','numerical']:
                row[kind+'_paired_nrmse'] = metric(paired,kind) if len(paired) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_conditions(frames, cohort, output):
    fig,axs = plt.subplots(2,3,figsize=(15,7),layout='constrained')
    labels = {'baseline':'Masked grid','polygon_cached':'Polygon + cache'}
    for j,variable in enumerate(['initial_a','target_depth','time']):
        table = conditional_table(frames,variable)
        table.to_csv(output/f'{cohort}_by_{variable}.csv',index=False)
        for variant,group in table.groupby('variant',sort=False):
            x = np.arange(len(group))
            axs[0,j].plot(x,group.coverage,'o-',label=labels[variant])
            axs[1,j].plot(x,group.arclength_paired_nrmse,'o-',label=labels[variant])
        group = table[table.variant=='polygon_cached']
        axs[1,j].plot(np.arange(len(group)),group.ideal_paired_nrmse,'k:',label='Continuous arclength')
        for ax in axs[:,j]:
            ax.set_xticks(np.arange(len(group)),group.bin,rotation=60,fontsize=7)
            ax.set_xlabel({'initial_a':'Initial anatomical depth','target_depth':'Target anatomical depth','time':'Developmental time'}[variable])
            ax.grid(alpha=.2)
        axs[0,j].set_ylim(0,1.04)
    axs[0,0].set_ylabel('Full-cohort coverage')
    axs[1,0].set_ylabel('Paired displacement NRMSE')
    axs[0,0].legend(fontsize=9)
    axs[1,0].legend(fontsize=9)
    fig.suptitle(f'{cohort.title()} cohort: geometry-only control; paired errors use identical successes')
    fig.savefig(output/f'{cohort}_depth_time.svg')
    fig.savefig(output/f'{cohort}_depth_time.png',dpi=160)
    plt.close(fig)


def boundary_plot(output):
    t,z = 1.,.996
    masked,_ = build_field(t,500,420,'baseline')
    clipped,_ = build_field(t,500,420,'polygon_cached')
    fig,axs = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax in axs:
        p = mask_polygon(t)
        ax.plot(p[:,0],p[:,1],color='.5',lw=1,label='Tissue polygon')
        exact = geom(np.linspace(0,1,2001),z,t)
        ax.plot(exact[:,0],exact[:,1],'k--',lw=2,label='Exact contour')
        ax.set_aspect('equal')
        ax.set(xlabel='Physical x',ylabel='Physical y')
    for i,c in enumerate(masked.contours(z)):
        axs[0].plot(c.xy[:,0],c.xy[:,1],color='tab:red',lw=2,label='Masked fragments' if i==0 else None)
    c = clipped.unique_contour(z)
    axs[1].plot(c.xy[:,0],c.xy[:,1],color='tab:blue',label='Polygon-clipped contour')
    axs[0].set_title(f'Mask mode: {len(masked.contours(z))} fragments')
    axs[1].set_title('Polygon mode: one complete open contour')
    for ax in axs:
        ax.legend(fontsize=8)
    fig.savefig(output/'boundary_failure.svg')
    fig.savefig(output/'boundary_failure.png',dpi=160)
    plt.close(fig)


def run(output):
    output = Path(output)
    summaries = []
    for path in sorted(output.glob('*_summary.json')):
        summary = json.loads(path.read_text())
        frame = read_endpoints(str(path).replace('_summary.json','_endpoints.csv'))
        summary.update(bootstrap(frame),failure_causes=causes(frame))
        summaries.append(summary)
    (output/'bootstrap_summaries.json').write_text(json.dumps(summaries,indent=2))
    scalar = [{k:v for k,v in s.items() if not isinstance(v,(dict,list))} for s in summaries]
    pd.DataFrame(scalar).to_csv(output/'comparison.csv',index=False)
    paired = {}
    for cohort in ['original','heldout']:
        paths = [output/f'{cohort}_{v}_500x420_endpoints.csv' for v in ['baseline','polygon_cached']]
        if not all(p.exists() or p.with_suffix(p.suffix+'.gz').exists() for p in paths):
            continue
        frames = {variant:read_endpoints(output/f'{cohort}_{variant}_500x420_endpoints.csv')
                  for variant in ['baseline','polygon_cached']}
        for f in frames.values():
            f.sort_values(['time','source_index'],inplace=True)
            f.reset_index(drop=True,inplace=True)
        paired[cohort] = paired_bootstrap(frames['baseline'],frames['polygon_cached'])
        plot_conditions(frames,cohort,output)
    (output/'paired_metrics.json').write_text(json.dumps(paired,indent=2))
    # Compare development variants on one joint successful set.
    development = {s['variant']:read_endpoints(output/f"development_{s['variant']}_320x280_endpoints.csv")
                   for s in summaries if s['cohort']=='development'}
    if development:
        common = np.logical_and.reduce([f.success.to_numpy() for f in development.values()])
        records = [dict(variant=v,n_paired=int(common.sum()),arclength_nrmse=metric(f[common],'arclength'),
                        numerical_nrmse=metric(f[common],'numerical')) for v,f in development.items()]
        pd.DataFrame(records).to_csv(output/'development_paired.csv',index=False)
    boundary_plot(output)
    check_ideal_convergence(output)
    # Matplotlib includes trailing spaces inside SVG path strings. Newlines
    # preserve XML/path token separation while keeping committed output tidy.
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(json.dumps(paired,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='results/arclength')
    run(parser.parse_args().output)
