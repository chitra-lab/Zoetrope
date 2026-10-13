"""Build an isolated copy of the selected runner; never edit production modules."""
import sys
import types
from pathlib import Path
import torch

def build(candidate, weight):
    import flow_stp.stage1.stage1a as production
    source=Path(production.__file__).read_text()
    needle='            total = data_nll\n'
    assert source.count(needle)==1
    source=source.replace(needle,'            total = data_nll + experimental_penalty(model, batch, data)\n')
    module=types.ModuleType('temporal_gauge_experimental_stage1')
    module.__file__=str(Path(__file__).resolve())
    sys.modules[module.__name__]=module
    exec(compile(source,production.__file__,'exec'),module.__dict__)
    if candidate=='tissue_range':
        @torch.no_grad()
        def refresh(model,data,config,device):
            # Spatial-reference endpoints, not matching whole cell quantiles.
            # Valid only when sections have comparable anatomical support.
            model.eval()
            for s in range(data['n_slices']):
                raw=model.coordinate_maps.raw_one_slice(data['coords'][s][data['gauge_idx'][s]].to(device),s)
                low=torch.quantile(raw,config.gauge_lower_quantile)
                high=torch.quantile(raw,config.gauge_upper_quantile)
                if high-low<=1e-5:raise ValueError('Degenerate tissue ruler')
                model.coordinate_maps.gauge_low[s]=low
                model.coordinate_maps.gauge_high[s]=high
        module._refresh_gauge=refresh
    def penalty(model,batch,data):
        if candidate=='tissue_range':return batch['counts'].new_zeros(())
        terms=[]
        for s in range(data['n_slices']):
            # Fixed 64-cell observed subset; no truth. The generator randomizes
            # cell order, but this small pilot subset is not a dense spatial grid.
            idx=data['gauge_idx'][s][:64]
            x=data['coords'][s][idx].detach().clone().requires_grad_(True)
            z=model.coordinate_maps.one_slice(x,s)
            grad=torch.autograd.grad(z.sum(),x,create_graph=True)[0]
            row=data['xy_norm_params'].iloc[s]
            scales=x.new_tensor([row.scale_x,row.scale_y])
            unit=torch.sqrt(torch.mean(scales.square()))
            transform=unit/scales
            g=grad*transform
            norm=torch.sqrt(g.square().sum(1)+1e-8)
            reference=norm.mean().detach().clamp_min(.05)
            if candidate=='eikonal':
                terms.append(((norm/reference-1)**2).mean())
            elif candidate=='normal_hessian':
                dg=torch.autograd.grad(norm.sum(),x,create_graph=True)[0]*transform
                normal=g/norm[:,None]
                terms.append(((dg*normal).sum(1)/reference).square().mean())
            else:raise ValueError(candidate)
        return float(weight)*torch.stack(terms).mean()
    module.experimental_penalty=penalty
    return module
