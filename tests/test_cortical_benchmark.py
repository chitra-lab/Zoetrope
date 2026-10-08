from dataclasses import replace
import numpy as np
import pytest
from flow_stp.simulations.cortical_benchmark import Config, generate, geometry, velocity, flow, area, rate
from flow_stp.simulations.cortical_benchmark.simulation import expression_parameters, derivatives, jacobian

BASE = Config(cells_at_t0=24)

def test_frozen_geometry():
    u=np.linspace(.05,.95,31)
    x,y,_,_=geometry(u,np.zeros_like(u),0.)
    np.testing.assert_allclose(y,0.,atol=1e-10)
    assert area(1)>area(0)
    assert np.min(jacobian(u[:,None],np.linspace(0,1,21)[None,:],1.))>0
    assert np.all(np.diff(flow(np.linspace(0,1,91),1.))>0)
    assert max(velocity(np.linspace(0,1,100),1.)) < .11*max(velocity(np.linspace(0,1,100),0.))

def test_regime_b_exact_fisher_recovery():
    a=np.linspace(.1,.9,61);u=np.full_like(a,.5)
    q,pz,vf=derivatives(a,u,.5,BASE,expression_parameters(BASE.seed))
    cos=np.abs(np.sum(q*pz,axis=1))/(np.linalg.norm(q,axis=1)*np.linalg.norm(pz,axis=1)+1e-10)
    assert np.max(cos)<1e-5
    np.testing.assert_allclose(vf,velocity(flow(a,.5),.5),atol=1e-6)

def test_regime_a_has_no_intrinsic_change():
    cfg=replace(BASE,regime="A")
    a=np.linspace(.1,.9,23)
    q,_,_=derivatives(a,np.full_like(a,.5),.4,cfg,expression_parameters(cfg.seed))
    assert np.max(np.abs(q))<1e-7

def test_regime_c_bias():
    cfg=replace(BASE,regime="C",parallel_strength=.5)
    a=np.linspace(.1,.9,41)
    q,pz,vf=derivatives(a,np.full_like(a,.5),.5,cfg,expression_parameters(cfg.seed))
    assert np.max(np.abs(np.sum(q*pz,axis=1)))>.001
    assert np.max(np.abs(vf-velocity(flow(a,.5),.5)))>.001

def test_regime_d_transverse_expression():
    cfg=replace(BASE,regime="D",offaxis_strength=.5)
    a=np.array([.35,.35])
    lam=rate(a,np.array([.1,.35]),.5,cfg)
    assert np.max(np.abs(lam[0]-lam[1]))>.01

def test_sampling_and_poisson_and_repeatability():
    a=generate(BASE);b=generate(BASE)
    assert a["observed"].equals(b["observed"])
    assert a["counts"].equals(b["counts"])
    assert len(a["observed"])>10*BASE.cells_at_t0
    assert len(a["observed"])==len(a["hidden"])==len(a["counts"])
    assert "z_true" not in a["observed"].columns
    Y=a["counts"].iloc[:,1:].to_numpy()
    assert np.issubdtype(Y.dtype,np.integer) and (Y>=0).all()
    f=generate(replace(BASE,population="constant"))
    assert len(f["observed"])==10*BASE.cells_at_t0

def test_fair_regime_comparison_same_coordinates():
    a=generate(BASE)
    c=generate(replace(BASE,regime="C",parallel_strength=.25))
    np.testing.assert_array_equal(a["observed"][["x","y"]],c["observed"][["x","y"]])
    assert not a["counts"].equals(c["counts"])

def test_reject_invalid_controls():
    with pytest.raises(ValueError):
        generate(replace(BASE,parallel_strength=.2))
