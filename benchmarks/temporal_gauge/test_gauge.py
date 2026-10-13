"""Guard scientific evaluation invariants rather than implementation details."""
import numpy as np
from scipy.integrate import solve_ivp
from run import calibration, inverse

def test_gauge_score_is_invariant_to_global_affine_scale():
    grid=np.linspace(.05,.95,181);depth=np.linspace(0,1,6001)
    hs=np.array([calibration(depth,depth**1.3+c*depth*(1-depth),grid,1) for c in [0,.1,.2]])
    def score(h):
        mean=h.mean(0);return np.mean((h-mean)**2)/(mean[-1]-mean[0])**2
    assert np.isclose(score(hs),score(1e-4*hs+19),rtol=1e-6)

def test_time_dependent_gauge_preserves_material_correspondence():
    # An exact drifted field can have high gauge error and perfect material flow.
    def h(z,t):return z+.3*t*z*(1-z)
    def hz(z,t):return 1+.3*t*(1-2*z)
    def ht(z,t):return .3*z*(1-z)
    def physical_v(z,t):return .6*z*(1-z)
    a=.3
    truth=solve_ivp(lambda t,z:physical_v(z,t),(0,1),[a],rtol=1e-10,atol=1e-12)
    def transformed_v(t,w):
        z=inverse(h(np.linspace(0,1,10001),t),np.linspace(0,1,10001),w)
        return hz(z,t)*physical_v(z,t)+ht(z,t)
    estimate=solve_ivp(transformed_v,(0,1),[h(a,0)],rtol=1e-9,atol=1e-11)
    assert abs(estimate.y[0,-1]-h(truth.y[0,-1],1))<1e-7

def test_inverse_does_not_hide_unsupported_flow():
    grid=np.linspace(.05,.95,181)
    assert np.isnan(inverse(grid**2,grid,[2.])[0])
