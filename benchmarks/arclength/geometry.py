"""Frozen Regime B/D1 geometry from the 2026-10-12 validation archive.

Oracle geometry-only controls; independent of learned models. No generator edits.
"""
import numpy as np
from scipy.spatial import cKDTree
L=1.25; H0=.17; CURV=3.35; EDGE=.82; CENTER=.70; WIDTH=.26

def hfun(u,t):
    s=np.asarray(u)-.5
    return H0*(1+t*(EDGE+CENTER*np.exp(-.5*(s/WIDTH)**2)))
def hprime(u,t):
    s=np.asarray(u)-.5
    return H0*t*CENTER*np.exp(-.5*(s/WIDTH)**2)*(-s/(WIDTH**2))
def geom(u,z,t):
    u,z=np.broadcast_arrays(np.asarray(u,float),np.asarray(z,float))
    s=u-.5;k=CURV*float(t)**1.15;theta=k*s
    h=hfun(u,t)
    cx=L*s*np.sinc(theta/np.pi)
    cy=.5*L*k*s*s*np.sinc(theta/(2*np.pi))**2
    return np.stack([cx+z*h*np.sin(theta),cy-z*h*np.cos(theta)],axis=-1)
def inverse_z(xy,t):
    """Oracle inverse chart using analytic Newton Jacobian; no true cell IDs."""
    shape=xy.shape[:-1]; p=xy.reshape(-1,2)
    uref=np.linspace(0,1,1401)
    kd=cKDTree(geom(uref,.5,t))
    u=uref[kd.query(p,workers=1)[1]].copy()
    q=geom(u,0.,t)
    theta=CURV*t**1.15*(u-.5)
    normal=np.stack((np.sin(theta),-np.cos(theta)),axis=-1)
    z=np.einsum('ij,ij->i',p-q,normal)/hfun(u,t)
    z=np.clip(z,-.3,1.3)
    for _ in range(22):
        pred=geom(u,z,t);res=pred-p
        k=CURV*t**1.15;theta=k*(u-.5)
        tan=np.stack((np.cos(theta),np.sin(theta)),axis=-1)
        nor=np.stack((np.sin(theta),-np.cos(theta)),axis=-1)
        du=np.einsum('ij,ij->i',res,tan)/(L+z*hfun(u,t)*k)
        dz=(np.einsum('ij,ij->i',res,nor)-z*hprime(u,t)*du)/hfun(u,t)
        u=np.clip(u-du,-.22,1.22);z=np.clip(z-dz,-.8,1.8)
    resid=np.linalg.norm(geom(u,z,t)-p,axis=1)
    return z.reshape(shape),u.reshape(shape),resid.reshape(shape)

def mask_polygon(t,n=1201):
    u=np.linspace(0,1,n)
    v=np.linspace(0,1,129)
    return np.vstack((geom(u,0,t),geom(np.ones_like(v),v,t),
            geom(u[::-1],1,t),geom(np.zeros_like(v),v[::-1],t)))

def velocity(z,t):
    spatial=1.8*z*(1-z)*(1+.45*(2*z-1))
    temporal=1-(1-.1)*(3*t*t-2*t*t*t)
    return spatial*temporal

def true_flow(a,t,n=300):
    z=np.asarray(a,float).copy();step=t/n
    for i in range(n):
        tau=i*step
        k1=velocity(z,tau)
        k2=velocity(z+.5*step*k1,tau+.5*step)
        k3=velocity(z+.5*step*k2,tau+.5*step)
        k4=velocity(z+step*k3,tau+step)
        z+=step*(k1+2*k2+2*k3+k4)/6
    return z
