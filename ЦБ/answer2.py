# -*- coding: utf-8 -*-
"""Ответ 2 — правило КПМ ЦБ на сетке 25 б.п. от текущей ставки, при r_bar=2.23, w_cc=0.75."""
import os, numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
P = os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(P,"prep.csv"),encoding='utf-8'); T=len(d)
def hp(x,lam=1600.):
    x=np.asarray(x,float);n=len(x);I=np.eye(n);D=np.zeros((n-2,n))
    for t in range(n-2): D[t,t]=1.;D[t,t+1]=-2.;D[t,t+2]=1.
    tr=np.linalg.solve(I+lam*D.T@D,x);return tr,x-tr
Lg=lambda v,k: np.r_[np.full(k,np.nan),np.asarray(v,float)[:-k]]
PIE=d['pi'].values; PIE4=d['pi4'].values; RS=d['i'].values
Ytr,Yg=hp(d['y'].values)
G1,G2,G3=0.75,1.5,0.5; TGT=100*np.log(1.04); RBAR=2.23
i0=RS[-1]; grid=i0+np.arange(-8,9)*0.25
def snap(x): return grid[int(np.argmin(np.abs(grid-x)))]
def rule(p4e,g2=G2,g3=G3,rbar=RBAR,tgt=TGT):
    return G1*i0+(1-G1)*(rbar+p4e+g2*(p4e-tgt)+g3*Yg[-1])
print("="*80)
print(" ОТВЕТ 2 — правило КПМ ЦБ (gamma1=0.75, gamma2=1.5, gamma3=0.5)")
print("="*80)
print(f"  RS(239Q2)={i0:.4f}  PIE4={PIE4[-1]:.2f}  Y_GAP={Yg[-1]:+.2f}  RR_EQ={RBAR:.2f}")
print(f"  RS_NEUTRAL = RR_EQ + PIE_TAR = {RBAR+TGT:.2f}%   |  цель = 100*ln(1.04) = {TGT:.3f}")
print(f"  сетка: шаг кратен 25 б.п. от {i0:.4f}\n")
ok=~np.isnan(PIE4)&~np.isnan(Lg(PIE4,1))&~np.isnan(Lg(PIE4,2))
XA=np.column_stack([np.ones(T),Lg(PIE4,1),Lg(PIE4,2)])[ok]
bA=np.linalg.lstsq(XA,PIE4[ok],rcond=None)[0]; p1,p2=PIE4[-1],PIE4[-2]
for _ in range(3): p1,p2=bA[0]+bA[1]*p1+bA[2]*p2,p1
rows=[("текущая PIE4 (наивный E3)",PIE4[-1]),("прогноз AR(2) на 3 кв.",p1),
      ("ожидания заякорены на цели",TGT),("прогноз нашей модели (240Q2)",3.31),
      ("прогноз самой КПМ (отвергнут)",-2.77)]
print("  чем брать E3_PIE4          знач.   сырая RS   на сетке")
for nm,v in rows:
    r=rule(v); print(f"  {nm:28s} {v:6.2f}   {r:7.2f}%   {snap(r):6.2f}%")
core=[rule(v) for nm,v in rows[:3]]
print(f"\n  Ядро (без модельных прогнозов): {min(core):.2f}–{max(core):.2f}%  ->  на сетке {snap(np.mean(core)):.2f}%")
print("\n  Чувствительность ядра:")
for nm,kw in [("monetary_hard gamma2=1.9",dict(g2=1.9)),("gamma3=0",dict(g3=0.0)),
              ("RR_EQ=1.5",dict(rbar=1.5)),("RR_EQ=2.5",dict(rbar=2.5)),
              ("цель 4.00 арифм.",dict(tgt=4.0))]:
    v=[rule(x,**kw) for _,x in rows[:3]]
    print(f"    {nm:28s} {min(v):.2f}–{max(v):.2f}%  -> на сетке {snap(np.mean(v)):.2f}%")
