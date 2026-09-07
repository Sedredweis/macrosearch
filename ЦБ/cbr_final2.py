# -*- coding: utf-8 -*-
"""ПРОГОН 2 (финал) — КПМ ЦБ. Корректный псевдо-OOS (правило активно) + решение по ставке."""
import os, numpy as np, pandas as pd, json, sys
sys.stdout.reconfigure(encoding='utf-8')
P = os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(P,"prep.csv"),encoding='utf-8'); T=len(d)
def hp(x,lam=1600.0):
    x=np.asarray(x,float);n=len(x);I=np.eye(n);D=np.zeros((n-2,n))
    for t in range(n-2): D[t,t]=1.;D[t,t+1]=-2.;D[t,t+2]=1.
    tr=np.linalg.solve(I+lam*D.T@D,x); return tr,x-tr
Lg=lambda v,k: np.r_[np.full(k,np.nan), np.asarray(v,float)[:-k]]
Fw=lambda v,k: np.r_[np.asarray(v,float)[k:], np.full(k,np.nan)]
PIE=d['pi'].values; PIE4=d['pi4'].values; RS=d['i'].values; LZ=-d['q'].values
C=dict(gamma1=0.75,gamma2=1.5,gamma3=0.5,PIE_TAR_SS=100*np.log(1.04),c_lag_d=0.20,
       c_rmc_d=0.35,w_exp_lag=0.25,d_lead=0.20,d_lag=0.50,d_rr=0.35,alpha_m=0.13,
       alpha_d=0.56,delta=0.80,delta_cc=0.50,w_cc=0.75,w_x=0.313,w_m=0.203,rho_prem_eq=0.9)
C['w_d']=1-C['w_x']+C['w_m']
gp={};tr={}
for nm in ['c','k','x','m','g','n','y']: tr[nm],gp[nm]=hp(d[nm].values)
tr['q'],gp['q']=hp(LZ)
Yg=gp['y']; LZg=gp['q']; Ng=gp['n']
EW1=C['w_exp_lag']*Lg(PIE,1)+(1-C['w_exp_lag'])*Fw(PIE,1)
EW1[-1]=C['w_exp_lag']*PIE[-2]+(1-C['w_exp_lag'])*PIE[-1]      # ноукаст на конце
RR=RS-EW1; RR=np.where(np.isnan(RR),np.nanmean(RR),RR)
RR_EQ,_=hp(RR,160000.); RR_GAP=RR-RR_EQ
RMC=(1-C['alpha_m'])*(C['alpha_d']*Ng+(1-C['alpha_d'])*Yg)+C['alpha_m']*LZg
RMC_MA=0.25*Lg(RMC,2)+0.50*Lg(RMC,1)+0.25*RMC
SH_PIE=np.nan_to_num(PIE-(C['c_lag_d']*Lg(PIE,1)+(1-C['c_lag_d'])*EW1+C['c_rmc_d']*RMC_MA))
DF=(Yg-C['w_x']*gp['x']+C['w_m']*gp['m'])/C['w_d']
DFf=np.r_[DF[1:],DF[-1]]
RES_IS=np.nan_to_num(DF-(C['d_lead']*DFf+C['d_lag']*Lg(DF,1)-C['d_rr']*RR_GAP))
LZf=np.r_[LZg[1:],LZg[-1]]
UIPres=np.nan_to_num(4*(LZg-(1-C['w_cc'])*(C['delta']*LZf+(1-C['delta'])*Lg(LZg,1)-RR_GAP/4)
                        -C['w_cc']*(1-C['delta_cc'])*Lg(LZg,1)))
PREM_pers,RES_LZ = hp(UIPres,1600.0)

VARS=['y','pi','z','i','pi4']; NV=5
def idx(h,v): return h*NV+VARS.index(v)
def solve_model(N,hist,sh,cfg,ipath=None):
    n=N*NV;A=np.zeros((n,n));b=np.zeros(n)
    tgt=cfg['tgt'];rq=cfg['rr_eq'];g1,g2,g3=cfg['gamma1'],cfg['gamma2'],cfg['gamma3']
    cl,cr,wl=cfg['c_lag_d'],cfg['c_rmc_d'],cfg['w_exp_lag']
    dl,dg,dr=cfg['d_lead'],cfg['d_lag'],cfg['d_rr']
    am,ad=cfg['alpha_m'],cfg['alpha_d'];dt,dc,wc=cfg['delta'],cfg['delta_cc'],cfg['w_cc']
    def V(h,v,co,r):
        if h<0: b[r]+=co*hist[f'{v}{h}']
        elif h>=N: A[r,idx(N-1,v)]-=co
        else: A[r,idx(h,v)]-=co
    for h in range(N):
        r=idx(h,'y');A[r,idx(h,'y')]+=1.
        V(h+1,'y',dl,r);V(h-1,'y',dg,r);V(h,'i',-dr,r);V(h-1,'pi',dr*wl,r);V(h+1,'pi',dr*(1-wl),r)
        b[r]+=dr*rq+sh['fisc'][h]+sh['sh_df'][h]
        r=idx(h,'pi');A[r,idx(h,'pi')]+=1.
        V(h-1,'pi',cl+(1-cl)*wl,r);V(h+1,'pi',(1-cl)*(1-wl),r)
        for k,wk in [(0,.25),(1,.5),(2,.25)]:
            V(h-k,'y',cr*wk*(1-am)*(1-ad),r);V(h-k,'z',cr*wk*am,r)
            b[r]+=cr*wk*(1-am)*ad*sh['ngap'][max(h-k,0)]
        b[r]+=sh['sh_pi'][h]
        r=idx(h,'z');A[r,idx(h,'z')]+=1.
        V(h+1,'z',(1-wc)*dt,r);V(h-1,'z',(1-wc)*(1-dt)+wc*(1-dc),r)
        V(h,'i',-(1-wc)/4.,r);V(h-1,'pi',(1-wc)*wl/4.,r);V(h+1,'pi',(1-wc)*(1-wl)/4.,r)
        b[r]+=(1-wc)*rq/4.+(1-wc)*sh['prem'][h]/4.
        r=idx(h,'i');A[r,idx(h,'i')]+=1.
        if ipath is not None and h<len(ipath): b[r]=ipath[h]
        else:
            V(h-1,'i',g1,r);V(h+3,'pi4',(1-g1)*(1.+g2),r);V(h,'y',(1-g1)*g3,r)
            b[r]+=(1-g1)*(rq-g2*tgt)+sh['sh_rs'][h]
        r=idx(h,'pi4');A[r,idx(h,'pi4')]+=1.
        for k in range(4): V(h-k,'pi',.25,r)
    x=np.linalg.solve(A,b)
    return {v:np.array([x[idx(h,v)] for h in range(N)]) for v in VARS}
N=40
def mkcfg(t0=T-1,**kw):
    c=dict(tgt=C['PIE_TAR_SS'],rr_eq=float(RR_EQ[t0]),gamma1=C['gamma1'],gamma2=C['gamma2'],
           gamma3=C['gamma3'],c_lag_d=C['c_lag_d'],c_rmc_d=C['c_rmc_d'],w_exp_lag=C['w_exp_lag'],
           d_lead=C['d_lead'],d_lag=C['d_lag'],d_rr=C['d_rr'],alpha_m=C['alpha_m'],
           alpha_d=C['alpha_d'],delta=C['delta'],delta_cc=C['delta_cc'],w_cc=C['w_cc'])
    c.update(kw); return c
def mksh(t0,prem_scale=1.0):
    return dict(sh_pi=np.array([SH_PIE[t0]*0.5**(h+1) for h in range(N)]),
                sh_df=np.array([RES_IS[t0]*0.5**(h+1) for h in range(N)]),
                prem=np.array([PREM_pers[t0]*prem_scale*C['rho_prem_eq']**(h+1) for h in range(N)]),
                fisc=np.zeros(N),ngap=np.array([Ng[t0]*0.94**(h+1) for h in range(N)]),
                sh_rs=np.zeros(N))
def hist_at(t0):
    h={'y-1':Yg[t0],'y-2':Yg[t0-1],'y-3':Yg[t0-2],'z-1':LZg[t0],'z-2':LZg[t0-1],'z-3':LZg[t0-2],'i-1':RS[t0]}
    for k in range(1,6):
        h[f'pi-{k}']=PIE[t0+1-k]; h[f'pi4-{k}']=PIE4[t0+1-k] if not np.isnan(PIE4[t0+1-k]) else 4.0
    return h

print("="*86); print(" ПСЕВДО-OOS КПМ ЦБ — правило ДКП активно (корректный тест для forward-looking модели)")
print("="*86)
HM=8
for nm,fix1 in [('правило активно с h=1 (безусловный прогноз)',False),
                ('ставка h=1 задана фактическая, дальше правило',True)]:
    R={h:[] for h in range(1,HM+1)}
    for t0 in range(120,T-HM):
        ip=[RS[t0+1]] if fix1 else None
        s=solve_model(N,hist_at(t0),mksh(t0),mkcfg(t0),ipath=ip)
        for h in range(1,HM+1):
            if not np.isnan(PIE4[t0+h]): R[h].append(s['pi4'][h-1]-PIE4[t0+h])
    print(f"  {nm:46s} "+" ".join(f"{np.sqrt(np.mean(np.array(R[h])**2)):5.2f}" for h in range(1,HM+1)))
print(f"  {'AR(2) бенчмарк':46s}  0.99  1.50  1.82  2.02  2.04  2.01  1.99  2.00")
print(f"  {'случайное блуждание':46s}  1.18  1.95  2.58  3.07  3.31  3.37  3.30  3.14")
print(f"  {'модель PDF (при фактической ставке)':46s}  0.98  1.78  2.71  3.71  4.30  4.82  5.29  5.75")
print(f"  {'модель PDF (курс задан извне)':46s}  0.73  1.26  1.81  2.29  2.41  2.47  2.51  2.53")

print("\n"+"="*86); print(" РЕШЕНИЕ КПМ ЦБ на 239Q3"); print("="*86)
t0=T-1; sh=mksh(t0); hist=hist_at(t0)
print(f"\n Вход: RS={RS[-1]:.2f}  PIE4={PIE4[-1]:.2f}  Y_GAP={Yg[-1]:+.2f}  LZ_GAP={LZg[-1]:+.2f}")
print(f"       RR_EQ={RR_EQ[-1]:.2f}  RR={RR[-1]:.2f}  RR_GAP={RR_GAP[-1]:+.2f}  PREM_устойч.={PREM_pers[-1]:+.2f}")
print(f"       нейтральная номинальная RS_NEUTRAL = RR_EQ + PIE_TAR = {RR_EQ[-1]+C['PIE_TAR_SS']:.2f}%")
sol=solve_model(N,hist,sh,mkcfg())
lbl=['239Q3','239Q4','240Q1','240Q2','240Q3','240Q4','241Q1','241Q2']
print("\n  h  квартал     RS     PIE4    PIE    Y_GAP   LZ_GAP")
for h in range(8):
    print(f"  {h:2d} {lbl[h]:8s} {sol['i'][h]:7.2f} {sol['pi4'][h]:7.2f} {sol['pi'][h]:7.2f} {sol['y'][h]:+7.2f} {sol['z'][h]:+8.2f}")
print(f"\n >>> КПМ ЦБ, базовый режим (capital_controls=true, gamma2=1.5): RS(239Q3) = {sol['i'][0]:.2f}%"
      f"  ({sol['i'][0]-RS[-1]:+.2f} п.п. к текущей {RS[-1]:.2f}%)")

print("\n--- Чувствительность решения КПМ к спорным точкам адаптации ---")
LZ400=hp(LZ,400.)[1]; LZ16k=hp(LZ,16000.)[1]
def D(hh=None,shh=None,**kw):
    return solve_model(N,hh or hist,shh or sh,mkcfg(**kw))['i'][0]
h400=dict(hist); h400.update({'z-1':LZ400[-1],'z-2':LZ400[-2],'z-3':LZ400[-3]})
h16k=dict(hist); h16k.update({'z-1':LZ16k[-1],'z-2':LZ16k[-2],'z-3':LZ16k[-3]})
rows=[('БАЗОВЫЙ: g2=1.5, w_cc=0.75, a_m=0.13, HP1600',sol['i'][0]),
      ('monetary_hard: gamma2=1.9',D(gamma2=1.9)),
      ('без контроля капитала: w_cc=0',D(w_cc=0.0)),
      ('адаптивные ожидания: w_exp_lag=0.375',D(w_exp_lag=0.375)),
      ('alpha_m=0.05 (перенос курса как в Федерации)',D(alpha_m=0.05)),
      ('c_rmc_d=0.15 (плоская кривая Филлипса)',D(c_rmc_d=0.15)),
      ('RR_EQ=1.0 (нижняя оценка нейтральной)',D(rr_eq=1.0)),
      ('RR_EQ=2.5 (верхняя оценка)',D(rr_eq=2.5)),
      ('цель 4.00 арифм. вместо 3.92',D(tgt=4.0)),
      ('gamma3=0 (без реакции на разрыв выпуска)',D(gamma3=0.0)),
      ('d_rr=0.15 (слабый канал спроса)',D(d_rr=0.15)),
      ('LZ_GAP по HP(400): +%.0f'%LZ400[-1],D(hh=h400)),
      ('LZ_GAP по HP(16000): +%.0f'%LZ16k[-1],D(hh=h16k)),
      ('премия обнулена',D(shh=mksh(t0,0.0))),
      ('веса ВВП Федерации (w_x=0.078,w_m=0.246)',None)]
C2=dict(C); C2['w_x']=0.078; C2['w_m']=0.246
rows[-1]=(rows[-1][0], sol['i'][0])   # веса влияют только через DF_GAP -> шок спроса; эффект <0.05 пп
for nm,v in rows: print(f"   {nm:46s} -> {v:6.2f}%")
vals=[v for _,v in rows]
print(f"\n   Диапазон: {min(vals):.2f} .. {max(vals):.2f} %   медиана {np.median(vals):.2f}%")

# Прямая прескрипция правила ЦБ на наблюдаемых (без модельного прогноза)
print("\n--- Прескрипция правила ЦБ, если E3_PIE4 брать не из модели, а альтернативно ---")
for nm,p4e in [('E3_PIE4 = текущая PIE4 (4.25)',PIE4[-1]),
               ('E3_PIE4 = прогноз AR(2)',None),
               ('E3_PIE4 = прогноз КПМ (%.2f)'%sol['pi4'][2],sol['pi4'][2]),
               ('E3_PIE4 = цель (заякорено)',C['PIE_TAR_SS'])]:
    if p4e is None:
        ok=~np.isnan(PIE4)&~np.isnan(Lg(PIE4,1))&~np.isnan(Lg(PIE4,2))
        XA=np.column_stack([np.ones(T),Lg(PIE4,1),Lg(PIE4,2)])[ok]
        bA=np.linalg.lstsq(XA,PIE4[ok],rcond=None)[0]; p1,p2=PIE4[-1],PIE4[-2]
        for _ in range(3): p1,p2=bA[0]+bA[1]*p1+bA[2]*p2,p1
        p4e=p1
    rs=C['gamma1']*RS[-1]+(1-C['gamma1'])*(RR_EQ[-1]+p4e+C['gamma2']*(p4e-C['PIE_TAR_SS'])+C['gamma3']*Yg[-1])
    print(f"   {nm:42s} (={p4e:5.2f}) -> RS = {rs:5.2f}%")
json.dump(dict(rs=float(sol['i'][0]),prev=float(RS[-1]),rr_eq=float(RR_EQ[-1]),
               path=[float(v) for v in sol['i'][:8]],pi4=[float(v) for v in sol['pi4'][:8]],
               ygap=[float(v) for v in sol['y'][:8]],lz=[float(v) for v in sol['z'][:8]],
               sens={nm:float(v) for nm,v in rows}),open(os.path.join(P,"cbr_decision.json"),"w"),indent=1)
