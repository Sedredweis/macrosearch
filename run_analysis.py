import numpy as np, pandas as pd, numpy.linalg as la, json
from scipy.stats import chi2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import svar_soe as M

V = M.V; P4 = 4
X = M.load('data_239Q2.csv'); Y = X[V].values; T = len(Y)
X = M.load('data_239Q2.csv'); Y = X[V].values; T = len(Y)

kR, kP, kX = V.index('rG'), V.index('dPC'), V.index('dNFX')
OUT = {}

# ---------------------------------------------------------------- 1. данные
OUT['T'] = T; OUT['period'] = [X.index[0], X.index[-1]]
OUT['na_err'] = float(np.abs(M.agg_dY(Y) - X.dY_obs.values).max())
OUT['na_w'] = M.W_Y
ps = X.pistar.values
OUT['pistar'] = dict(mean=400*ps.mean(), sd=400*ps.std(),
                     ar1=float(np.corrcoef(ps[1:], ps[:-1])[0, 1]))
alt = X.dPC.values - (X.pistar.values - X.dPC.values)     # альтернативное соглашение
OUT['pistar_alt'] = dict(mean=400*alt.mean(), sd=400*alt.std())
desc = pd.DataFrame({'mean': Y.mean(0), 'sd': Y.std(0),
                     'ar1': [np.corrcoef(Y[1:, j], Y[:-1, j])[0, 1] for j in range(len(V))]},
                    index=V)
OUT['desc'] = desc.round(4).to_dict('index')

# ---------------------------------------------------------------- 2. оценка
m = M.BVAR(Y, p=P4)
OUT['maxroot'] = float(m.max_root())
S = la.cholesky(m.Sigma)
R2 = 1 - m.U.var(0) / Y[P4:].var(0)
lb = [M.ljung_box(m.U[:, j], 8) for j in range(len(V))]
OUT['fit'] = {V[j]: dict(R2=round(float(R2[j]), 3), LB8=round(lb[j], 2),
                         p=round(float(1 - chi2.cdf(lb[j], 8)), 3)) for j in range(len(V))}

# правило ЦБ
A0 = la.inv(S); A0 = A0 / np.diag(A0)[:, None]
row = -A0[kR].copy(); row[kR] = 0
Pi = [m.B[l*m.n:(l+1)*m.n].T for l in range(P4)]
lagr = np.array([(A0 @ Pi[l])[kR] for l in range(P4)])
rho = float(lagr[:, kR].sum())
OUT['rule'] = dict(rho=round(rho, 3),
                   phi_pi=round(float((row[kP] + lagr[:, kP].sum())/(1-rho)/400), 3),
                   phi_dy=round(float((row @ M.WVEC + (lagr @ M.WVEC).sum())/(1-rho)/400), 3),
                   phi_e=round(float((row[kX] + lagr[:, kX].sum())/(1-rho)/100), 4),
                   sd_shock=round(float(S[kR, kR]), 3))

# ---------------------------------------------------------------- 3. IRF
IRc = m.irf(20, S); DYc = M.agg_dY(IRc)
Sset, kid = M.identify(m, ndraw=6000, seed=7)
OUT['nrot'] = len(Sset)
IR = np.array([m.irf(20, s) for s in Sset]); DY = np.array([M.agg_dY(x) for x in IR])
mp = kid['mp']
OUT['mp_irf'] = [dict(h=h,
    rG=round(float(IRc[h, kR, kR]), 3),
    pi=round(float(400*IRc[h, kP, kR]), 3),
    pi16=round(float(400*np.percentile(IR[:, h, kP, mp], 16)), 3),
    pi84=round(float(400*np.percentile(IR[:, h, kP, mp], 84)), 3),
    dY=round(float(100*DYc[h, kR]), 3),
    dY16=round(float(100*np.percentile(DY[:, h, mp], 16)), 3),
    dY84=round(float(100*np.percentile(DY[:, h, mp], 84)), 3),
    dC=round(float(100*IRc[h, V.index('dC'), kR]), 3),
    dI=round(float(100*IRc[h, V.index('dI'), kR]), 3),
    dL=round(float(100*IRc[h, V.index('dL'), kR]), 3),
    dIM=round(float(100*IRc[h, V.index('dIM'), kR]), 3),
    dEX=round(float(100*IRc[h, V.index('dEX'), kR]), 3),
    dNFX=round(float(100*IRc[h, kX, kR]), 2)) for h in [0,1,2,3,4,5,6,8,10,12,16]]
OUT['mp_cum_price'] = round(float(100*IRc[1:21, kP, kR].sum()), 3)
OUT['mp_trough_y'] = round(float(100*DYc[:, kR].min()), 3)
OUT['mp_appreciate_share'] = round(float(np.mean(IR[:, 0, kX, mp] < 0)), 3)
OUT['chol_in_set'] = dict(rG012=[round(float(IRc[h, kR, kR]), 3) for h in (0,1,2)],
                          sum_pi=round(float(IRc[2:7, kP, kR].sum()), 5),
                          sum_dy=round(float(DYc[1:5, kR].sum()), 5))

# прочие шоки (Холецкий)
def irrow(j, var, sc=100):
    return [round(float(sc*IRc[h, V.index(var), j]), 3) for h in [0,1,2,4,8,12]]
OUT['other'] = {nm: dict(rG=[round(float(IRc[h, kR, V.index(nm)]), 3) for h in [0,1,2,4,8,12]],
                         pi=irrow(V.index(nm), 'dPC', 400), dY=[round(float(100*DYc[h, V.index(nm)]), 3)
                         for h in [0,1,2,4,8,12]], dNFX=irrow(V.index(nm), 'dNFX'))
                for nm in ['dPC','dNFX','dG','dC','dI','dEX','pistar','dInc']}

# FEVD (две упорядоченности)
def fevd(mm, Ss, h=20):
    ir = mm.irf(h, Ss); c = np.cumsum(ir**2, 0); return c / c.sum(2, keepdims=True)
F = fevd(m, S)
OUT['fevd_pi'] = {V[j]: round(float(100*F[3, kP, j]), 1) for j in range(m.n)}
OUT['fevd_r'] = {V[j]: round(float(100*F[3, kR, j]), 1) for j in range(m.n)}

# ---------------------------------------------------------------- 4. OOS
bench = {}
for nm, f in [('RW', lambda Yt, H: np.tile(Yt[-1], (H, 1))),
              ('среднее', lambda Yt, H: np.tile(Yt.mean(0), (H, 1)))]:
    e = {k: [] for k in ('pi1','pi4','r1','r4')}
    for t in range(120, T-8):
        fc = f(Y[:t], 8)
        e['pi1'].append(fc[0,kP]-Y[t,kP]); e['r1'].append(fc[0,kR]-Y[t,kR])
        e['pi4'].append(fc[:4,kP].sum()-Y[t:t+4,kP].sum()); e['r4'].append(fc[3,kR]-Y[t+3,kR])
    bench[nm] = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in e.items()}
grid = {}
for p in (2, 4, 6):
    for lam in (0.1, 0.35, 1.0, 2.0):
        grid[f'p={p}, λ={lam}'] = M.oos_rmse(Y, p, lam, H=8, start=120)
OUT['oos'] = {k: dict(pi1=round(400*v['pi1'],3), pi4=round(100*v['pi4'],3),
                      r1=round(v['r1'],3), r4=round(v['r4'],3))
              for k, v in {**grid, **bench}.items()}

# ---------------------------------------------------------------- 5. сценарии
scen = {}

#scen['плавное повышение до 6%'], _ = m.scenario(np.linspace(4.0, 6.0, 12), S, kR)
scen['траектория'], _ = m.scenario(np.array([-40.0, 45.0, 6.0, 6.0, 6.0, 6.5, 6.0, 6.5, 6.0, 6.5, 6.0, 6.5]), S, kR)
scen['оптимум 3.8/5.3/6.6%'], _ = m.scenario(np.repeat([3.82, 5.32, 6.64], 4), S, kR)
OUT['scen'] = {nm: dict(rG=[round(float(v), 2) for v in pth[:, kR]],
                        pi_y1=round(float(100*pth[0:4, kP].sum()), 2),
                        pi_y2=round(float(100*pth[4:8, kP].sum()), 2),
                        pi_y3=round(float(100*pth[8:12, kP].sum()), 2),
                        dY_y1=round(float(100*M.agg_dY(pth)[0:4].sum()), 2))
               for nm, pth in scen.items()}
# оптимальная постоянная ставка под метрику RMSE годовой инфляции от 4%
gridr = np.arange(0.0, 16.01, 0.25); best = None
for r in gridr:
    pth, _ = m.scenario(np.full(12, r), S, kR)
    a = np.array([100*pth[i:i+4, kP].sum() for i in (0, 4, 8)])
    rm = float(np.sqrt(np.mean((a - 4.0) ** 2)))
    if best is None or rm < best[1]:
        best = (float(r), rm, [round(float(v), 2) for v in a])
OUT['best_const_rate'] = dict(rate=best[0], rmse=round(best[1], 3), infl=best[2])

# ---------------------------------------------------------------- рисунки
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 150})
pan = [('rG', 'Ключевая ставка, п.п.', 1), ('dPC', 'Инфляция, % год.', 400),
       ('__dY', 'Выпуск, %', 100), ('dC', 'Потребление, %', 100),
       ('dI', 'Инвестиции, %', 100), ('dL', 'Занятость, %', 100),
       ('dNFX', 'Ном. курс, %', 100), ('dIM', 'Импорт, %', 100)]
fig, ax = plt.subplots(2, 4, figsize=(11, 4.6))
hh = np.arange(21)
for a, (v, ttl, sc) in zip(ax.ravel(), pan):
    if v == '__dY':
        c, lo, hi = sc*DYc[:, kR], sc*np.percentile(DY[:, :, mp], 16, 0), sc*np.percentile(DY[:, :, mp], 84, 0)
    else:
        j = V.index(v)
        c = sc*IRc[:, j, kR]; lo = sc*np.percentile(IR[:, :, j, mp], 16, 0); hi = sc*np.percentile(IR[:, :, j, mp], 84, 0)
    a.fill_between(hh, lo, hi, alpha=.25, color='#7a9bc4', lw=0)
    a.plot(hh, c, color='#1b3a5c', lw=1.6); a.axhline(0, color='k', lw=.6)
    a.set_title(ttl, fontsize=8); a.set_xlabel('кварталы', fontsize=7)
fig.suptitle('Отклик на шок ужесточения ДКП (+1 с.о. ≈ +0.56 п.п.): точечная схема и 68% знаково-идентифицированное множество', fontsize=9)
fig.tight_layout(); fig.savefig('fig_irf_mp.png', bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for nm, pth in scen.items():
    seq = np.concatenate([Y[-3:, kP], pth[:, kP]])       # склейка с фактом
    ann = [100*seq[i:i+4].sum() for i in range(12)]
    ax[0].plot(range(1, 13), pth[:, kR], lw=1.4, label=nm)
    ax[1].plot(range(1, 13), ann, lw=1.4, label=nm)
ax[1].axhline(4, color='r', ls='--', lw=1, label='цель 4%')
ax[0].set_title('Траектория ключевой ставки, %'); ax[1].set_title('Годовая инфляция, %')
for a in ax: a.set_xlabel('кварталы вперёд')
ax[1].legend(fontsize=6.5); fig.tight_layout(); fig.savefig('fig_scenarios.png', bbox_inches='tight'); plt.close(fig)

eps = la.solve(S, m.U.T).T[:, kR]
fig, a = plt.subplots(figsize=(9, 2.4))
a.bar(range(len(eps)), eps, color=np.where(eps > 0, '#b3402f', '#2f6fb3'), width=1.0)
a.set_title('Идентифицированный шок ДКП (структурная инновация правила), с.о.', fontsize=9)
a.set_xticks(range(0, len(eps), 20)); a.set_xticklabels(X.index[P4::20], fontsize=7)
fig.tight_layout(); fig.savefig('fig_mp_shock.png', bbox_inches='tight'); plt.close(fig)