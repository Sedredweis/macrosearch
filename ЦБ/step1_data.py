# -*- coding: utf-8 -*-
"""Шаг 1: загрузка данных Федерации, единицы, тренды/разрывы, знаковые проверки."""
import os, numpy as np, pandas as pd, sys, io
sys.stdout.reconfigure(encoding='utf-8')

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_239Q2.csv")

raw = pd.read_csv(CSV, skiprows=1, encoding='utf-8-sig')
raw.columns = [c.strip() for c in raw.columns]
print("cols:", list(raw.columns))
print("n =", len(raw), " period:", raw['period'].iloc[0], "->", raw['period'].iloc[-1])

d = pd.DataFrame(index=range(len(raw)))
d['period'] = raw['period'].values
d['i']   = raw['zzobs_r_G'].astype(float).values          # ключевая ставка, % годовых
dY   = raw['zzobs_dY'].astype(float).values
dPC  = raw['zzobs_dPC'].astype(float).values
dNFX = raw['zzobs_dNFX'].astype(float).values
dRFX = raw['zzobs_dRFX'].astype(float).values
SPPY = raw['zzobs_SP_PY'].astype(float).values            # ln(доля расходов бюджета в ВВП)
dInc = raw['zzobs_dInc'].astype(float).values
dC   = raw['zzobs_dC'].astype(float).values
dG   = raw['zzobs_dG'].astype(float).values
dI   = raw['zzobs_dI'].astype(float).values
dIM  = raw['zzobs_dIM'].astype(float).values
dEX  = raw['zzobs_dEX'].astype(float).values
dL   = raw['zzobs_dL'].astype(float).values

# --- единицы ---
# приросты: логарифмические кв/кв. В % годовых = x*400. Уровни (100*log) = 100*cumsum.
d['pi']   = 400*dPC                                        # квартальная инфляция, % годовых
d['ds']   = 400*dNFX                                       # прирост номинального курса, % годовых
d['dq']   = 400*dRFX                                       # прирост реального курса, % годовых
d['pi4']  = pd.Series(100*dPC).rolling(4).sum().values      # годовая инфляция, %
d['dtau'] = 400*dInc
d['gy']   = 100*SPPY                                        # 100*ln(доля расходов в ВВП)
d['dgy']  = np.r_[np.nan, np.diff(100*SPPY)]

for nm, s in [('y',dY),('c',dC),('k',dI),('g',dG),('x',dEX),('m',dIM),('n',dL),
              ('s',dNFX),('q',dRFX)]:
    d[nm] = 100*np.cumsum(s)                                # уровни, 100*log

# --- зарубежная инфляция из тождества  dq = ds + pi* - pi ---
d['pistar'] = d['dq'] - d['ds'] + d['pi']

print("\n=== средние темпы, % годовых ===")
for nm, s in [('выпуск',dY),('потребление',dC),('инвестиции',dI),('госзакупки',dG),
              ('экспорт',dEX),('импорт',dIM),('занятость',dL),('ИПЦ',dPC),
              ('ном.курс',dNFX),('реал.курс',dRFX),('доходы бюдж.',dInc)]:
    print(f"  {nm:16s} {400*np.mean(s):8.3f}   sd(кв,%год) {400*np.std(s):7.2f}")
print(f"  {'ставка':16s} {np.mean(d['i']):8.3f}   sd {np.std(d['i']):7.2f}")
print(f"  {'pi4':16s} {np.nanmean(d['pi4']):8.3f}   sd {np.nanstd(d['pi4']):7.2f}")
print(f"  {'pi*':16s} {np.nanmean(d['pistar']):8.3f}   sd {np.nanstd(d['pistar']):7.2f}")
print(f"  {'реальная i-pi':16s} {np.mean(d['i']-d['pi']):8.3f}")
print(f"  {'реальная i-pi4':16s} {np.nanmean(d['i']-d['pi4']):8.3f}")
print(f"  доля расходов бюдж. в ВВП: {np.exp(SPPY).mean()*100:.1f}%  (мин {np.exp(SPPY).min()*100:.1f}, макс {np.exp(SPPY).max()*100:.1f})")

# ================= ЗНАКОВЫЕ ПРОВЕРКИ =================
print("\n=== ПРОВЕРКА 1: реальный vs номинальный курс (регрессия dq на ds) ===")
X = np.column_stack([np.ones(len(d)), d['ds'].values])
b = np.linalg.lstsq(X, d['dq'].values, rcond=None)[0]
res = d['dq'].values - X@b
se = np.sqrt(np.sum(res**2)/(len(d)-2)*np.linalg.inv(X.T@X)[1,1])
print(f"  dq = {b[0]:.3f} + {b[1]:.4f}*ds ,  se={se:.4f}, t={(b[1]-1)/se:.2f} против H0: beta=1")
print(f"  corr(ds,dq) = {np.corrcoef(d['ds'],d['dq'])[0,1]:.4f}")
print("  => одинаковая котировка" if abs(b[1]-1)<0.1 else "  => ВНИМАНИЕ: разная котировка/масштаб")

print("\n=== ПРОВЕРКА 2: направление котировки (рост курса = ослабление или укрепление?) ===")
pi4 = d['pi4'].values
for k in range(0,7):
    a = d['ds'].values[:len(d)-k]; bb = pi4[k:]
    ok = ~np.isnan(a) & ~np.isnan(bb)
    print(f"  corr(ds_t, pi4_t+{k}) = {np.corrcoef(a[ok],bb[ok])[0,1]:+.3f}")
print("  Если рост показателя = ОСЛАБЛЕНИЕ валюты -> корреляция с будущей инфляцией должна быть ПОЛОЖИТЕЛЬНОЙ.")

# HP-фильтр
def hp(x, lam=1600.0):
    x = np.asarray(x, float); T = len(x)
    I = np.eye(T); D = np.zeros((T-2, T))
    for t in range(T-2):
        D[t,t]=1.0; D[t,t+1]=-2.0; D[t,t+2]=1.0
    trend = np.linalg.solve(I + lam*D.T@D, x)
    return trend, x-trend

print("\n=== ПРОВЕРКА 3: разрыв выпуска (HP,1600) и кривая Филлипса ===")
ytr, ygap = hp(d['y'].values)
d['y_tr'], d['y_gap'] = ytr, ygap
print(f"  sd(y_gap) = {ygap.std():.2f}  ; y_gap(239Q2) = {ygap[-1]:+.2f}")
# простая проверка наклона
Xp = np.column_stack([np.ones(len(d)-1), d['pi'].values[:-1], ygap[1:]])
bp = np.linalg.lstsq(Xp, d['pi'].values[1:], rcond=None)[0]
print(f"  pi_t = {bp[0]:.2f} + {bp[1]:.3f} pi_t-1 + {bp[2]:+.4f} ygap_t   (наклон Филлипса b2)")

print("\n=== последние 8 кварталов ===")
tail = d[['period','i','pi','pi4','ds','dq','pistar','y_gap']].tail(8)
print(tail.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

d.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"prep.csv"), index=False, encoding='utf-8')
print("\nсохранено prep.csv")
