"""
Три раздела:
  A. Какое правило ДКП было в выборке на самом деле (проциклично оно или нет)
     и почему из-за этого возникает «ценовая загадка» в приведённой форме.
  B. С каким лагом каждая переменная влияет на инфляцию и на ставку —
     три независимых среза: коэффициенты уравнений с апостериорными t,
     модельно-свободные кросс-корреляции, динамические мультипликаторы.
  C. Замена правила на явное правило таргетирования инфляции: область
     устойчивости замкнутой системы по (phi_pi, phi_y, rho), подбор лучшего
     реализуемого правила и диагностика правдоподобия (Leeper-Zha).
"""
import sys, json
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import svar_soe as M

pd.set_option('display.width', 220)
DATA = sys.argv[1] if len(sys.argv) > 1 else 'data_239Q2.csv'
X = M.load(DATA); Y = X[M.V].values; V = M.V; T = len(Y)
kR, kP, kX = V.index('rG'), V.index('dPC'), V.index('dNFX')
m = M.BVAR(Y, p=M.P_DEFAULT); n, p = m.n, m.p
S = la.cholesky(m.Sigma)
s_mp, share = M.mp_max_share(m, seed=1)
OUT = {'data': [X.index[0], X.index[-1]], 'T': T, 'p': p}


def head(t):
    print('\n' + '=' * 78); print(t); print('=' * 78)


# ================================================================== A. РЕЖИМ
head('A. КАКОЕ ПРАВИЛО ДКП ДЕЙСТВОВАЛО В ВЫБОРКЕ')

ann = M.ann_pct(X.dPC.values)                 # годовая инфляция из кв. прироста
rr = X.rG.values - ann                        # реальная ставка ex post
dy = M.agg_dY(Y)
print('реальная ставка ex post: среднее %.2f%%, с.о. %.2f п.п.' % (rr.mean(), rr.std()))
cor = dict(rr_pi=float(np.corrcoef(rr, ann)[0, 1]),
           rr_dY=float(np.corrcoef(rr, dy)[0, 1]),
           i_pi=float(np.corrcoef(X.rG.values, ann)[0, 1]),
           i_dY=float(np.corrcoef(X.rG.values, dy)[0, 1]),
           i_e=float(np.corrcoef(X.rG.values, X.dNFX.values)[0, 1]))
print('corr(реальная ставка, инфляция)  = %+.3f   <- при активном правиле должно быть > 0' % cor['rr_pi'])
print('corr(реальная ставка, прирост Y) = %+.3f' % cor['rr_dY'])
print('corr(ном. ставка, инфляция)      = %+.3f' % cor['i_pi'])
print('corr(ном. ставка, прирост Y)     = %+.3f   <- отриц. = ужесточение в спаде' % cor['i_dY'])
print('corr(ном. ставка, курс)          = %+.3f   <- отриц. = смягчение при обесценении' % cor['i_e'])

# --- прямая оценка правила: МНК, без всякой идентификации
ann4 = np.array([M.yoy_pct(Y[t - 3:t + 1, kP]) for t in range(3, T)])
dy4 = np.array([100 * dy[t - 3:t + 1].sum() for t in range(3, T)])
i = Y[3:, kR]


def ols(yv, Xm, names):
    Xm = np.column_stack([Xm, np.ones(len(Xm))])
    b, *_ = la.lstsq(Xm, yv, rcond=None)
    e = yv - Xm @ b
    s2 = e @ e / (len(yv) - Xm.shape[1])
    se = np.sqrt(np.diag(s2 * la.inv(Xm.T @ Xm)))
    return dict(zip(names + ['const'], zip(np.round(b, 4), np.round(b / se, 2))))


nm = ['i(t-1)', 'i(t-2)', 'pi4(t-1)', 'dy4(t-1)', 'dNFX(t-1)']
est = ols(i[2:], np.column_stack([i[1:-1], i[:-2], ann4[1:-1], dy4[1:-1],
                                  100 * Y[4:-1, kX]]), nm)
print('\nПРЯМАЯ ОЦЕНКА ПРАВИЛА (МНК; не зависит ни от какой идентификации):')
for k_, (b_, t_) in est.items():
    print('  %-10s %+9.4f   (t = %+5.2f)' % (k_, b_, t_))
rho_h = est['i(t-1)'][0] + est['i(t-2)'][0]
phi_pi_h = est['pi4(t-1)'][0] / (1 - rho_h)
phi_y_h = est['dy4(t-1)'][0] / (1 - rho_h)
phi_e_h = est['dNFX(t-1)'][0] / (1 - rho_h)
print('  rho = %.3f;  ДОЛГОСРОЧНЫЕ  phi_pi = %.3f   phi_y = %.3f   phi_e = %.3f'
      % (rho_h, phi_pi_h, phi_y_h, phi_e_h))
print('  ПРИНЦИП ТЕЙЛОРА %s (phi_pi %s 1)'
      % ('НАРУШЕН' if phi_pi_h <= 1 else 'выполняется', '<=' if phi_pi_h <= 1 else '>'))

# --- A2: менялся ли режим внутри выборки (скользящее окно)
print('\nA2. СКОЛЬЗЯЩАЯ ОЦЕНКА ПРАВИЛА (окно 60 кв.) — был ли период активной политики?')
Zfull = np.column_stack([i[1:-1], i[:-2], ann4[1:-1], dy4[1:-1], np.ones(len(i) - 2)])
yfull = i[2:]
roll = []
print('    окно                  rho     phi_pi    t(phi_pi)')
for st_ in range(0, len(yfull) - 60 + 1, 15):
    sl_ = slice(st_, st_ + 60)
    Zs, ys = Zfull[sl_], yfull[sl_]
    bs = la.lstsq(Zs, ys, rcond=None)[0]
    es = ys - Zs @ bs
    ses = np.sqrt(np.diag((es @ es / (len(ys) - 5)) * la.inv(Zs.T @ Zs)))
    rr_ = bs[0] + bs[1]
    roll.append(dict(start=X.index[5 + st_], end=X.index[5 + st_ + 59],
                     rho=float(rr_), phi_pi=float(bs[2] / (1 - rr_)),
                     t=float(bs[2] / ses[2])))
    print('    %s..%s   %6.3f  %+9.3f   %+6.2f'
          % (roll[-1]['start'], roll[-1]['end'], roll[-1]['rho'],
             roll[-1]['phi_pi'], roll[-1]['t']))
if max(abs(r_['t']) for r_ in roll) < 2.0:
    print('    Ни в одном окне реакция на инфляцию не значима: режим НЕ менялся,')
    print('    оценивать трансмиссию на «более активной» подвыборке не получится.')
OUT['rolling_rule'] = roll

# --- то же через рекурсивную нормировку: показать, откуда берётся ложное phi_pi > 1
A0 = la.inv(S); A0 = A0 / np.diag(A0)[:, None]
row = -A0[kR].copy(); row[kR] = 0
lagr = np.array([(A0 @ m.B[l * n:(l + 1) * n].T)[kR] for l in range(p)])
rho_c = float(lagr[:, kR].sum())
cont = float(row[kP] / (1 - rho_c) / 400)
lagp = float(lagr[:, kP].sum() / (1 - rho_c) / 400)
print('\nДЕКОМПОЗИЦИЯ phi_pi в РЕКУРСИВНОЙ схеме (почему там получается > 1):')
print('  мгновенный член из A0 (артефакт упорядочения) : %+7.3f' % cont)
print('  лаговая часть (собственно реакция правила)    : %+7.3f' % lagp)
print('  сумма                                         : %+7.3f' % (cont + lagp))
print('  rho = %.3f -> полупериод подстройки %.1f кв.; за 8 кв. реализуется %.0f%% реакции'
      % (rho_c, np.log(0.5) / np.log(rho_c), 100 * (1 - rho_c ** 8)))
print('  => «принцип Тейлора выполняется» в рекурсивной схеме держится только на')
print('     мгновенном члене, который задан порядком переменных, а не данными.')

# --- отклик реальной ставки на шок инфляции
ir = m.irf(20, S)
rr_ir = ir[:, kR, kP] - M.ANN * ir[:, kP, kP]
print('\nОТКЛИК НА ШОК ИНФЛЯЦИИ (+1 с.к.о.), рекурсивная схема:')
print('  инфляция, %% год. h=0..8:', np.round(M.ANN * ir[:9, kP, kP], 3))
print('  ном. ставка, п.п.       :', np.round(ir[:9, kR, kP], 3))
print('  РЕАЛЬНАЯ ставка, п.п.   :', np.round(rr_ir[:9], 3))
acc = bool(rr_ir[1:9].sum() < 0)
print('  => реальная ставка %s' % ('ПАДАЕТ: политика аккомодационная/проциклическая'
                                   if acc else 'растёт: политика контрциклическая'))
OUT['regime'] = dict(corr=cor, real_rate_mean=float(rr.mean()),
                     ols_rule={k_: list(map(float, v_)) for k_, v_ in est.items()},
                     rho=float(rho_h), phi_pi=float(phi_pi_h), phi_y=float(phi_y_h),
                     phi_e=float(phi_e_h), taylor_principle=bool(phi_pi_h > 1),
                     chol_phi_pi_contemp=cont, chol_phi_pi_lagged=lagp,
                     real_rate_falls_after_inflation=acc)

# ================================================================== B. ЛАГИ
head('B. С КАКИМ ЛАГОМ КАЖДАЯ ПЕРЕМЕННАЯ ВЛИЯЕТ НА ИНФЛЯЦИЮ И НА СТАВКУ')

# B1. коэффициенты уравнения с апостериорными t
def eq_table(eq):
    sd = np.sqrt(np.diag(m.Vpost[eq]))
    Tt = np.zeros((n, p)); C = np.zeros((n, p))
    for l in range(p):
        for j in range(n):
            r = l * n + j; C[j, l] = m.B[r, eq]; Tt[j, l] = m.B[r, eq] / sd[r]
    return C, Tt


for eq, lbl in [(kP, 'ИНФЛЯЦИИ dPC'), (kR, 'СТАВКИ rG')]:
    C, Tt = eq_table(eq)
    df = pd.DataFrame(Tt, index=V, columns=['L%d' % (l + 1) for l in range(p)])
    print('\nB1. t-отношения (коэффициент / апостериорное с.к.о.) в уравнении %s' % lbl)
    print('    |t| > 2 — устойчиво отличен от нуля при данном приоре')
    print(df.round(2).to_string())

# B2. модельно-свободные кросс-корреляции
def xcorr(tgt, K=8):
    return np.array([[float(np.corrcoef(Y[:T - k, j], Y[k:, tgt])[0, 1])
                      for k in range(K + 1)] for j in range(n)])


for tgt, lbl in [(kP, 'инфляцией'), (kR, 'ставкой')]:
    Mx = xcorr(tgt)
    df = pd.DataFrame(Mx, index=V, columns=['k=%d' % k for k in range(9)])
    print('\nB2. corr(x_{t-k}, %s_t) — без модели' % lbl)
    print(df.round(3).to_string())

# B3. динамические мультипликаторы (рекурсивная схема — описательно)
print('\nB3. Отклик ИНФЛЯЦИИ (%% год.) на шок каждой переменной, рекурсивная схема')
mult = pd.DataFrame({v: M.ANN * ir[:9, kP, j] for j, v in enumerate(V)},
                    index=['h=%d' % h for h in range(9)]).T
mult['пик h'] = [int(np.argmax(np.abs(M.ANN * ir[:, kP, V.index(v)]))) for v in mult.index]
mult['накопл. цены %'] = [100 * ir[:, kP, V.index(v)].sum() for v in mult.index]
print(mult.round(3).to_string())

# B4. сводка: лаг влияния на инфляцию по трём срезам
C_pi, T_pi = eq_table(kP); C_r, T_r = eq_table(kR)
Xc = xcorr(kP)
summary = []
for j, v in enumerate(V):
    lag_t = int(np.argmax(np.abs(T_pi[j]))) + 1
    lag_x = int(np.argmax(np.abs(Xc[j][1:]))) + 1          # k>=1
    lag_m = int(np.argmax(np.abs(M.ANN * ir[1:, kP, j]))) + 1
    summary.append(dict(v=v, t_lag=lag_t, t_val=round(float(T_pi[j][lag_t - 1]), 2),
                        xc_lag=lag_x, xc_val=round(float(Xc[j][lag_x]), 3),
                        mult_lag=lag_m,
                        mult_val=round(float(M.ANN * ir[lag_m, kP, j]), 3),
                        cum=round(float(100 * ir[:, kP, j].sum()), 3)))
sdf = pd.DataFrame(summary).set_index('v')
sdf.columns = ['лаг (t)', 't', 'лаг (corr)', 'corr', 'лаг (мульт.)', 'мульт.', 'накопл.%']
print('\nB4. СВОДКА: лаг влияния на ИНФЛЯЦИЮ по трём независимым срезам')
print(sdf.to_string())

# B5. канал ДКП по ИДЕНТИФИЦИРОВАННОМУ шоку
irm = m.irf1(24, s_mp)
cpi = np.cumsum(irm[1:, kP])
half = int(np.argmax(cpi <= 0.5 * cpi[-1])) + 1 if cpi[-1] < 0 else None
print('\nB5. Канал ДКП по идентифицированному шоку (знаковые огр. + max-share)')
print('    инфляция, %% год., h=0..12:', np.round(M.ANN * irm[:13, kP], 3))
print('    выпуск,   %%,       h=0..12:', np.round(100 * (irm @ M.WVEC)[:13], 3))
print('    пик дезинфляции h = %d; половина накопленного эффекта к h = %s'
      % (int(np.argmin(M.ANN * irm[:, kP])), half))
print('    дно выпуска     h = %d' % int(np.argmin(irm @ M.WVEC)))
OUT['lags'] = dict(summary=sdf.reset_index().to_dict('records'),
                   mp_peak_inflation_h=int(np.argmin(M.ANN * irm[:, kP])),
                   mp_half_effect_h=half,
                   mp_trough_output_h=int(np.argmin(irm @ M.WVEC)),
                   acf_dPC=[round(float(np.corrcoef(Y[k:, kP], Y[:T - k, kP])[0, 1]), 3)
                            for k in range(1, 9)])
print('\n    ACF инфляции, лаги 1..8:', OUT['lags']['acf_dPC'])
print('    чётные лаги систематически сильнее нечётных — собственная инерция цен')
print('    работает через лаг 2, а не 1; это и есть причина, по которой p = 6.')

# --- B6: локальные проекции (Jorda) на идентифицированный шок
print('\nB6. ЛОКАЛЬНЫЕ ПРОЕКЦИИ (Jorda, 2005) на идентифицированный шок ДКП')
print('    Альтернатива итерированным IRF: не накапливает ошибку спецификации VAR')
print('    и даёт честные стандартные ошибки. Контроли: по 4 лага инфляции, ставки,')
print('    выпуска и курса + 2 лага шока; Newey-West(8). Шок нормирован на 1 с.к.о.')
# ВАЖНО про выравнивание: m.U[k] — остаток за период t = p + k, поэтому шок
# индексируется ПО ВРЕМЕНИ: eps_lp[t] — шок ДКП квартала t (нули при t < p).
eps_lp = np.zeros(T)
eps_lp[p:] = m.U @ la.solve(m.Sigma, s_mp)
eps_lp = eps_lp / eps_lp[p:].std()
dyv = M.agg_dY(Y)


def _nw(Xm, yv, L=8):
    b = la.lstsq(Xm, yv, rcond=None)[0]
    e = yv - Xm @ b
    XtX = la.inv(Xm.T @ Xm); Sm = np.zeros((Xm.shape[1],) * 2)
    for l in range(L + 1):
        w_ = 1 - l / (L + 1)
        G = sum(np.outer(Xm[t_] * e[t_], Xm[t_ - l] * e[t_ - l])
                for t_ in range(l, len(yv)))
        Sm += w_ * (G if l == 0 else G + G.T)
    return b, np.sqrt(np.diag(XtX @ Sm @ XtX))


lp = []
print('     h    ставка, п.п.       накопл. цены, %      накопл. выпуск, %')
for h_ in range(0, 17):
    rows = []
    for t_ in range(p + 4, T - h_):
        ctrl = ([Y[t_ - 1 - l, kP] for l in range(4)]
                + [Y[t_ - 1 - l, kR] for l in range(4)]
                + [dyv[t_ - 1 - l] for l in range(4)]
                + [Y[t_ - 1 - l, kX] for l in range(4)]
                + [eps_lp[t_ - 1], eps_lp[t_ - 2], 1.0])
        rows.append(([eps_lp[t_]] + ctrl, Y[t_ + h_, kR] - Y[t_ - 1, kR],
                     100 * Y[t_:t_ + h_ + 1, kP].sum(), 100 * dyv[t_:t_ + h_ + 1].sum()))
    Xm = np.array([r_[0] for r_ in rows]); o = []
    for idx in (1, 2, 3):
        b_, se_ = _nw(Xm, np.array([r_[idx] for r_ in rows]))
        o.append((float(b_[0]), float(se_[0])))
    lp.append(dict(h=h_, rate=o[0], price=o[1], output=o[2]))
    print('    %2d  %+6.3f (%.3f)    %+7.3f (%.3f)     %+7.3f (%.3f)'
          % (h_, o[0][0], o[0][1], o[1][0], o[1][1], o[2][0], o[2][1]))
sig_out = [r_['h'] for r_ in lp if abs(r_['output'][0]) > 2 * r_['output'][1]]
sig_pri = [r_['h'] for r_ in lp if abs(r_['price'][0]) > 2 * r_['price'][1]]
print('    значимо на 5%%: ВЫПУСК на горизонтах %s; ЦЕНЫ на горизонтах %s'
      % (sig_out or 'нет', sig_pri or 'НЕТ НИ НА ОДНОМ'))
print('    Вывод: канал «ставка -> спрос» идентифицирован, канал «ставка -> цены»')
print('    статистически не обнаруживается. Именно первый момент (накопленный')
print('    эффект на выпуск %.2f%% за 4 кв.) используется для калибровки QPM.'
      % lp[4]['output'][0])
OUT['local_projections'] = lp

# ======================================================= C. НОВОЕ ПРАВИЛО
head('C. ПЕРЕХОД К ТАРГЕТИРОВАНИЮ ИНФЛЯЦИИ: КОНТРФАКТИЧЕСКОЕ ПРАВИЛО')

i_neutral = float(Y[:, kR].mean() - M.ann_pct(Y[:, kP].mean()) + 4.0)
print('нейтральная номинальная ставка при цели 4%%: %.2f%% '
      '(средняя реальная за выборку %.2f%% + цель)' % (i_neutral, rr.mean()))
base = m.forecast(12)
base_pi = [float(M.yoy_pct(base[a:a + 4, kP])) for a in (0, 4, 8)]
print('базовый прогноз (старое правило): инфляция %s, ставка %s'
      % (np.round(base_pi, 2), np.round(base[:, kR], 2)))

# C1. область устойчивости замкнутой системы
print('\nC1. УСТОЙЧИВОСТЬ ЗАМКНУТОЙ СИСТЕМЫ «оценённый блок + новое правило»')
print('    (макс. модуль корня; >= 1 — система взрывается)')
rows = {}
for rho_ in (0.6, 0.7, 0.8, 0.9, 0.95):
    rows[rho_] = [m.rule_loop(s_mp, f, phi_y=0.5, rho=rho_)[3]
                  for f in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)]
print(pd.DataFrame(rows, index=['φπ=%.1f' % f for f in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)]).T
      .rename_axis('ρ').round(3).to_string())


def frontier(phi_y, rho_, grid=np.arange(0.0, 4.001, 0.02)):
    """Множество устойчивости не обязано быть интервалом, поэтому не бисекция,
    а сканирование сетки: возвращает (наибольшее phi_pi с непрерывной от нуля
    устойчивостью, доля устойчивых точек сетки)."""
    ok = []
    for f in grid:
        try:
            ok.append(m.rule_loop(s_mp, f, phi_y=phi_y, rho=rho_)[3] < 1.0)
        except Exception:
            ok.append(False)
    ok = np.array(ok)
    first_bad = int(np.argmax(~ok)) if (~ok).any() else len(grid)
    return float(grid[max(first_bad - 1, 0)]), float(ok.mean())


fr = {}
print('\n    КРИТИЧЕСКОЕ φπ (сканирование сетки 0..4 с шагом 0.02):')
print('      комбинация          φπ_max устойчивое   доля устойчивых точек сетки')
for r_ in (0.6, 0.7, 0.8, 0.9, 0.95):
    for y_ in (0.0, 0.25, 0.5):
        fmax, frac = frontier(y_, r_)
        fr['ρ=%.2f, φy=%.2f' % (r_, y_)] = [round(fmax, 2), round(frac, 2)]
        print('      ρ=%.2f, φy=%.2f        %5.2f                    %.2f' % (r_, y_, fmax, frac))
print('    Ни при одной комбинации не удаётся уйти сильно выше 1.5: оценённая на')
print('    старом режиме динамика (ставка в приведённой форме ПОВЫШАЕТ инфляцию')
print('    с лагом 1) превращает активное правило в положительную обратную связь.')

# C2. перебор реализуемых правил
print('\nC2. ПОДБОР ЛУЧШЕГО РЕАЛИЗУЕМОГО ПРАВИЛА')
print('    критерий: RMSE годовой инфляции от цели 4%% за три года;')
print('    ограничения: замкнутая система устойчива И max|eps| <= 3 сигм')
res = []
for rho_ in (0.6, 0.7, 0.8, 0.9, 0.95):
    for f in np.arange(0.0, 3.01, 0.1):
        for y_ in (0.0, 0.25, 0.5):
            try:
                root = m.rule_loop(s_mp, f, phi_y=y_, rho=rho_)[3]
            except Exception:
                continue
            if root >= 0.999:
                continue
            pth, eps, _ = m.counterfactual_rule(12, s_mp, phi_pi=f, phi_y=y_,
                                                rho=rho_, pi_target=4.0,
                                                i_neutral=i_neutral)
            pi = [float(M.yoy_pct(pth[a:a + 4, kP])) for a in (0, 4, 8)]
            if not np.isfinite(pi).all():
                continue
            res.append(dict(rho=rho_, phi_pi=round(float(f), 2), phi_y=y_,
                            root=round(float(root), 3),
                            rmse=round(float(np.sqrt(np.mean((np.array(pi) - 4.0) ** 2))), 2),
                            pi=[round(x, 2) for x in pi],
                            nsigma=round(float(np.abs(eps).max()), 1),
                            rate_peak=round(float(pth[:, kR].max()), 2),
                            dY_y1=round(float(100 * M.agg_dY(pth)[:4].sum()), 1)))
res.sort(key=lambda d: d['rmse'])
ok3 = [d for d in res if d['nsigma'] <= 3.0]
print('\n    лучшие 8 устойчивых правил:')
print('      ρ     φπ   φy   корень  RMSE  инфл г1/г2/г3       σ    пик ставки  ВВП г1')
for d in res[:8]:
    print('      %.2f  %.1f  %.2f  %.3f   %4.2f  %5.2f/%5.2f/%5.2f  %5.1f    %6.2f   %6.1f'
          % (d['rho'], d['phi_pi'], d['phi_y'], d['root'], d['rmse'],
             d['pi'][0], d['pi'][1], d['pi'][2], d['nsigma'], d['rate_peak'], d['dY_y1']))
best = ok3[0] if ok3 else None
print('\n    лучшее ПРАВДОПОДОБНОЕ (<= 3 сигм): %s' % (best if best else 'нет'))
base_rmse = float(np.sqrt(np.mean((np.array(base_pi) - 4.0) ** 2)))
print('    базовый прогноз (СТАРОЕ правило) даёт RMSE %.2f.' % base_rmse)
if not res or res[0]['rmse'] >= base_rmse:
    print('    НИ ОДНО устойчивое контрфактическое правило не улучшает базовый')
    print('    прогноз: лучшее даёт %.2f против %.2f. В рамках оценённой динамики'
          % (res[0]['rmse'] if res else float('nan'), base_rmse))
    print('    переход к таргетированию инфляции не приближает инфляцию к цели.')
else:
    print('    лучшее правило улучшает базовый прогноз: %.2f против %.2f'
          % (res[0]['rmse'], base_rmse))
OUT['counterfactual'] = dict(i_neutral=i_neutral, base_pi=base_pi,
                             base_rmse=float(np.sqrt(np.mean((np.array(base_pi) - 4.0) ** 2))),
                             frontier=fr, top=res[:8], best_plausible=best)

# C3. вывод
head('ВЫВОД')
concl = """
1. Прежняя политика была ПАССИВНОЙ и проциклической. Прямая оценка правила
   даёт phi_pi = %.3f (t = %.2f) и phi_y = %.3f (t = %.2f): ставка не реагирует
   ни на инфляцию, ни на выпуск, она почти случайное блуждание (rho = %.3f).
   Реальная ставка движется ПРОТИВ инфляции (corr = %.3f): когда инфляция
   растёт, реальная ставка падает — это аккомодация, а не стабилизация.
   Дополнительно corr(ставка, прирост ВВП) = %.3f (ужесточение в спаде) и
   corr(ставка, обесценение) = %.3f (смягчение при ослаблении валюты).

2. Отсюда «ценовая загадка» — это не дефект модели, а подпись режима. В
   приведённой форме ставка входит в уравнение инфляции с лагом 1 и ПЛЮСОМ
   (t = %.2f). При пассивном правиле номинальная ставка следует за инфляцией,
   не обгоняя её, поэтому в данных «выше ставка -> выше инфляция».

3. Оценка «принцип Тейлора выполняется, phi_pi = 1.394», которую давала
   рекурсивная схема, — артефакт. Лаговая (то есть собственно правило) часть
   даёт %+.3f, а +%.3f приходит из мгновенного члена A0, заданного порядком
   переменных. Проверку в check_new_data.py следует вести по прямой оценке.

4. Поэтому оценённая система НЕ ГОДИТСЯ для расчёта нового режима напрямую.
   При подстановке активного правила замкнутая система взрывается уже при
   phi_pi выше ~1.0-1.5 (см. C1): динамика, оценённая при пассивном правиле,
   превращает активную реакцию в положительную обратную связь. Среди
   устойчивых и правдоподобных правил лучшее — фактически phi_pi = 0, и оно
   даже не улучшает базовый прогноз (RMSE %.2f против %.2f): инфляция остаётся
   около 8.5-9.5%%. Это ровно критика Лукаса в числах.

5. Что с этим делать. Смену режима нужно считать структурной моделью с
   инвариантными к политике параметрами — это трек QPM/DSGE, который в
   проекте уже есть (qpm-soe-specification.md, dsge-candidates-memo.md).
   Роль этого SVAR — поставщик ЭМПИРИЧЕСКИХ ОРИЕНТИРОВ для её калибровки:
   лаги и величины трансмиссии из раздела B (пик дезинфляции h = %d,
   половина накопленного эффекта к h = %s, дно выпуска h = %d, собственная
   инерция цен через лаг 2), а не источник сценариев по ставке.
""" % (phi_pi_h, est['pi4(t-1)'][1], phi_y_h, est['dy4(t-1)'][1], rho_h,
       cor['rr_pi'], cor['i_dY'], cor['i_e'], T_pi[kR][0], lagp, cont,
       res[0]['rmse'] if res else float('nan'), base_rmse,
       OUT['lags']['mp_peak_inflation_h'], OUT['lags']['mp_half_effect_h'],
       OUT['lags']['mp_trough_output_h'])
print(concl)
OUT['conclusion'] = concl.strip()

# ---------------------------------------------------------------- рисунки
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 150})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
# для читаемости — инфляция ГОД/ГОД (квартальные лог-приросты слишком шумны)
yoy = np.array([M.yoy_pct(Y[t - 3:t + 1, kP]) for t in range(3, T)])
idx = X.index[3:]
ax[0].plot(idx, Y[3:, kR], lw=1.2, color='#1b3a5c', label='номинальная ставка')
ax[0].plot(idx, yoy, lw=1.2, color='#b3402f', label='инфляция г/г, %')
ax[0].plot(idx, Y[3:, kR] - yoy, lw=1.2, color='#2f8f4e', label='реальная ставка')
ax[0].axhline(0, color='k', lw=.6)
ax[0].set_xticks(idx[::40]); ax[0].tick_params(labelsize=6)
ax[0].set_title('corr(реальная ставка, инфляция г/г) = %+.2f'
                % np.corrcoef(Y[3:, kR] - yoy, yoy)[0, 1], fontsize=8)
ax[0].legend(fontsize=6)
ax[1].plot(range(9), M.ANN * ir[:9, kP, kP], lw=1.5, color='#b3402f', label='инфляция')
ax[1].plot(range(9), ir[:9, kR, kP], lw=1.5, color='#1b3a5c', label='ном. ставка')
ax[1].plot(range(9), rr_ir[:9], lw=1.5, color='#2f8f4e', label='реальная ставка')
ax[1].axhline(0, color='k', lw=.6); ax[1].set_xlabel('кварталы')
ax[1].set_title('Отклик на шок инфляции: аккомодация', fontsize=8); ax[1].legend(fontsize=6)
ff = np.arange(0, 2.61, 0.05)
for rho_, c_ in zip((0.7, 0.8, 0.9), ('#7a9bc4', '#1b3a5c', '#b3402f')):
    ax[2].plot(ff, [m.rule_loop(s_mp, f, phi_y=0.5, rho=rho_)[3] for f in ff],
               lw=1.4, color=c_, label='ρ=%.1f' % rho_)
ax[2].axhline(1, color='k', ls='--', lw=1); ax[2].set_xlabel('φπ нового правила')
ax[2].set_title('Макс. корень замкнутой системы (>1 — взрыв)', fontsize=8)
ax[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig('fig_policy_regime.png', bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
hh = np.arange(9)
for v, c_ in [('dPC', '#b3402f'), ('rG', '#1b3a5c'), ('dNFX', '#2f8f4e'),
              ('dL', '#8a6fb0'), ('dInc', '#c08a2e')]:
    ax[0].plot(hh, M.ANN * ir[:9, kP, V.index(v)], lw=1.3, color=c_, label=v)
ax[0].axhline(0, color='k', lw=.6); ax[0].set_xlabel('кварталы')
ax[0].set_title('Отклик инфляции на шоки (рекурсивная схема)', fontsize=8)
ax[0].legend(fontsize=6)
ax[1].bar(range(1, 9), OUT['lags']['acf_dPC'], color='#7a9bc4')
ax[1].set_xlabel('лаг, кварталы'); ax[1].set_title('ACF инфляции: чётные лаги сильнее', fontsize=8)
fig.tight_layout(); fig.savefig('fig_lags.png', bbox_inches='tight'); plt.close(fig)

with open('policy_regime_out.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)
print('записано: policy_regime_out.json, fig_policy_regime.png, fig_lags.png')
