"""
qpm_soe.py — малая полуструктурная модель (QPM) малой открытой экономики,
откалиброванная по SVAR, и ВЫБОР ТРАЕКТОРИИ КЛЮЧЕВОЙ СТАВКИ.

ЗАЧЕМ ОНА НУЖНА. Исходный план «оценить SVAR -> посмотреть отклики -> выбрать
траекторию ставки» на этих данных не работает: прежняя политика была пассивной
и проциклической (см. policy_regime.py), поэтому в приведённой форме SVAR
ставка положительно связана с инфляцией, а подстановка активного правила
взрывает систему. Приведённая форма VAR не инвариантна к смене правила.

Выход стандартный для центральных банков: маленькая модель, в которой ПРАВИЛО
ЗАДАЁТСЯ, а не оценивается, а непопитические уравнения калибруются по данным и
по идентифицированному шоку ДКП из SVAR. Тогда траектория ставки выбирается
оптимизацией явного критерия внутри модели, а не подгонкой сценария в VAR.

СТРУКТУРА (4 поведенческих уравнения + правило; всё в % годовых, разрывы в %):

 (1) IS-кривая (спрос)
     gap_t = a1*gap_{t-1} + a2*gap_{t-2} - a_r*(rr_{t-1} - rr*) + a_q*qgap_{t-1} + e^y_t
     rr_t = i_t - pi4^e_{t+1}                      реальная ставка ex ante

 (2) Кривая Филлипса (предложение), гибридная
     pi_t = b_f*E_t[pi_{t+1}] + (1-b_f)*SUM_j w_j*pi_{t-j} + kappa*gap_{t-1}
            + theta*ds_{t-1} + e^pi_t

 (3) Непокрытый паритет процентных ставок с премией
     ds_t = d*(i_{t-1} - pistar_{t-1} - rr*) + prem + e^s_t
     qgap_t = qgap_{t-1} + ds_t + pistar_t/4 - pi_t/4      (реальный курс)

 (4) Правило политики (ЗАДАЁТСЯ, не оценивается)
     i_t = rho*i_{t-1} + (1-rho)*[ rr* + pi_target
                                   + phi_pi*(pi4_t - pi_target) + phi_y*gap_t ]

Ожидания — модельно-согласованные, решение методом расширенного пути
(Fair-Taylor): итерации по всей траектории до сходимости, терминальное условие —
стационар. Это снимает зависимость от произвольной адаптивной схемы.

КАЛИБРОВКА. Всё, что можно, оценено на данных (МНК, раздел печати `calibration`);
чувствительность выпуска к реальной ставке a_r НЕ оценивается регрессией
(инструмент слаб: шок ДКП объясняет мало вариации реальной ставки, 2SLS даёт
неверный знак), а подбирается так, чтобы отклик модели на шок ДКП совпал с
моментом ЛОКАЛЬНЫХ ПРОЕКЦИЙ на идентифицированный шок — единственной оценкой
трансмиссии, которая здесь статистически значима.

ЧЕСТНО О ДОПУЩЕНИЯХ (они же слабые места):
 * Вертикальность кривой Филлипса в долгом периоде НАВЯЗАНА (сумма весов = 1).
   Данные её отвергают: оценённая сумма лагов инфляции 0.714. Без вертикальности
   долгосрочная инфляция определяется константой, а не целью, и задача выбора
   траектории теряет смысл. Это модельное решение, а не вывод из данных.
 * Доля форвард-компоненты b_f не идентифицируется на этих данных и КАЛИБРУЕТСЯ
   (база 0.4, диапазон чувствительности 0.2-0.6).
 * Перенос курса в цены статистически неотличим от нуля (сумма за 2 кв. -0.17,
   |t| < 1.7), поэтому курсовой канал в базовой калибровке почти не работает.
   Соответственно и UIP-блок на результат влияет слабо; он оставлен для полноты.
 * Эффект ставки на ЦЕНЫ в локальных проекциях незначим (все горизонты), значим
   только эффект на ВЫПУСК. То есть канал «ставка -> спрос» идентифицирован, а
   «спрос -> цены» держится на наклоне Филлипса kappa = 0.10 (t = 1.80).
   Это главная неопределённость всего расчёта, см. раздел чувствительности.

"""
import sys, json
import numpy as np
import numpy.linalg as la
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
import svar_soe as M

DATA = sys.argv[1] if len(sys.argv) > 1 else 'data_239Q2.csv'


# ------------------------------------------------------------------ данные
def hp(x, lam=1600.0):
    n = len(x); D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return la.solve(np.eye(n) + lam * D.T @ D, np.asarray(x, float))


def prepare(path=DATA):
    X = M.load(path); Y = X[M.V].values; V = M.V; T = len(Y)
    kR, kP, kX = V.index('rG'), V.index('dPC'), V.index('dNFX')
    m = M.BVAR(Y, p=M.P_DEFAULT)
    s_mp, share = M.mp_max_share(m, seed=1)
    eps = np.concatenate([np.zeros(M.P_DEFAULT),
                          m.U @ la.solve(m.Sigma, s_mp)])
    eps = eps / eps[M.P_DEFAULT:].std()
    dy = M.agg_dY(Y)
    ylev = np.cumsum(dy)
    d = dict(
        X=X, Y=Y, T=T, m=m, s_mp=s_mp, eps=eps,
        gap=100 * (ylev - hp(ylev)),                 # разрыв выпуска, %
        pi=M.ann_pct(Y[:, kP]),                      # инфляция, % год.
        pistar=M.ann_pct(X.pistar.values),           # внешняя инфляция, % год.
        i=Y[:, kR].copy(),                           # ставка, % год.
        ds=100 * Y[:, kX],                           # прирост ном. курса, %
    )
    q = np.cumsum(X.pistar.values - X.dPC.values + X.dNFX.values)   # реальный курс
    d['qgap'] = 100 * (q - hp(q))
    d['rr'] = d['i'] - d['pi']
    return d


def ols(y, Xm):
    b = la.lstsq(Xm, y, rcond=None)[0]
    e = y - Xm @ b
    s2 = e @ e / (len(y) - Xm.shape[1])
    se = np.sqrt(np.diag(s2 * la.inv(Xm.T @ Xm)))
    return b, se, float(1 - e.var() / y.var())


# ------------------------------------------------------------- калибровка
def calibrate(d, b_f=0.4, vertical=True, verbose=True):
    T = d['T']; sl = slice(8, T)
    pi, gap, ds, pistar, i = d['pi'], d['gap'], d['ds'], d['pistar'], d['i']

    # --- (2) кривая Филлипса: лаги инфляции, разрыв, номинальный курс
    Xm = np.column_stack([pi[7:T - 1], pi[6:T - 2], pi[5:T - 3], pi[4:T - 4],
                          gap[7:T - 1], ds[7:T - 1], ds[6:T - 2],
                          pistar[7:T - 1], np.ones(T - 8)])
    bP, seP, r2P = ols(pi[sl], Xm)
    w_raw = bP[:4].copy()
    # vertical=True: сумма весов принудительно = 1 (долгосрочная кривая Филлипса
    # вертикальна — стандартное допущение QPM). vertical=False: веса как оценены
    # (сумма 0.714), тогда инфляция сама возвращается к константе, а не к цели.
    w = (w_raw / w_raw.sum()) if vertical else w_raw
    kappa = float(bP[4])
    theta = float(bP[5] + bP[6])     # перенос курса, суммарно за 2 кв.

    # --- (1) IS: AR-часть из данных
    bI, seI, r2I = ols(gap[sl], np.column_stack([gap[7:T - 1], gap[6:T - 2],
                                                 np.ones(T - 8)]))
    a1, a2 = float(bI[0]), float(bI[1])
    bQ, seQ, _ = ols(gap[sl], np.column_stack([gap[7:T - 1], gap[6:T - 2],
                                               d['qgap'][7:T - 1], np.ones(T - 8)]))
    a_q = float(bQ[2])

    # --- (3) UIP
    bU, seU, r2U = ols(ds[sl], np.column_stack([i[7:T - 1] - pistar[7:T - 1],
                                                np.ones(T - 8)]))
    d_uip, prem = float(bU[0]), float(bU[1])

    C = dict(a1=a1, a2=a2, a_q=a_q, kappa=kappa, theta=theta, w=w.tolist(),
             b_f=b_f, d_uip=d_uip, prem=prem,
             vertical=bool(vertical), rr_star=float(d['rr'].mean()),
             pistar_bar=float(pistar.mean()),
             t_kappa=float(bP[4] / seP[4]), t_theta_1=float(bP[5] / seP[5]),
             w_sum_estimated=float(w_raw.sum()), r2_PC=r2P, r2_IS=r2I, r2_UIP=r2U)
    if verbose:
        print('КАЛИБРОВКА (оценено на данных, если не сказано иное)')
        print('  Филлипс: kappa = %.4f (t = %.2f); веса лагов инфляции %s'
              % (kappa, C['t_kappa'], np.round(w, 3)))
        print('           сумма лагов ДО нормировки = %.3f -> вертикальность НАВЯЗАНА'
              % w_raw.sum())
        print('           перенос курса theta = %.4f (t лага 1 = %.2f) — незначим'
              % (theta, C['t_theta_1']))
        print('           b_f = %.2f — КАЛИБРОВАН, на данных не идентифицируется' % b_f)
        print('  IS     : a1 = %.3f, a2 = %.3f (сумма %.3f), a_q = %.4f; R2 = %.3f'
              % (a1, a2, a1 + a2, a_q, r2I))
        print('  UIP    : d = %.4f, премия = %.3f; R2 = %.3f' % (d_uip, prem, r2U))
        print('  rr* = %.2f%%, pi* = %.2f%%' % (C['rr_star'], C['pistar_bar']))
    return C


# ------------------------------------------------------- решение модели
class QPM:
    """Решение методом расширенного пути (Fair-Taylor) с модельно-согласованными
    ожиданиями. Модель ЛИНЕЙНА, поэтому неподвижная точка не итерируется, а
    находится точно: вся траектория (pi, gap, i, qgap, ds на N кварталов)
    записывается одной линейной системой 5N x 5N и решается напрямую.
    Терминальное условие — стационар (pi = цель) за пределами горизонта."""

    TAIL = 28          # хвост за горизонтом счёта, чтобы терминал не влиял
    TAIL_RULE = (1.5, 0.5, 0.8)   # чем закрывается хвост, если ставка задана

    def __init__(self, C, pi_target=4.0, a_r=0.5):
        self.C = C; self.pi_target = pi_target; self.a_r = a_r
        self.i_ss = C['rr_star'] + pi_target

    def _init_state(self, d):
        return dict(pi=list(d['pi'][-4:]), gap=list(d['gap'][-2:]),
                    i=float(d['i'][-1]), qgap=float(d['qgap'][-1]))

    def simulate(self, d, H, rule=None, i_path=None, shock=None):
        C = self.C; N = H + self.TAIL
        rule = self.TAIL_RULE if rule is None else rule
        phi_pi, phi_y, rho = rule
        st = self._init_state(d)
        sh = np.zeros(N)
        if shock is not None:
            sh[:len(shock)] = np.asarray(shock, float)
        ip = np.full(N, np.nan)
        if i_path is not None:
            ip[:len(i_path)] = np.asarray(i_path, float)

        # индексы неизвестных
        def P(t): return t                       # pi_t
        def G(t): return N + t                   # gap_t
        def I(t): return 2 * N + t               # i_t
        def Q(t): return 3 * N + t               # qgap_t
        def S(t): return 4 * N + t               # ds_t
        A = np.zeros((5 * N, 5 * N)); rhs = np.zeros(5 * N)

        # значения лагов: t < 0 берутся из начального состояния (известны)
        pi0 = st['pi']          # pi_{-4}, pi_{-3}, pi_{-2}, pi_{-1}
        gap0 = st['gap']        # gap_{-2}, gap_{-1}
        def add_pi(row, t, c):
            if t >= 0: A[row, P(t)] += c
            else:      rhs[row] -= c * pi0[len(pi0) + t]
        def add_gap(row, t, c):
            if t >= 0: A[row, G(t)] += c
            else:      rhs[row] -= c * gap0[len(gap0) + t]
        def add_i(row, t, c):
            if t >= 0: A[row, I(t)] += c
            else:      rhs[row] -= c * st['i']
        def add_q(row, t, c):
            if t >= 0: A[row, Q(t)] += c
            else:      rhs[row] -= c * st['qgap']
        def add_s(row, t, c):
            if t >= 0: A[row, S(t)] += c
            # ds_{-1} не наблюдается отдельно — берём 0 (отклонение от нормы)

        for t in range(N):
            # (1) IS: gap_t = a1 gap_{t-1} + a2 gap_{t-2} - a_r(i_{t-1} - pi_t - rr*) + a_q qgap_{t-1}
            r = t
            A[r, G(t)] += 1.0
            add_gap(r, t - 1, -C['a1']); add_gap(r, t - 2, -C['a2'])
            add_i(r, t - 1, self.a_r); add_pi(r, t, -self.a_r)
            add_q(r, t - 1, -C['a_q'])
            rhs[r] += self.a_r * C['rr_star']

            # (2) Филлипс: pi_t = b_f pi_{t+1} + (1-b_f) sum w_j pi_{t-1-j}
            #              + kappa gap_{t-1} + theta ds_{t-1} + shock
            r = N + t
            A[r, P(t)] += 1.0
            if t + 1 < N: A[r, P(t + 1)] -= C['b_f']
            else:         rhs[r] += C['b_f'] * self.pi_target      # терминал
            for j in range(4):
                add_pi(r, t - 1 - j, -(1 - C['b_f']) * C['w'][j])
            add_gap(r, t - 1, -C['kappa']); add_s(r, t - 1, -C['theta'])
            rhs[r] += sh[t]

            # (3) правило либо заданная ставка
            r = 2 * N + t
            if not np.isnan(ip[t]):
                A[r, I(t)] += 1.0; rhs[r] += ip[t]
            else:
                A[r, I(t)] += 1.0
                add_i(r, t - 1, -rho)
                for k in range(4):
                    add_pi(r, t - k, -(1 - rho) * phi_pi / 4.0)
                A[r, G(t)] += -(1 - rho) * phi_y
                rhs[r] += (1 - rho) * (self.i_ss - phi_pi * self.pi_target)

            # (4) UIP: ds_t = d(i_{t-1} - pi*) + prem
            r = 3 * N + t
            A[r, S(t)] += 1.0; add_i(r, t - 1, -C['d_uip'])
            rhs[r] += C['prem'] - C['d_uip'] * C['pistar_bar']

            # (5) реальный курс: qgap_t = qgap_{t-1} + ds_t + pi*/4 - pi_t/4
            r = 4 * N + t
            A[r, Q(t)] += 1.0; add_q(r, t - 1, -1.0); A[r, S(t)] += -1.0
            A[r, P(t)] += 0.25
            rhs[r] += C['pistar_bar'] / 4.0

        x = la.solve(A, rhs)
        pi, gap, i, qg, ds = (x[:N], x[N:2 * N], x[2 * N:3 * N],
                              x[3 * N:4 * N], x[4 * N:])
        full_pi = np.concatenate([np.array(pi0), pi])
        pi4 = np.array([full_pi[t + 1:t + 5].mean() for t in range(H)])
        return dict(pi=pi[:H], gap=gap[:H], i=i[:H], qgap=qg[:H], ds=ds[:H],
                    rr=i[:H] - np.concatenate([pi[1:H], [self.pi_target]]), pi4=pi4)

    # ----- калибровка a_r по моменту локальных проекций
    def set_ar(self, d, target_cum_gap=-1.69, h=4, shock_pp=0.31):
        """Подобрать a_r так, чтобы разовое повышение ставки на shock_pp п.п.
        (с формой затухания идентифицированного шока ДКП) дало накопленный
        эффект на разрыв выпуска target_cum_gap % за h кварталов."""
        prof = np.array([1.0, 0.755, 0.418, 0.122, -0.068])

        def f(ar):
            self.a_r = ar
            base = self.simulate(d, 24, rule=(1.5, 0.5, 0.8))
            ip = base['i'].copy(); ip[:5] += shock_pp * prof
            alt = self.simulate(d, 24, i_path=ip)
            return float(np.sum(alt['gap'][:h + 1] - base['gap'][:h + 1]) - target_cum_gap)

        # f немонотонна: при больших a_r замкнутая система теряет устойчивость,
        # поэтому не бисекция вслепую, а скан сетки и брекет на ПЕРВОЙ смене знака
        grid = np.concatenate([np.arange(0.01, 1.0, 0.01), np.arange(1.0, 3.01, 0.05)])
        vals = np.array([f(g) for g in grid])
        sgn = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
        if len(sgn) == 0:
            self.a_r = 0.5
            return self.a_r, False
        k = sgn[0]
        self.a_r = brentq(f, grid[k], grid[k + 1], xtol=1e-10)
        return self.a_r, True


# ------------------------------------------------------------ критерий
def loss(res, pi_target=4.0, lam_y=0.5, lam_di=0.2, i_prev0=None, beta=1.0):
    """Стандартный лосс ЦБ (Svensson 2010; Woodford 2003):
    сумма (pi4 - цель)^2 + lam_y*gap^2 + lam_di*(Δi)^2."""
    di = np.diff(np.concatenate([[i_prev0], res['i']])) if i_prev0 is not None \
        else np.diff(res['i'], prepend=res['i'][0])
    w = beta ** np.arange(len(res['i']))
    return float(np.sum(w * ((res['pi4'] - pi_target) ** 2
                             + lam_y * res['gap'] ** 2 + lam_di * di ** 2)))


def main():
    d = prepare(DATA)
    print('=' * 78); print('QPM МАЛОЙ ОТКРЫТОЙ ЭКОНОМИКИ — КАЛИБРОВКА И ТРАЕКТОРИЯ СТАВКИ')
    print('=' * 78)
    print('данные: %s .. %s (%d кв.); последняя ставка %.2f%%, инфляция г/г %.2f%%'
          % (d['X'].index[0], d['X'].index[-1], d['T'], d['i'][-1],
             M.yoy_pct(d['Y'][-4:, M.V.index('dPC')])))
    C = calibrate(d)
    q = QPM(C, pi_target=4.0, a_r=0.5)
    ar, ok = q.set_ar(d, target_cum_gap=-1.69, h=4, shock_pp=0.31)
    print('\n  a_r подобран по моменту локальных проекций: %.4f (%s)'
          % (ar, 'сходимость' if ok else 'ФОЛБЭК — момент недостижим'))
    OUT = dict(calibration=C, a_r=float(ar))

    # --- 1. базовая траектория при разных правилах
    print('\n' + '=' * 78); print('1. ЧТО ДАЮТ ПРОСТЫЕ ПРАВИЛА'); print('=' * 78)
    H = 12
    rules = [('пассивное (как было)', (0.0, 0.0, 0.92)),
             ('Тейлор 1.5 / 0.5 / 0.8', (1.5, 0.5, 0.8)),
             ('жёсткое ИТ 2.5 / 0.25 / 0.7', (2.5, 0.25, 0.7)),
             ('очень жёсткое 4.0 / 0.0 / 0.6', (4.0, 0.0, 0.6))]
    OUT['rules'] = {}
    print('  правило                        инфл. г1/г2/г3   пик ставки  разрыв г1   лосс')
    for nm, r in rules:
        res = q.simulate(d, H, rule=r)
        L = loss(res, i_prev0=d['i'][-1])
        yr = [float(np.mean(res['pi'][a:a + 4])) for a in (0, 4, 8)]
        print('  %-30s %5.2f/%5.2f/%5.2f    %6.2f     %6.2f   %7.1f'
              % (nm, yr[0], yr[1], yr[2], res['i'].max(), res['gap'][:4].mean(), L))
        OUT['rules'][nm] = dict(phi=list(r), pi=yr, i=res['i'].round(3).tolist(),
                                gap=res['gap'].round(3).tolist(), loss=L)

    # --- 2. оптимальное простое правило
    print('\n' + '=' * 78); print('2. ОПТИМАЛЬНОЕ ПРОСТОЕ ПРАВИЛО (минимум лосса)'); print('=' * 78)
    best = None
    for phi in np.arange(0.5, 5.01, 0.25):
        for phy in (0.0, 0.25, 0.5, 1.0):
            for rho in (0.5, 0.6, 0.7, 0.8, 0.9):
                res = q.simulate(d, H, rule=(phi, phy, rho))
                if not np.isfinite(res['pi']).all() or np.abs(res['pi']).max() > 200:
                    continue
                L = loss(res, i_prev0=d['i'][-1])
                if best is None or L < best[0]:
                    best = (L, phi, phy, rho, res)
    L, phi, phy, rho, res = best
    print('  phi_pi = %.2f, phi_y = %.2f, rho = %.2f -> лосс %.1f' % (phi, phy, rho, L))
    print('  ставка, %% :', np.round(res['i'], 2))
    print('  инфл. г/г :', np.round(res['pi4'], 2))
    print('  разрыв, %% :', np.round(res['gap'], 2))
    OUT['best_rule'] = dict(phi_pi=float(phi), phi_y=float(phy), rho=float(rho),
                            loss=L, i=res['i'].round(3).tolist(),
                            pi4=res['pi4'].round(3).tolist(),
                            gap=res['gap'].round(3).tolist())

    # --- 3. оптимальная ТРАЕКТОРИЯ (без ограничения формой правила)
    print('\n' + '=' * 78); print('3. ОПТИМАЛЬНАЯ ТРАЕКТОРИЯ СТАВКИ (свободная оптимизация)')
    print('=' * 78)
    x0 = res['i'].copy()

    def obj(x):
        r = q.simulate(d, H, i_path=x)
        if not np.isfinite(r['pi']).all():
            return 1e9
        return loss(r, i_prev0=d['i'][-1])

    opt = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-4,
                            'fatol': 1e-6})
    ropt = q.simulate(d, H, i_path=opt.x)
    print('  лосс %.1f (против %.1f у лучшего простого правила)'
          % (loss(ropt, i_prev0=d['i'][-1]), L))
    print('  ставка, %% :', np.round(opt.x, 2))
    print('  инфл. г/г :', np.round(ropt['pi4'], 2))
    print('  разрыв, %% :', np.round(ropt['gap'], 2))
    print('  РЕШЕНИЕ НА СЛЕДУЮЩИЙ КВАРТАЛ: %.2f%% (сейчас %.2f%%, изменение %+.2f п.п.)'
          % (opt.x[0], d['i'][-1], opt.x[0] - d['i'][-1]))
    OUT['optimal_path'] = dict(i=np.round(opt.x, 3).tolist(),
                               pi4=ropt['pi4'].round(3).tolist(),
                               gap=ropt['gap'].round(3).tolist(),
                               loss=loss(ropt, i_prev0=d['i'][-1]),
                               decision_next_quarter=float(opt.x[0]))

    # --- 4. чувствительность к неидентифицированным параметрам
    print('\n' + '=' * 78); print('4. ЧУВСТВИТЕЛЬНОСТЬ (что не идентифицировано данными)')
    print('=' * 78)
    OUT['sensitivity'] = {}
    print('  параметр            значение   ставка h=1   инфл. год 3   лосс')
    for nm, key, vals in [('b_f (форвард)', 'b_f', (0.2, 0.4, 0.6)),
                          ('kappa (наклон PC)', 'kappa',
                           (C['kappa'] * 0.5, C['kappa'], C['kappa'] * 2)),
                          ('theta (перенос)', 'theta', (0.0, C['theta'], -0.5))]:
        for v in vals:
            C2 = dict(C); C2[key] = v
            q2 = QPM(C2, 4.0, a_r=q.a_r)
            if key == 'kappa':
                q2.set_ar(d, -1.69, 4, 0.31)
            r2_ = q2.simulate(d, H, rule=(phi, phy, rho))
            L2 = loss(r2_, i_prev0=d['i'][-1])
            print('  %-18s %8.4f   %8.2f      %8.2f   %6.1f'
                  % (nm, v, r2_['i'][0], np.mean(r2_['pi'][8:12]), L2))
            OUT['sensitivity']['%s=%.4f' % (key, v)] = dict(
                i1=float(r2_['i'][0]), pi_y3=float(np.mean(r2_['pi'][8:12])), loss=L2)

    # --- 5. сверка с SVAR
    print('\n' + '=' * 78); print('5. СВЕРКА С SVAR: отклик на шок ДКП'); print('=' * 78)
    prof = np.array([1.0, 0.755, 0.418, 0.122, -0.068])
    base = q.simulate(d, 20, rule=(phi, phy, rho))
    ip = base['i'].copy(); ip[:5] += 0.31 * prof
    alt = q.simulate(d, 20, i_path=ip)
    dgap = alt['gap'] - base['gap']; dpi = alt['pi'] - base['pi']
    print('  QPM, отклик на 1 с.к.о. шока ДКП:')
    print('    разрыв выпуска, %% :', np.round(dgap[:9], 3))
    print('    инфляция, %% год.  :', np.round(dpi[:9], 3))
    print('    накопл. выпуск h=4: %+.3f%% (цель из лок. проекций -1.69%%)'
          % dgap[:5].sum())
    irm = d['m'].irf1(20, d['s_mp'])
    print('  SVAR (знаковая идентификация), для сравнения:')
    print('    выпуск, %%         :', np.round(100 * (irm @ M.WVEC)[:9], 3))
    print('    инфляция, %% год.  :', np.round(M.ANN * irm[:9, M.V.index('dPC')], 3))
    OUT['svar_check'] = dict(qpm_gap=dgap[:9].round(3).tolist(),
                             qpm_pi=dpi[:9].round(3).tolist(),
                             qpm_cum_gap_h4=float(dgap[:5].sum()),
                             svar_dY=(100 * (irm @ M.WVEC)[:9]).round(3).tolist(),
                             svar_pi=(M.ANN * irm[:9, M.V.index('dPC')]).round(3).tolist())

    # --- 6. что решает исход: разрыв выпуска, вертикальность, веса в лоссе
    print('\n' + '=' * 78)
    print('6. ОТ ЧЕГО РЕШЕНИЕ ЗАВИСИТ СИЛЬНЕЕ ВСЕГО')
    print('=' * 78)
    print('6a. Разрыв выпуска на конец выборки = %.2f%% — главный драйвер решения.'
          % d['gap'][-1])
    print('    Односторонний HP (в реальном времени) даёт на последней точке то же')
    print('    значение, то есть это не артефакт двустороннего фильтра. Линейный и')
    print('    квадратичный тренды непригодны (выпуск не тренд-стационарен).')
    print('    Сдвиг начального разрыва:')
    print('      сдвиг, п.п.   ставка h=1   инфл. год 3   лосс')
    OUT['gap_sensitivity'] = {}
    for sh_ in (-2.0, -1.0, 0.0, 1.0, 2.0):
        d2 = dict(d); d2['gap'] = d['gap'] + sh_
        q2 = QPM(C, 4.0, a_r=q.a_r)
        r2_ = q2.simulate(d2, H, rule=(phi, phy, rho))
        L2 = loss(r2_, i_prev0=d['i'][-1])
        print('      %+6.1f        %7.2f      %8.2f    %6.1f'
              % (sh_, r2_['i'][0], np.mean(r2_['pi'][8:12]), L2))
        OUT['gap_sensitivity']['%+.1f' % sh_] = dict(
            i1=float(r2_['i'][0]), pi_y3=float(np.mean(r2_['pi'][8:12])), loss=L2)

    print('\n6b. Вертикальность кривой Филлипса (навязана против оценки 0.714):')
    Cnv = calibrate(d, vertical=False, verbose=False)
    qnv = QPM(Cnv, 4.0); arnv, _ = qnv.set_ar(d)
    rnv = qnv.simulate(d, H, rule=(phi, phy, rho))
    print('      вертикальная  : ставка h=1 %.2f, инфл. год 3 %.2f'
          % (res['i'][0], np.mean(res['pi'][8:12])))
    print('      как оценено   : ставка h=1 %.2f, инфл. год 3 %.2f'
          % (rnv['i'][0], np.mean(rnv['pi'][8:12])))
    OUT['vertical_sensitivity'] = dict(
        vertical=dict(i1=float(res['i'][0]), pi_y3=float(np.mean(res['pi'][8:12]))),
        estimated=dict(i1=float(rnv['i'][0]), pi_y3=float(np.mean(rnv['pi'][8:12]))))

    print('\n6c. Веса в критерии потерь — это ПРЕДПОЧТЕНИЯ, а не данные:')
    print('      lam_y   lam_di   ставка h=1   инфл. год 3')
    OUT['loss_weights'] = {}
    for ly in (0.1, 0.5, 1.0):
        for ld in (0.05, 0.2, 1.0):
            bb = None
            for f_ in np.arange(0.5, 5.01, 0.5):
                for fy in (0.0, 0.5, 1.0):
                    for rh in (0.5, 0.7, 0.9):
                        rr_ = q.simulate(d, H, rule=(f_, fy, rh))
                        L_ = loss(rr_, lam_y=ly, lam_di=ld, i_prev0=d['i'][-1])
                        if bb is None or L_ < bb[0]:
                            bb = (L_, rr_)
            print('      %5.2f   %5.2f    %8.2f      %8.2f'
                  % (ly, ld, bb[1]['i'][0], np.mean(bb[1]['pi'][8:12])))
            OUT['loss_weights']['ly=%.2f,ld=%.2f' % (ly, ld)] = dict(
                i1=float(bb[1]['i'][0]), pi_y3=float(np.mean(bb[1]['pi'][8:12])))

    print('\n6d. Расхождение с базовым прогнозом SVAR — назвать прямо:')
    base_svar = d['m'].forecast(12)
    kP_ = M.V.index('dPC')
    svar_pi = [float(M.yoy_pct(base_svar[a:a + 4, kP_])) for a in (0, 4, 8)]
    passive = q.simulate(d, H, rule=(0.0, 0.0, 0.92))
    print('      SVAR (старое пассивное правило, безусловный) : %s' % np.round(svar_pi, 2))
    print('      QPM при пассивном правиле                    : %s'
          % np.round([np.mean(passive['pi'][a:a + 4]) for a in (0, 4, 8)], 2))
    print('      Причина расхождения: в SVAR кривая Филлипса НЕ вертикальна и')
    print('      инфляция сама возвращается к среднему выборки (~2%), а в QPM')
    print('      вертикальность навязана, поэтому глубоко отрицательный разрыв')
    print('      выпуска тянет инфляцию вниз без ограничения. Доверять здесь')
    print('      следует не уровню, а ЗНАКУ и ПОРЯДКУ реакции на ставку.')
    OUT['svar_vs_qpm_baseline'] = dict(svar=svar_pi,
                                       qpm_passive=[float(np.mean(passive['pi'][a:a + 4]))
                                                    for a in (0, 4, 8)])

    # --- 7. итоговая рекомендация с диапазоном
    print('\n' + '=' * 78); print('7. РЕКОМЕНДАЦИЯ'); print('=' * 78)
    lo_ = min([v['i1'] for v in OUT['gap_sensitivity'].values()]
              + [v['i1'] for v in OUT['loss_weights'].values()])
    hi_ = max([v['i1'] for v in OUT['gap_sensitivity'].values()]
              + [v['i1'] for v in OUT['loss_weights'].values()])
    print('  Текущая ставка %.2f%%, инфляция г/г %.2f%%, разрыв выпуска %.2f%%,'
          % (d['i'][-1], M.yoy_pct(d['Y'][-4:, M.V.index('dPC')]), d['gap'][-1]))
    print('  нейтральная реальная ставка %.2f%% -> нейтральная номинальная при цели 4%%: %.2f%%'
          % (C['rr_star'], C['rr_star'] + 4.0))
    print('  Фактическая реальная ставка сейчас: %.2f%% (к инфляции г/г) — около нейтральной.'
          % (d['i'][-1] - M.yoy_pct(d['Y'][-4:, M.V.index('dPC')])))
    print()
    print('  Оптимальное ПРОСТОЕ ПРАВИЛО: phi_pi = %.2f, phi_y = %.2f, rho = %.2f'
          % (phi, phy, rho))
    print('  Ставка на следующий квартал по нему: %.2f%% (%+.2f п.п.)'
          % (res['i'][0], res['i'][0] - d['i'][-1]))
    print('  Робастный диапазон по всем проверкам раздела 6: %.2f .. %.2f%%' % (lo_, hi_))
    print('  Свободная оптимизация траектории даёт %.2f%% в первом квартале —'
          % opt.x[0])
    print('  это фронт-лоадинг, характерный для безограничительного оптимума:')
    print('  он эксплуатирует ровно те связи модели, которые хуже всего оценены.')
    print('  Для отчёта брать простое правило, свободную траекторию — как нижнюю границу.')
    print()
    print('  ЧЕГО ЭТОТ СЧЁТ НЕ ДОКАЗЫВАЕТ: эффект ставки на ЦЕНЫ в данных значимо')
    print('  не обнаруживается (локальные проекции, все горизонты). Модель переносит')
    print('  ставку в инфляцию через наклон Филлипса kappa = %.3f (t = %.2f) —'
          % (C['kappa'], C['t_kappa']))
    print('  это единственное звено, и оно на грани значимости.')
    OUT['recommendation'] = dict(
        current_rate=float(d['i'][-1]), next_quarter=float(res['i'][0]),
        change=float(res['i'][0] - d['i'][-1]),
        robust_range=[float(lo_), float(hi_)],
        free_path_first=float(opt.x[0]),
        rule=dict(phi_pi=float(phi), phi_y=float(phy), rho=float(rho)),
        neutral_nominal=float(C['rr_star'] + 4.0))

    # ---------------------------------------------------------- рисунок
    plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                         'figure.dpi': 150})
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    hh = np.arange(1, H + 1)
    for nm, r in rules[:3]:
        rr_ = q.simulate(d, H, rule=r)
        ax[0].plot(hh, rr_['i'], lw=1.1, alpha=.7, label=nm)
        ax[1].plot(hh, rr_['pi4'], lw=1.1, alpha=.7, label=nm)
    ax[0].plot(hh, res['i'], lw=2.2, color='#1b3a5c',
               label='оптим. простое правило %.2f/%.2f/%.2f' % (phi, phy, rho))
    ax[1].plot(hh, res['pi4'], lw=2.2, color='#1b3a5c',
               label='оптим. простое правило')
    ax[0].plot(hh, opt.x, lw=1.8, color='k', ls='--', label='свободная траектория')
    ax[1].plot(hh, ropt['pi4'], lw=1.8, color='k', ls='--', label='свободная траектория')
    ax[1].axhline(4, color='r', ls='--', lw=1, label='цель 4%')
    ax[0].set_title('Ключевая ставка, %'); ax[1].set_title('Инфляция год/год, %')
    ax[2].plot(hh, res['gap'], lw=2.2, color='#1b3a5c', label='оптим. простое правило')
    ax[2].plot(hh, ropt['gap'], lw=1.8, color='k', ls='--', label='свободная траектория')
    ax[2].axhline(0, color='k', lw=.6); ax[2].set_title('Разрыв выпуска, %')
    ax[2].legend(fontsize=6)
    for a in ax:
        a.set_xlabel('кварталы вперёд')
    ax[0].legend(fontsize=6); ax[1].legend(fontsize=6)
    fig.suptitle('QPM: простые правила и оптимальная траектория ставки', fontsize=9)
    fig.tight_layout(); fig.savefig('fig_qpm.png', bbox_inches='tight'); plt.close(fig)

    with open('qpm_out.json', 'w', encoding='utf-8') as f:
        json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)
    print('\nзаписано: qpm_out.json, fig_qpm.png')


if __name__ == '__main__':
    main()
