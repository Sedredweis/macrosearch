"""
SVAR/BVAR малой открытой экономики (IS-LM-BP + IS-MR-PC).
Приор Миннесоты с теоретическими окнами лагов, блочно-рекурсивная
идентификация + знаковая ротация ценово-финансового блока,
сценарный условный прогноз по траектории ключевой ставки.

Зависимости: numpy, pandas.  Использование: см. блок __main__ внизу.
"""
import numpy as np, pandas as pd

# ------------------------------------------------------------------ данные
RAW = ['zzobs_r_G','zzobs_dY','zzobs_dPC','zzobs_dNFX','zzobs_dRFX','zzobs_SP_PY',
       'zzobs_dInc','zzobs_dC','zzobs_dG','zzobs_dI','zzobs_dIM','zzobs_dEX','zzobs_dL']

# порядок блоков: внешний | фискальный | реальный | цены | финансы
V = ['pistar','dG','SP','dInc','dI','dL','dC','dEX','dIM','dPC','rG','dNFX']
# Размер ротируемого блока при знаковой идентификации.
# Было NB=3 (ротация только внутри ценово-финансового блока dPC,rG,dNFX поверх
# блочно-рекурсивных нулей). На данных 189Q3-239Q2 эта схема НЕ РАБОТАЕТ:
# при апостериорном среднем ни одна из 20 000 ротаций 3-мерного блока не даёт
# дезинфляции после ужесточения (0 принятых), а точечная схема Холецкого даёт
# ценовую загадку — инфляция растёт на всех горизонтах h=0..20. Загадка лежит
# в данных, а не в приоре: она воспроизводится и при диффузном приоре (lam=1e6).
# Поэтому ротация расширена на ВСЁ пространство (12 переменных) — чистая
# знаковая идентификация в духе Uhlig (2005), без нулевых ограничений A0.
# Подробности и точный критерий выбора точечного шока — в identify() и
# mp_max_share() ниже.

W_Y = {'dC':0.545891,'dG':0.169220,'dI':0.201795,'dEX':0.193976,'dIM':-0.110875}
LEVELS = ['rG','SP']                     # ряды в уровнях -> приорное среднее 1

def load(path):
    """CSV (';', две строки заголовка) -> DataFrame с переменными модели."""
    d = pd.read_csv(path, sep=',', skiprows=[0])
    d['period'] = d['period'].astype(str).str.replace("'", '').str.strip()
    d = d.set_index('period')[RAW].astype(float)
    X = pd.DataFrame(index=d.index)
    # тождество q = e + p* - p  ->  имплицитная внешняя инфляция (внешний блок)
    X['pistar'] = d.zzobs_dPC + d.zzobs_dRFX - d.zzobs_dNFX
    X['dG'], X['SP'], X['dInc'] = d.zzobs_dG, d.zzobs_SP_PY, d.zzobs_dInc
    X['dI'], X['dL'], X['dC'] = d.zzobs_dI, d.zzobs_dL, d.zzobs_dC
    X['dEX'], X['dIM'] = d.zzobs_dEX, d.zzobs_dIM
    X['dPC'], X['rG'], X['dNFX'] = d.zzobs_dPC, d.zzobs_r_G, d.zzobs_dNFX
    X['dY_obs'] = d.zzobs_dY                        # только для проверки
    return X

WVEC = np.array([W_Y.get(v, 0.0) for v in V])

# Порядок VAR по умолчанию. На новых данных 189Q3-239Q2 выбран p = 6:
# это единственный порядок из сетки {2, 4, 6}, при котором тест Люнга-Бокса
# не отвергает отсутствие автокорреляции остатков НИ В ОДНОМ уравнении
# (при p = 4 остаётся автокорреляция в уравнении экспорта, LB(8) = 22.2 при
# критическом значении 15.5; при p = 6 — LB(8) = 15.2). Внепрогнозная ошибка
# при этом практически не меняется (RMSE годовой инфляции 0.0245 против
# 0.0244 при p = 4), а трансмиссия ДКП оценивается сильнее и правдоподобнее.
P_DEFAULT = 6

# ------------------------------------------- пересчёт квартал <-> год (цены)
# Данные — логарифмические приросты: dPC_t = ln(P_t / P_{t-1}), см. заголовок
# CSV «кв/кв ln(x/100)». Валовой квартальный множитель = exp(dPC_t), значит
#   год из квартала : (1 + pi_q)^4 - 1   <=>  exp(4*dPC) - 1
#   квартал из года : (1 + pi_a)^(1/4) - 1  <=>  ln(1 + pi_a)/4   (корень 4-й ст.)
# Линейные x400 (квартал->год) и /400 (год->квартал) — приближение первого
# порядка, верное только при малых приростах: при 4% год. ошибка ~0.08 п.п.,
# при 15% год. — уже ~0.8 п.п., и она систематически завышает годовую цифру.
ANN = 400.0    # линейный множитель. Оставлен ТОЛЬКО там, где компаундирование
               # не определено: отклики (IRF, отклонения от базы), с.к.о., RMSE.


def ann_pct(q):
    """Квартальный лог-прирост цен -> годовая ставка, %: (exp(4q) - 1) * 100."""
    return (np.exp(4.0 * np.asarray(q, float)) - 1.0) * 100.0


def yoy_pct(q4, axis=0):
    """Четыре квартальных лог-прироста -> инфляция год/год, %.
    Логарифмы складываются (= множители перемножаются), затем exp(.) - 1."""
    return (np.exp(np.sum(np.asarray(q4, float), axis=axis)) - 1.0) * 100.0


def q_from_ann(pi_ann_pct):
    """Годовая инфляция, % -> квартальный лог-прирост: ln(1 + pi/100) / 4.
    Это и есть «корень 4-й степени», а не деление на 4 (на 400 в процентах)."""
    return np.log1p(np.asarray(pi_ann_pct, float) / 100.0) / 4.0


def agg_dY(a):
    """Прирост выпуска по тождеству СНС. Ось 1 = переменные:
    (H,n)->(H,) для прогноза, (h,n,n)->(h,n_shocks) для IRF."""
    return np.tensordot(np.asarray(a), WVEC, axes=([1], [0]))

# --------------------------------------------- теоретические окна лагов
# (уравнение, регрессор): (lmin, lmax); вне окна приорная дисперсия * TH
# Окна канала ставки (i -> C, I, pi, L, EX, IM) в Части I задавались как 2-6 кв.
# На данных это оказалось СЛИШКОМ МЕДЛЕННО: пик отклика инфляции приходится на
# h=2, и окно 3-6 сжимало истинный эффект вдвое. Скорректировано до 1-6
# (при p=4 равносильно отсутствию ограничения) — см. отчёт, разд. 5.
WIN = {('dC','rG'):(1,6), ('dI','rG'):(1,6), ('dPC','rG'):(1,6), ('dL','rG'):(1,6),
       ('dEX','rG'):(1,6), ('dIM','rG'):(1,6),
       ('dEX','dNFX'):(2,6),        # J-кривая: эффект курса на экспорт не раньше 2 кв.
       ('dPC','dNFX'):(1,3),        # эффект переноса затухает к 4-му кварталу
       ('rG','rG'):(1,2),           # сглаживание ставки
       ('rG','dPC'):(1,4),          # правило на годовую инфляцию
       ('dL','dC'):(1,2), ('dL','dI'):(1,2),      # закон Оукена
       ('dG','dG'):(1,2), ('dG','SP'):(1,2)}      # инерция бюджетного процесса
TH = 0.03          # "мягкий ноль": во сколько раз ужимается приор вне окна
BLOCK = {'pistar':0,'dG':1,'SP':1,'dInc':1,'dI':2,'dL':2,'dC':2,'dEX':2,'dIM':2,
         'dPC':3,'rG':4,'dNFX':4}


class BVAR:
    """VAR(p) с приором Миннесоты; оценка поуравненчно (Theil mixed estimation)."""

    def __init__(self, Y, p=4, lam=1.0, decay=2.0, cross=0.5, mu=0.0):
        self.Y, self.p, self.n = np.asarray(Y, float), p, Y.shape[1]
        self.lam, self.decay, self.cross, self.mu = lam, decay, cross, mu
        self._fit()

    # -------- вспомогательное: матрица регрессоров
    def _lagmat(self, Y):
        T, n, p = len(Y), self.n, self.p
        Z = [Y[p - l:T - l] for l in range(1, p + 1)]
        return np.column_stack(Z + [np.ones(T - p)])          # (T-p, n*p+1)

    def _prior(self):
        """Приорное среднее b (n*p+1, n) и с.к.о. sd той же формы."""
        n, p = self.n, self.p
        s = np.array([np.std(self.Y[:, j]) for j in range(n)])
        # масштаб через остатки AR(p) — устойчивее чистого std
        for j in range(n):
            Zj = self._lagmat(self.Y[:, [j]])
            bj, *_ = np.linalg.lstsq(Zj, self.Y[p:, j], rcond=None)
            s[j] = np.std(self.Y[p:, j] - Zj @ bj)
        b = np.zeros((n * p + 1, n)); sd = np.zeros((n * p + 1, n))
        for i in range(n):                                    # уравнение
            for l in range(1, p + 1):
                for j in range(n):                            # регрессор
                    r = (l - 1) * n + j
                    if i == j and l == 1 and V[i] in LEVELS:
                        b[r, i] = 1.0
                    v = (self.lam / l ** self.decay) * (s[i] / s[j])
                    if i != j:
                        v *= self.cross if BLOCK[V[i]] != BLOCK[V[j]] else 1.0
                    lo, hi = WIN.get((V[i], V[j]), (1, p))
                    if not (lo <= l <= hi):
                        v *= np.sqrt(TH)
                    if V[i] == 'pistar' and i != j:
                        v = 1e-8                              # блочная экзогенность
                    sd[r, i] = v
            sd[-1, i] = 100 * s[i]                            # константа — диффузно
        self.s = s
        return b, sd

    def _fit(self):
        Y, p, n = self.Y, self.p, self.n
        Z, y = self._lagmat(Y), Y[self.p:]
        b0, sd = self._prior()
        # sum-of-coefficients: штраф на сумму лаговых коэффициентов (против ложного тренда)
        ybar = Y[:p].mean(0)
        Yd = np.diag(ybar) * self.mu
        Xd = np.column_stack([Yd for _ in range(p)] + [np.zeros((n, 1))])
        Zf, Yf = np.vstack([Z, Xd]), np.vstack([y, Yd])
        B = np.zeros((n * p + 1, n)); Vp = []
        for i in range(n):
            Pi = np.diag(1.0 / sd[:, i] ** 2)
            A = Zf.T @ Zf / self.s[i] ** 2 + Pi
            rhs = Zf.T @ Yf[:, i] / self.s[i] ** 2 + Pi @ b0[:, i]
            Vi = np.linalg.inv(A); B[:, i] = Vi @ rhs; Vp.append(Vi)
        self.B, self.Vpost = B, Vp
        self.U = y - Z @ B
        self.Sigma = self.U.T @ self.U / (len(y) - (n * p + 1))
        self.Z, self.y = Z, y

    # -------- динамика
    def companion(self, B=None):
        B = self.B if B is None else B
        n, p = self.n, self.p
        A = np.zeros((n * p, n * p))
        A[:n] = B[:n * p].T
        if p > 1:
            A[n:, :-n] = np.eye(n * (p - 1))
        return A

    def draw_B(self, rng):
        """Драв из (нормального) апостериорного распределения коэффициентов."""
        B = np.empty_like(self.B)
        for i in range(self.n):
            L = np.linalg.cholesky(self.Vpost[i] + 1e-14 * np.eye(len(self.B)))
            B[:, i] = self.B[:, i] + L @ rng.standard_normal(len(self.B))
        return B

    def max_root(self):
        return np.abs(np.linalg.eigvals(self.companion())).max()

    def irf(self, h, S=None, B=None):
        """(h+1, n, n): отклик переменных на единичные структурные шоки."""
        S = np.linalg.cholesky(self.Sigma) if S is None else S
        A, n, p = self.companion(B), self.n, self.p
        C = np.zeros((n * p, n)); C[:n] = S
        out = np.zeros((h + 1, n, n)); M = np.eye(n * p)
        for k in range(h + 1):
            out[k] = (M @ C)[:n]; M = A @ M
        return out

    def irf1(self, h, s, B=None):
        """(h+1, n): отклик на ОДИН структурный шок с импакт-вектором s.
        Нужен для знаковой идентификации, где однозначно определён только
        столбец шока ДКП, а не вся матрица S."""
        A, n, p = self.companion(B), self.n, self.p
        v = np.zeros(n * p); v[:n] = np.asarray(s, float)
        out = np.zeros((h + 1, n)); M = np.eye(n * p)
        for k in range(h + 1):
            out[k] = (M @ v)[:n]; M = A @ M
        return out

    def forecast(self, H, Y0=None, B=None):
        """Безусловный прогноз (шоки = 0)."""
        B = self.B if B is None else B
        Y = (self.Y if Y0 is None else Y0)[-self.p:].copy()
        out = []
        for _ in range(H):
            z = np.concatenate([Y[-l] for l in range(1, self.p + 1)] + [[1.0]])
            nx = z @ B; out.append(nx); Y = np.vstack([Y, nx])
        return np.array(out)

    # -------- сценарий по ставке
    def scenario(self, rate_path, s_mp, Y0=None, B=None):
        """Условный прогноз при заданной траектории ставки, реализуемой
        ТОЛЬКО шоками ДКП (Waggoner-Zha с одним инструментом).

        s_mp — импакт-вектор шока ДКП (столбец S при рекурсивной схеме или
        идентифицированный знаковыми ограничениями вектор)."""
        rate_path = np.asarray(rate_path, float); H = len(rate_path)
        base = self.forecast(H, Y0, B); ir = self.irf1(H, s_mp, B)
        kr = V.index('rG')
        M = np.zeros((H, H))
        for t in range(H):
            for q in range(t + 1):
                M[t, q] = ir[t - q, kr]
        eps = np.linalg.solve(M, rate_path - base[:, kr])
        add = np.zeros((H, self.n))
        for t in range(H):
            for q in range(t + 1):
                add[t] += ir[t - q, :] * eps[q]
        return base + add, eps

    # -------- контрфактическое правило ДКП
    def counterfactual_rule(self, H, s_mp, phi_pi, phi_y=0.0, phi_e=0.0,
                            rho=0.8, pi_target=4.0, i_neutral=None,
                            Y0=None, B=None, hist=None):
        """Смена монетарного режима: историческое правило заменяется явным
        правилом таргетирования инфляции, остальные уравнения системы
        оставляются как оценены (Bernanke-Gertler-Watson 1997, 2004;
        Sims-Zha 2006).

        Новое правило (в процентах годовых):
            i_t = rho*i_{t-1} + (1-rho)*[ i_neutral
                                          + phi_pi*(pi4_t - pi_target)
                                          + phi_y*dy4_t
                                          + phi_e*100*dNFX_t ]
        где pi4_t — инфляция год/год ВКЛЮЧАЯ текущий квартал, dy4_t — прирост
        выпуска за 4 квартала. Отклонение фактической (модельной) ставки от
        предписанной правилом реализуется шоками ДКП eps_t с импакт-вектором
        s_mp: в каждом квартале решается скалярное уравнение
            i_model_t + s_mp[rG]*eps_t  =  i_rule_t(состояние + s_mp*eps_t),
        нелинейное только из-за компаундирования в pi4 (решается Ньютоном).

        ВАЖНО, критика Лукаса. Коэффициенты непопитических уравнений оценены
        ПРИ СТАРОМ режиме и при смене правила, вообще говоря, меняются. Такой
        счёт правомерен ровно настолько, насколько требуемые eps_t укладываются
        в разброс исторически наблюдавшихся шоков ДКП (McKay-Wolf, 2023):
        возвращаемый eps в сигмах — это и есть мера того, насколько сильную
        смену режима вы просите у модели. При max|eps| заметно больше 3 сигм
        результат следует считать иллюстрацией, а не прогнозом.

        Возвращает (path (H,n), eps (H,) в единицах шока, i_rule (H,)).
        """
        B = self.B if B is None else B
        kr, kp = V.index('rG'), V.index('dPC')
        s_mp = np.asarray(s_mp, float)
        hist = (self.Y if Y0 is None else Y0) if hist is None else hist
        if i_neutral is None:
            # нейтральная номинальная ставка = средняя реальная за выборку + цель
            i_neutral = float(hist[:, kr].mean() - ann_pct(hist[:, kp].mean())
                              + pi_target)
        Yl = hist[-self.p:].copy()
        pi_hist = list(hist[-3:, kp])          # три последних кв. лог-прироста цен
        dy_hist = list(agg_dY(hist[-3:]))
        i_prev = float(hist[-1, kr])
        out, eps_out, rule_out = [], [], []
        for _ in range(H):
            z = np.concatenate([Yl[-l] for l in range(1, self.p + 1)] + [[1.0]])
            nx0 = z @ B                                   # модельный шаг без шока ДКП

            def gap(e):
                nx = nx0 + s_mp * e
                pi4 = yoy_pct(pi_hist[-3:] + [nx[kp]])
                dy4 = 100.0 * (sum(dy_hist[-3:]) + float(nx @ WVEC))
                i_rule = (rho * i_prev + (1 - rho) *
                          (i_neutral + phi_pi * (pi4 - pi_target)
                           + phi_y * dy4 + phi_e * 100.0 * nx[V.index('dNFX')]))
                return nx[kr] + 0.0 - i_rule, i_rule

            # правило линейно по e всюду, кроме exp(.) в pi4 -> два-три шага Ньютона
            e = 0.0
            for _ in range(60):
                g0, _r = gap(e)
                g1, _ = gap(e + 1e-6)
                d = (g1 - g0) / 1e-6
                if abs(d) < 1e-12:
                    break
                step = g0 / d
                e -= step
                if abs(step) < 1e-12:
                    break
            g, i_rule = gap(e)
            nx = nx0 + s_mp * e
            out.append(nx); eps_out.append(e); rule_out.append(i_rule)
            pi_hist.append(nx[kp]); dy_hist.append(float(nx @ WVEC))
            i_prev = float(nx[kr]); Yl = np.vstack([Yl, nx])
        return np.array(out), np.array(eps_out), np.array(rule_out)

    def rule_loop(self, s_mp, phi_pi, phi_y=0.0, phi_e=0.0, rho=0.8, B=None):
        """Замкнутая система «оценённый непопитический блок + новое правило».

        Правило линеаризуется (pi4 ~ 100*сумма четырёх квартальных лог-приростов,
        что верно с точностью до компаундирования) и подставляется в систему:
        шок ДКП становится ЛИНЕЙНОЙ функцией состояния, eps_t = a'x_{t-1} + b,
        а приведённая форма — Pi_cl = Pi + s_mp * a'.

        Возвращает (A_cl, a, b, max_root). max_root >= 1 означает, что при таком
        правиле система взрывается: с оценённой (на старом режиме) динамикой
        такое правило нереализуемо. Требует p >= 4, чтобы в состоянии были все
        четыре лага, входящие в годовую инфляцию.
        """
        if self.p < 4:
            raise ValueError('нужно p >= 4: правило смотрит на инфляцию год/год')
        B = self.B if B is None else B
        n, p = self.n, self.p
        kr, kp, kx = V.index('rG'), V.index('dPC'), V.index('dNFX')
        Ay = B[:n * p].T                                  # (n, n*p)
        c0 = B[-1].copy()
        s = np.asarray(s_mp, float)
        g = np.zeros(n); g[kr] = 1.0                      # отбор ставки
        k = (1 - rho) * (phi_pi * 100 * np.eye(n)[kp]
                         + phi_y * 100 * WVEC
                         + phi_e * 100 * np.eye(n)[kx])
        # строки состояния: y_{t-l} занимает позиции [(l-1)n : l*n]
        def sel(l, vec):
            r = np.zeros(n * p); r[(l - 1) * n:l * n] = vec; return r
        g1 = sel(1, np.eye(n)[kr])                        # i_{t-1}
        Lp = sum(sel(l, np.eye(n)[kp]) for l in (1, 2, 3))
        Lw = sum(sel(l, WVEC) for l in (1, 2, 3))
        den = float((g - k) @ s)
        if abs(den) < 1e-12:
            raise RuntimeError('правило не реализуемо шоком ДКП: (g-k)@s = 0')
        a = (rho * g1 + (1 - rho) * 100 * (phi_pi * Lp + phi_y * Lw)
             - (g - k) @ Ay) / den
        b = float((1 - rho) * (0.0) - (g - k) @ c0) / den    # константа правила
        Pi_cl = Ay + np.outer(s, a)
        A = np.zeros((n * p, n * p)); A[:n] = Pi_cl
        if p > 1:
            A[n:, :-n] = np.eye(n * (p - 1))
        return A, a, b, float(np.abs(np.linalg.eigvals(A)).max())


# ------------------------------------------------- идентификация
# Знаковые ограничения на шок УЖЕСТОЧЕНИЯ ДКП (в духе Uhlig, 2005).
# Импакт цен не ограничивается: цены в квартале шока липкие, знак на h=0 —
# это предпосылка, а не вывод. Реакция курса НЕ ограничивается вовсе.
SIGN_R  = (0, 1, 2)             # ставка растёт
SIGN_PI = tuple(range(1, 7))    # dPC <= 0 на h = 1..6 (дезинфляция)
SIGN_DY = tuple(range(0, 5))    # dY  <= 0 на h = 0..4 (спад спроса)


def _mult(m, B=None, h=20):
    """(h+1, n, n): мультипликаторы приведённой формы, ir(h) = _mult[h] @ s."""
    A, n, p = m.companion(B), m.n, m.p
    out = np.zeros((h + 1, n, n)); M = np.eye(n * p)
    for k in range(h + 1):
        out[k] = M[:n, :n]; M = A @ M
    return out


def _sign_ok(ir):
    """Проверка знаковых ограничений по IRF формы (h+1, n)."""
    kP, kR = V.index('dPC'), V.index('rG')
    dy = ir @ WVEC
    return (all(ir[h, kR] > 0 for h in SIGN_R)
            and all(ir[h, kP] <= 0 for h in SIGN_PI)
            and all(dy[h] <= 0 for h in SIGN_DY))


def mp_share(m, s):
    """Доля шока в мгновенной дисперсии ставки: s[rG]^2 / Sigma[rG,rG]."""
    kR = V.index('rG')
    return float(s[kR] ** 2 / m.Sigma[kR, kR])


def mp_max_share(m, B=None, seed=0, nstart=40, h=20):
    """Точечная идентификация шока ДКП: среди всех шоков, удовлетворяющих
    знаковым ограничениям, берётся объясняющий МАКСИМАЛЬНУЮ долю мгновенной
    дисперсии ключевой ставки (max-share; Uhlig 2003, Barsky-Sims 2011).

    Это снимает произвол «какой элемент допустимого множества показывать»
    (проблема Fry-Pagan): выбор однозначен и имеет экономический смысл —
    шок правила ЦБ должен двигать саму ставку сильнее любого другого.

    Возвращает (s, share). Требует scipy.optimize.
    """
    from scipy.optimize import minimize
    P = np.linalg.cholesky(m.Sigma)
    Mp = _mult(m, B, h)
    kR, n = V.index('rG'), m.n
    var_r = m.Sigma[kR, kR]

    def unit(q):
        s = P @ (q / np.linalg.norm(q))
        return -s if s[kR] < 0 else s

    def cons(q):
        ir = Mp @ unit(q); dy = ir @ WVEC
        return np.array([ir[h_, kR] for h_ in SIGN_R]
                        + [-ir[h_, V.index('dPC')] for h_ in SIGN_PI]
                        + [-dy[h_] for h_ in SIGN_DY])

    rng = np.random.default_rng(seed); best = None
    for _ in range(nstart):
        q0 = rng.standard_normal(n); q0 /= np.linalg.norm(q0)
        r = minimize(lambda q: -unit(q)[kR] ** 2 / var_r, q0, method='SLSQP',
                     constraints=[{'type': 'ineq', 'fun': cons}],
                     options={'maxiter': 500, 'ftol': 1e-12})
        if (cons(r.x) >= -1e-9).all():
            s = unit(r.x); sh = s[kR] ** 2 / var_r
            if best is None or sh > best[1]:
                best = (s, float(sh))
    if best is None:
        raise RuntimeError('знаковые ограничения несовместимы с данными')
    return best


def identify(m, ndraw=2000, nB=150, seed=0, h=20, draw_B=True, rdom=0.0):
    """Апостериорное множество допустимых шоков ДКП: совместная неопределённость
    оценки (драв B из апостериорного распределения) и идентификации (случайная
    ортонормированная ротация во ВСЁМ 12-мерном пространстве).

    Почему ротация полная (NB = 12, было 3). На данных 189Q3-239Q2 ротация
    только внутри ценово-финансового блока (dPC, rG, dNFX) поверх блочно-
    рекурсивных нулей даёт 0 принятых из 20 000 при апостериорном среднем:
    нули A0 несовместимы со знаковыми ограничениями. Точечная схема Холецкого
    на этих данных даёт ценовую загадку (инфляция растёт на всех h = 0..20),
    причём загадка лежит в данных, а не в приоре — воспроизводится и при
    диффузном приоре. Расширение ротации на все переменные снимает конфликт.

    Исправлена ошибка прежней версии: ограничения проверялись на СЛУЧАЙНОМ
    драве коэффициентов B, а IRF затем считались на апостериорном среднем, —
    возвращаемое множество не удовлетворяло собственным ограничениям. Теперь
    каждый элемент — пара (s, B), и все IRF считаются с тем же B.

    rdom — минимальная доля шока в мгновенной дисперсии ставки (0 = без порога).
    Возвращает список словарей {'s', 'B', 'share'}.
    """
    rng = np.random.default_rng(seed)
    P = np.linalg.cholesky(m.Sigma)
    kR, n = V.index('rG'), m.n
    var_r = m.Sigma[kR, kR]
    out = []
    for _ in range(nB):
        B = m.draw_B(rng) if draw_B else m.B
        Mp = _mult(m, B, h)
        for _ in range(ndraw):
            q = rng.standard_normal(n); q /= np.linalg.norm(q)
            s = P @ q
            if s[kR] < 0:
                s = -s
            if s[kR] ** 2 < rdom * var_r:
                continue
            if _sign_ok(Mp @ s):
                out.append({'s': s, 'B': B, 'share': float(s[kR] ** 2 / var_r)})
    if not out:
        raise RuntimeError('ни одна ротация не прошла знаковые ограничения')
    return out


def mp_irf_set(m, Sset, h=20):
    """(ndraw, h+1, n): IRF на шок ДКП по всему принятому множеству,
    каждая со СВОИМ B (см. identify)."""
    return np.array([m.irf1(h, d['s'], d['B']) for d in Sset])


# ------------------------------------------------- валидация
def oos_rmse(Y, p, lam, H=8, start=140, **kw):
    """Псевдо-внепрогнозная RMSE (расширяющееся окно) для dPC и rG."""
    kP, kR = V.index('dPC'), V.index('rG')
    e = {k: [] for k in ('pi1','pi4','r1','r4')}
    for t in range(start, len(Y) - H):
        m = BVAR(Y[:t], p=p, lam=lam, **kw); f = m.forecast(H)
        e['pi1'].append(f[0,kP] - Y[t,kP]); e['r1'].append(f[0,kR] - Y[t,kR])
        e['pi4'].append(f[:4,kP].sum() - Y[t:t+4,kP].sum())
        e['r4'].append(f[3,kR] - Y[t+3,kR])
    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in e.items()}


def ljung_box(u, lags=8):
    T = len(u); r = []
    for l in range(1, lags + 1):
        c = np.corrcoef(u[l:], u[:-l])[0, 1]; r.append(c * c / (T - l))
    return T * (T + 2) * np.sum(r)


if __name__ == '__main__':
    import sys, numpy.linalg as la
    X = load(sys.argv[1] if len(sys.argv) > 1 else 'data_239Q2.csv')
    Y = X[V].values
    m = BVAR(Y, p=P_DEFAULT)
    kR, kP = V.index('rG'), V.index('dPC')
    print('данные: %s .. %s (%d кв.)' % (X.index[0], X.index[-1], len(X)))
    print('макс. модуль корня      : %.3f' % m.max_root())
    print('с.к.о. инновации ставки : %.3f п.п.' % np.sqrt(m.Sigma[kR, kR]))

    # --- рекурсивная схема (СПРАВОЧНО: на этих данных даёт ценовую загадку)
    Sc = la.cholesky(m.Sigma)
    irc = m.irf1(12, Sc[:, kR])
    print('\n[рекурсивная схема Холецкого — СПРАВОЧНО]')
    print('sigma шока ДКП, п.п.    : %.3f (доля в дисп. ставки %.2f)'
          % (Sc[kR, kR], Sc[kR, kR] ** 2 / m.Sigma[kR, kR]))
    print('IRF ставки,   h=0..8    :', np.round(irc[:9, kR], 3))
    print('IRF инфляции, h=0..8, %г:', np.round(ANN * irc[:9, kP], 3))
    if (irc[1:9, kP] > 0).all():
        print('  ВНИМАНИЕ: ценовая загадка — инфляция растёт после ужесточения.')

    # --- ОСНОВНАЯ СХЕМА: знаковые ограничения + max-share
    s, share = mp_max_share(m, seed=1)
    ir = m.irf1(20, s)
    print('\n[знаковые ограничения + max-share — ОСНОВНАЯ СХЕМА]')
    print('sigma шока ДКП, п.п.    : %.3f' % s[kR])
    print('доля в дисперсии ставки : %.3f' % share)
    # IRF — отклонение от базы, не уровень: компаундирование не определено,
    # аннуализация остаётся линейной (ANN), см. комментарий у ann_pct().
    print('IRF ставки,   h=0..8    :', np.round(ir[:9, kR], 3))
    print('IRF инфляции, h=0..8, %г:', np.round(ANN * ir[:9, kP], 3))
    print('IRF выпуска,  h=0..8, % :', np.round(100 * (ir @ WVEC)[:9], 3))
    print('накопл. эффект на уровень цен, h=1..20, %%: %.3f' % (100 * ir[1:, kP].sum()))

    r0 = float(Y[-1, kR])
    for lvl in (r0 - 3.0, r0, r0 + 3.0):
        path, eps = m.scenario(np.full(12, lvl), s)
        yr = [yoy_pct(path[i:i + 4, kP]) for i in (0, 4, 8)]
        print('ставка %5.2f%%: годовая инфляция %s ; max|eps| = %.1f sigma'
              % (lvl, np.round(yr, 2), np.abs(eps).max() / s[kR]))

    Sset = identify(m, ndraw=1500, nB=60, seed=7)
    print('\nпринятых (s, B) из %d: %d' % (1500 * 60, len(Sset)))
    IR = mp_irf_set(m, Sset)
    print('68%% полоса инфляции, h=4, %%г: [%.3f, %.3f]'
          % (ANN * np.percentile(IR[:, 4, kP], 16), ANN * np.percentile(IR[:, 4, kP], 84)))
