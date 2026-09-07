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
NB = 3                                   # размер ротируемого блока (dPC,rG,dNFX)

W_Y = {'dC':0.387377,'dG':0.339971,'dI':0.428197,'dEX':0.240049,'dIM':-0.395594}
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
    def scenario(self, rate_path, S, k_mp, Y0=None):
        """Условный прогноз при заданной траектории ставки, реализуемой
        ТОЛЬКО шоками ДКП (Waggoner-Zha с одним инструментом)."""
        rate_path = np.asarray(rate_path, float); H = len(rate_path)
        base = self.forecast(H, Y0); ir = self.irf(H, S)
        kr = V.index('rG')
        M = np.zeros((H, H))
        for t in range(H):
            for q in range(t + 1):
                M[t, q] = ir[t - q, kr, k_mp]
        eps = np.linalg.solve(M, rate_path - base[:, kr])
        add = np.zeros((H, self.n))
        for t in range(H):
            for q in range(t + 1):
                add[t] += ir[t - q, :, k_mp] * eps[q]
        return base + add, eps


# ------------------------------------------------- идентификация
def _complete(q):
    """Дополнить единичный вектор q до ортонормированного базиса R^NB
    (первый столбец результата равен q)."""
    Q, _ = np.linalg.qr(np.column_stack([q, np.eye(NB)]))
    return Q * np.sign(Q[:, 0] @ q)


def identify(m, h=12, ndraw=4000, seed=0, draw_B=True, rdom=0.5):
    """Частичная знаковая идентификация шока ДКП (в духе Uhlig, 2005)
    внутри ценово-финансового блока (dPC, rG, dNFX); остальные 9 шоков
    остаются блочно-рекурсивными (нули A0 из раздела 3.3).

    Ограничения на шок ужесточения ДКП:
        rG > 0            h = 0,1,2
        sum dPC(h=2..6) < 0     (дезинфляция с лагом 3-6 кварталов)
        sum dY(h=1..4)  < 0     (спад спроса)
        доля шока в мгновенной дисперсии rG внутри блока > rdom
                                (шок должен быть именно шоком правила ЦБ)
    Реакция курса НЕ ограничивается — она результат, а не предпосылка.

    Возвращает (список матриц S, индексы шоков).  Второй и третий шоки
    блока получаются как ортогональное дополнение и упорядочиваются так,
    что 'fx' — тот, что сильнее двигает курс на импакте.
    """
    rng = np.random.default_rng(seed)
    n, i0 = m.n, m.n - NB
    kP, kR, kX = V.index('dPC'), V.index('rG'), V.index('dNFX')
    out, B0 = [], m.B
    for _ in range(ndraw):
        B = m.draw_B(rng) if draw_B else B0
        try:
            P = np.linalg.cholesky(m.Sigma)
        except np.linalg.LinAlgError:
            continue
        q = rng.standard_normal(NB); q /= np.linalg.norm(q)
        for sgn in (1.0, -1.0):
            Q3 = _complete(sgn * q)
            Q = np.eye(n); Q[i0:, i0:] = Q3
            S = P @ Q
            ir = m.irf(h, S, B); dy = agg_dY(ir)
            mp = i0                                   # первый столбец блока = ДКП
            if not (all(ir[k, kR, mp] > 0 for k in (0, 1, 2))
                    and ir[2:7, kP, mp].sum() < 0 and dy[1:5, mp].sum() < 0
                    and ir[0, kR, mp] ** 2 > rdom * np.sum(ir[0, kR, i0:] ** 2)):
                continue
            # упорядочить оставшиеся два: 'fx' сильнее двигает курс
            a_, b_ = i0 + 1, i0 + 2
            if abs(ir[0, kX, a_]) > abs(ir[0, kX, b_]):
                S[:, [a_, b_]] = S[:, [b_, a_]]
            for c in (i0 + 1, i0 + 2):                # нормировка: dPC растёт
                if S[kP, c] < 0:
                    S[:, c] *= -1
            out.append(S)
            break
    if not out:
        raise RuntimeError('ни одна ротация не прошла знаковые ограничения')
    return out, {'cost_push': i0 + 1, 'mp': i0, 'fx': i0 + 2}


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
    m = BVAR(Y, p=4)
    S = la.cholesky(m.Sigma)                     # точечная блочно-рекурсивная схема
    kR = V.index('rG')                           # шок ДКП = столбец ставки
    ir = m.irf(12, S)
    print('макс. модуль корня      : %.3f' % m.max_root())
    print('sigma шока ДКП, п.п.    : %.3f' % S[kR, kR])
    print('IRF ставки,   h=0..8    :', np.round(ir[:9, kR, kR], 3))
    print('IRF инфляции, h=0..8, %г:', np.round(400 * ir[:9, V.index('dPC'), kR], 3))
    print('IRF выпуска,  h=0..8, % :', np.round(100 * agg_dY(ir)[:9, kR], 3))
    for lvl in (4.0, 5.0, 7.0):
        path, eps = m.scenario(np.full(12, lvl), S, kR)
        yr = [100 * path[i:i+4, V.index('dPC')].sum() for i in (0, 4, 8)]
        print('ставка %4.1f%%: годовая инфляция %s ; max|eps| = %.1f sigma'
              % (lvl, np.round(yr, 2), np.abs(eps).max() / S[kR, kR]))
    Sset, k = identify(m, ndraw=3000)
    print('знаково-идентифицированных ротаций: %d' % len(Sset))
