"""
SVAR_MODEL_FINAL.py — рабочая оболочка над моделью `svar_soe.py`.

Отличие от run_analysis.py: тот считает всю валидацию и рисует картинки,
этот отвечает на три конкретных вопроса из командной строки.

    1) реакция всех переменных на ОДНО заданное значение ставки
       (ставка удерживается на этом уровне H кварталов)
    2) реакция всех переменных на ЗАДАННУЮ ТРАЕКТОРИЮ ставки
       (значения в хронологическом порядке)
    3) печать всех уравнений SVAR с численными коэффициентами
       (матрица мгновенных связей A0, лаговые матрицы A1..Ap, уравнения)

Запуск:
    python SVAR_MODEL_FINAL.py                     # интерактивное меню
    python SVAR_MODEL_FINAL.py --rate 7            # ставка 7% на 12 кварталов
    python SVAR_MODEL_FINAL.py --rates 5,5.5,6,6.5 # траектория (4 квартала)
    python SVAR_MODEL_FINAL.py --equations         # все уравнения
    python SVAR_MODEL_FINAL.py --irf               # отклик на шок ДКП в 1 с.к.о.

Опции:
    --H 12                число кварталов (для --rate; по умолчанию 12)
    --data data_train.csv  путь к данным
    --lags 4               порядок VAR
    --lam 1.0              жёсткость приора Миннесоты
    --csv out.csv          дополнительно сохранить таблицу отклика в CSV
    --full                 в --equations печатать также лаговые матрицы целиком

Зависимости: numpy, pandas, svar_soe.py и data_train.csv в той же папке.
"""
import sys, textwrap
import numpy as np
import numpy.linalg as la
import pandas as pd
import svar_soe as M

V = M.V

# подписи и масштабы вывода: (название, множитель, единица)
# Множитель M.ANN=400 применяется к ОТКЛОНЕНИЯМ от базового прогноза
# (линейная аннуализация корректна для откликов); уровни годовой инфляции
# считаются компаундированием через M.yoy_pct().
LBL = {
    'pistar': ('Внешняя инфляция', M.ANN, '% год.'),
    'dG':     ('Госзакупки',         100, '%'),
    'SP':     ('Доля расходов бюдж.',100, '%'),
    'dInc':   ('Доходы бюджета',     100, '%'),
    'dI':     ('Инвестиции',         100, '%'),
    'dL':     ('Занятость',          100, '%'),
    'dC':     ('Потребление',        100, '%'),
    'dEX':    ('Экспорт',            100, '%'),
    'dIM':    ('Импорт',             100, '%'),
    'dPC':    ('Инфляция',         M.ANN, '% год.'),
    'rG':     ('Ключевая ставка',      1, 'п.п.'),
    'dNFX':   ('Курс (рост = обесц.)',100, '%'),
}


# ------------------------------------------------------------------ модель
def build(data='data_239Q2.csv', p=4, lam=1.0):
    """Загрузить данные и оценить модель. Возвращает (m, S, X)."""
    X = M.load(data)
    m = M.BVAR(X[V].values, p=p, lam=lam)
    S = la.cholesky(m.Sigma)          # точечная блочно-рекурсивная схема
    return m, S, X


# ------------------------------------------- 1-2. реакция на траекторию ставки
def response(m, S, rate_path, csv=None):
    """Сценарий: ставка принудительно идёт по rate_path, реализуется только
    шоками ДКП. Печатает отклонение КАЖДОЙ переменной от базового прогноза."""
    kR = V.index('rG')
    path = np.asarray(rate_path, float)
    H = len(path)
    base = m.forecast(H)                       # без шоков ДКП: правило работает
    scen, eps = m.scenario(path, S, kR)
    dev = scen - base                          # чистый эффект интервенции

    sig = S[kR, kR]
    nsig = float(np.abs(eps).max() / sig)

    print('\n' + '=' * 78)
    print('РЕАКЦИЯ НА ЗАДАННУЮ ТРАЕКТОРИЮ КЛЮЧЕВОЙ СТАВКИ')
    print('=' * 78)
    print('Горизонт            : %d кварталов' % H)
    print('Заданная ставка, %%  : %s' % np.round(path, 2))
    print('Базовая ставка, %%   : %s' % np.round(base[:, kR], 2))

    # --- поквартальные отклонения от базового сценария
    tab = {}
    for j, v in enumerate(V):
        nm, sc, un = LBL[v]
        tab['%s, %s' % (nm, un)] = np.round(sc * dev[:, j], 3)
    dY = M.agg_dY(scen) - M.agg_dY(base)
    tab['ВВП, %'] = np.round(100 * dY, 3)
    T1 = pd.DataFrame(tab, index=['h=%d' % (t + 1) for t in range(H)]).T

    print('\n--- Отклонение от базового прогноза, поквартально ---')
    print(T1.to_string())

    # --- накопленный эффект на УРОВЕНЬ (приросты складываются)
    cum = {}
    for j, v in enumerate(V):
        if v == 'rG':
            continue
        nm, _, _ = LBL[v]
        cum[nm + ', %'] = np.round(100 * np.cumsum(dev[:, j]), 3)
    cum['ВВП, %'] = np.round(100 * np.cumsum(dY), 3)
    T2 = pd.DataFrame(cum, index=['h=%d' % (t + 1) for t in range(H)]).T

    print('\n--- Накопленный эффект на уровень переменной, % ---')
    print(T2.to_string())

    # --- сводка по инфляции по годам
    kP = V.index('dPC')
    # Годовая инфляция — УРОВЕНЬ: квартальные лог-приросты складываются
    # (перемножение множителей) и переводятся в % через exp(.) - 1,
    # обратный переход год->квартал — корень 4-й степени, см. M.q_from_ann().
    print('\n--- Годовая инфляция в сценарии, % ---')
    for a in range(H // 4):
        y_s = M.yoy_pct(scen[4 * a:4 * a + 4, kP])
        y_b = M.yoy_pct(base[4 * a:4 * a + 4, kP])
        print('  год %d: сценарий %6.2f   базовый %6.2f   разница %+6.2f'
              % (a + 1, y_s, y_b, y_s - y_b))

    # --- диагностика правдоподобия (Leeper-Zha)
    print('\n--- Правдоподобие интервенции ---')
    print('с.к.о. шока ДКП        : %.3f п.п.' % sig)
    print('требуемые шоки, с.к.о. : %s' % np.round(eps / sig, 2))
    print('максимум               : %.1f сигма' % nsig)
    if nsig > 3.0:
        print('  ВНИМАНИЕ: больше 3 сигм. Это уже не отклонение от правила,')
        print('  а другое правило. Результат нельзя показывать как прогноз модели.')
    else:
        print('  В пределах 3 сигм — интервенция "скромная", результат интерпретируем.')

    if csv:
        pd.concat([T1, T2.rename(index=lambda s: 'накопл.: ' + s)]).to_csv(csv)
        print('\nтаблица сохранена: %s' % csv)
    return scen, dev, eps


# ------------------------------------------------------- 3. печать уравнений
def structural(m, S):
    """Структурная форма A0*Y_t = c + A1*Y_{t-1} + ... + B*eps_t.
    Нормировка diag(A0)=1, тогда B = diag(S) — с.к.о. структурных шоков."""
    A0raw = la.inv(S)
    d = np.diag(A0raw).copy()
    A0 = A0raw / d[:, None]                    # диагональ = 1
    n, p = m.n, m.p
    Pi = [m.B[l * n:(l + 1) * n].T for l in range(p)]   # приведённая форма
    A = [A0 @ Pi[l] for l in range(p)]                  # лаговые матрицы
    c = A0 @ m.B[-1]                                    # константы
    sig = np.diag(S).copy()                             # с.к.о. шоков
    return A0, A, c, sig


def print_equations(m, S, full=False, tol=0.01):
    A0, A, c, sig = structural(m, S)
    n, p = m.n, m.p

    print('\n' + '=' * 78)
    print('СТРУКТУРНАЯ ФОРМА:  A0*Y_t = c + A1*Y_{t-1} + ... + A%d*Y_{t-%d} + B*eps_t' % (p, p))
    print('=' * 78)
    print('Порядок переменных (он же порядок рекурсивной идентификации):')
    for j, v in enumerate(V):
        print('  %2d. %-7s %s' % (j + 1, v, LBL[v][0]))

    print('\n--- Матрица мгновенных связей A0 (диагональ нормирована к 1) ---')
    print('Элемент (i,j) — коэффициент при Y_j в уравнении i, взятый с ОБРАТНЫМ')
    print('знаком (т.к. слева стоит A0*Y_t). Нули выше диагонали — ограничения')
    print('идентификации: переменная не реагирует внутри квартала на тех, кто ниже.')
    A0d = np.round(A0, 3) + 0.0
    A0d[np.triu_indices(n, 1)] = 0.0        # нули идентификации, ровно
    print(pd.DataFrame(A0d, index=V, columns=V).to_string())

    print('\n--- С.к.о. структурных шоков (диагональ B) ---')
    for j, v in enumerate(V):
        print('  sigma[%-7s] = %8.4f   (%s)' % (v, sig[j], LBL[v][2]))

    print('\n--- Импакт-матрица S = A0^{-1}B: мгновенный отклик (столбец = шок) ---')
    print(pd.DataFrame(np.round(S, 4), index=V, columns=['eps_' + v for v in V]).to_string())

    print('\n' + '-' * 78)
    print('УРАВНЕНИЯ С ЧИСЛЕННЫМИ КОЭФФИЦИЕНТАМИ')
    print('(показаны слагаемые с |коэффициентом| > %.3f; лаг L%d = Y_{t-%d})'
          % (tol, p, p))
    print('-' * 78)
    for i, v in enumerate(V):
        terms = []
        for j in range(n):                     # мгновенные связи
            if j != i and abs(A0[i, j]) > tol:
                terms.append('%+.3f*%s(t)' % (-A0[i, j], V[j]))
        for l in range(p):                     # лаги
            for j in range(n):
                if abs(A[l][i, j]) > tol:
                    terms.append('%+.3f*%s(t-%d)' % (A[l][i, j], V[j], l + 1))
        rhs = ('%+.4f ' % c[i]) + ' '.join(terms) + \
              ' %+.4f*eps_%s(t)' % (sig[i], v)
        print('\n[%2d] %s(t) =' % (i + 1, v))
        for ln in textwrap.wrap(rhs, 72):
            print('     ' + ln)

    if full:
        for l in range(p):
            print('\n--- Лаговая матрица A%d (строка = уравнение) ---' % (l + 1))
            print(pd.DataFrame(np.round(A[l], 4), index=V, columns=V).to_string())


# ---------------------------------------------- дополнительно: чистый шок ДКП
def print_irf(m, S, h=12):
    kR = V.index('rG')
    ir = m.irf(h, S)
    dY = M.agg_dY(ir)
    print('\n' + '=' * 78)
    print('ОТКЛИК НА ШОК ДКП В +1 С.К.О. (= %.3f п.п. ставки)' % S[kR, kR])
    print('=' * 78)
    tab = {}
    for j, v in enumerate(V):
        nm, sc, un = LBL[v]
        tab['%s, %s' % (nm, un)] = np.round(sc * ir[:, j, kR], 3)
    tab['ВВП, %'] = np.round(100 * dY[:, kR], 3)
    print(pd.DataFrame(tab, index=['h=%d' % k for k in range(h + 1)]).T.to_string())


# ------------------------------------------------------------------- запуск
def menu(m, S):
    print('\n1 — одно значение ставки   2 — траектория ставки'
          '   3 — уравнения   4 — шок ДКП   0 — выход')
    ch = input('выбор: ').strip()
    if ch == '1':
        r = float(input('ставка, %: ').replace(',', '.'))
        H = input('сколько кварталов [12]: ').strip()
        response(m, S, np.full(int(H) if H else 12, r))
    elif ch == '2':
        s = input('ставки через запятую, в хронологическом порядке: ')
        response(m, S, [float(x) for x in s.replace(',', ' ').split()])
    elif ch == '3':
        print_equations(m, S, full=input('печатать лаговые матрицы? [y/N]: ')
                        .strip().lower().startswith('y'))
    elif ch == '4':
        print_irf(m, S)
    return ch != '0'


def main():
    a = sys.argv[1:]

    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default

    m, S, X = build(opt('--data', 'data_239Q2.csv'),
                    int(opt('--lags', 4)), float(opt('--lam', 1.0)))
    print('данные: %s .. %s (%d кв.);  макс. модуль корня %.3f'
          % (X.index[0], X.index[-1], len(X), m.max_root()))

    if '--rate' in a:
        response(m, S, np.full(int(opt('--H', 12)), float(opt('--rate'))),
                 csv=opt('--csv'))
    elif '--rates' in a:
        path = [float(x) for x in opt('--rates').replace(',', ' ').split()]
        response(m, S, path, csv=opt('--csv'))
    elif '--equations' in a:
        print_equations(m, S, full='--full' in a)
    elif '--irf' in a:
        print_irf(m, S, int(opt('--H', 12)))
    else:
        while menu(m, S):
            pass


if __name__ == '__main__':
    main()
