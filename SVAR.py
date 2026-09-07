import svar_soe as M
X = M.load('data_239Q2.csv')
Y = X[M.V].values
m = M.BVAR(Y, p=4)
n = 12
p = 4
for i in range(n):
    print(f"\nУравнение {i+1}: {M.V[i]}")
    print(f"Константа: {m.B[-1, i]:.6f}")
    for l in range(p):
        for j in range(n):
            coef = m.B[l*n + j, i]
            if abs(coef) > 0.001:
                print(f"  {M.V[j]}_{{-{l+1}}}: {coef:.6f}")
print("\nКовариационная матрица Sigma:")
print(m.Sigma.round(6))