# -*- coding: utf-8 -*-
"""Выгрузка всех числовых коэффициентов и состояния редуцированной КПМ ЦБ."""
import os, numpy as np, pandas as pd, sys, io
sys.stdout.reconfigure(encoding='utf-8')
P=os.path.dirname(os.path.abspath(__file__))
src=io.open(os.path.join(P,"cbr_final2.py"),encoding='utf-8').read().split("VARS=['y','pi','z','i','pi4']")[0]
src=src.replace("sys.stdout.reconfigure(encoding='utf-8')","")
buf=io.StringIO(); old=sys.stdout; sys.stdout=buf
g = {'__name__':'__dump__', '__file__': os.path.join(P, 'cbr_final2.py')}; exec(compile(src,'cbr_part','exec'),g); sys.stdout=old
C=g['C']; RR_EQ=g['RR_EQ']; Y_GAP=g['Yg']; LZ_GAP=g['LZg']; N_GAP=g['Ng']
PIE=g['PIE']; PIE4=g['PIE4']; RS=g['RS']; RR=g['RR']; RR_GAP=g['RR_GAP']
SH_PIE=g['SH_PIE']; RES_IS=g['RES_IS']; PREM=g['PREM_pers']; RES_LZ=g['RES_LZ']
RMC=g['RMC']; DF=g['DF']

am,ad=C['alpha_m'],C['alpha_d']; cl,cr,wl=C['c_lag_d'],C['c_rmc_d'],C['w_exp_lag']
dl,dg,dr=C['d_lead'],C['d_lag'],C['d_rr']; dt,dc,wc=C['delta'],C['delta_cc'],C['w_cc']
g1,g2,g3=C['gamma1'],C['gamma2'],C['gamma3']
print("== ПРИВЕДЁННЫЕ ЧИСЛОВЫЕ КОЭФФИЦИЕНТЫ ==")
print(f" Филлипс: pi(-1) {cl+(1-cl)*wl:.4f}   pi(+1) {(1-cl)*(1-wl):.4f}   сумма {cl+(1-cl)*wl+(1-cl)*(1-wl):.4f}")
print(f"          c_rmc*(1-am)*(1-ad) [y] = {cr*(1-am)*(1-ad):.5f}")
print(f"          c_rmc*(1-am)*ad     [n] = {cr*(1-am)*ad:.5f}")
print(f"          c_rmc*am            [z] = {cr*am:.5f}")
print(f" IS:      y(+1) {dl:.2f}  y(-1) {dg:.2f}  rr_gap {-dr:.2f}")
print(f"          EW1pi = {wl:.2f}*pi(-1) + {1-wl:.2f}*pi(+1)")
print(f" UIP:     z(+1) {(1-wc)*dt:.4f}   z(-1) {(1-wc)*(1-dt)+wc*(1-dc):.4f}")
print(f"          i {-(1-wc)/4:.5f}   pi(-1) {(1-wc)*wl/4:.5f}   pi(+1) {(1-wc)*(1-wl)/4:.5f}")
print(f"          конст. {(1-wc)/4:.4f}*(r_bar + prem)")
print(f" Правило: i(-1) {g1:.2f}   pi4(+3) {(1-g1)*(1+g2):.4f}   y {(1-g1)*g3:.4f}")
print(f"          конст. {(1-g1):.2f}*(r_bar - {g2}*pi_tar) = {(1-g1)*(RR_EQ[-1]-g2*C['PIE_TAR_SS']):.4f}")
print(f" Цель:    PIE_TAR_SS = 100*ln(1.04) = {C['PIE_TAR_SS']:.4f}")
print("\n== СОСТОЯНИЕ НА 239Q2 (вход в решатель) ==")
print(f" RS(-1)={RS[-1]:.4f}  PIE4(-1)={PIE4[-1]:.4f}")
print(f" pi(-1..-4) = {PIE[-1]:.3f}, {PIE[-2]:.3f}, {PIE[-3]:.3f}, {PIE[-4]:.3f}")
print(f" Y_GAP(-1..-3)  = {Y_GAP[-1]:+.3f}, {Y_GAP[-2]:+.3f}, {Y_GAP[-3]:+.3f}")
print(f" LZ_GAP(-1..-3) = {LZ_GAP[-1]:+.3f}, {LZ_GAP[-2]:+.3f}, {LZ_GAP[-3]:+.3f}")
print(f" N_GAP(-1)={N_GAP[-1]:+.3f}   RMC(-1)={RMC[-1]:+.3f}   DF_GAP(-1)={DF[-1]:+.3f}")
print(f" RR={RR[-1]:.3f}  RR_EQ={RR_EQ[-1]:.3f}  RR_GAP={RR_GAP[-1]:+.3f}")
print(f" RS_NEUTRAL = RR_EQ + PIE_TAR = {RR_EQ[-1]+C['PIE_TAR_SS']:.3f}")
print("\n== ШОКИ (начальные значения и затухание) ==")
def ar1(x):
    L=np.r_[np.nan,x[:-1]]; ok=~np.isnan(x)&~np.isnan(L)
    return float(np.clip((L[ok]@x[ok])/(L[ok]@L[ok]),-0.95,0.97))
print(f" SHCK_PIE:  {SH_PIE[-1]:+.3f}, rho={ar1(SH_PIE):.3f}  (ЦБ: rho=0.50, sd=1.50; здесь sd={np.nanstd(SH_PIE):.2f})")
print(f" SHCK_DF:   {RES_IS[-1]:+.3f}, rho={ar1(RES_IS):.3f}  (ЦБ: rho=0.50, sd=1.00; здесь sd={np.nanstd(RES_IS):.2f})")
print(f" PREM:      {PREM[-1]:+.3f}, rho={C['rho_prem_eq']:.2f} (ЦБ)   sd устойч.={PREM.std():.2f}")
print(f" N_GAP:     {N_GAP[-1]:+.3f}, rho=0.94 (подобрано)")
