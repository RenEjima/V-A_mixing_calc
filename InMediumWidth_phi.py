import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
from matplotlib import gridspec

Gamma_f1_vac = 54.9
f_pi_0 = 93


#width

def mK_med(density_rate):
    #return (80.0/3.0)*density_rate+500.0
    return (510.0-495.0)*density_rate+495.0

def mKbar_med(density_rate):
    #return (-320.0/3.0)*density_rate+500.0
    return (380.0-495.0)*density_rate+495.0

def Gamma_phi_med(mK_med, mKbar_med):
    sqrt_s = 1020
    ks = np.sqrt((sqrt_s**2 - (mK_med+mKbar_med)**2)*(sqrt_s**2 - (mK_med-mKbar_med)**2))/(2*sqrt_s)
    Gamma_phiS=((1.69*4/3)*ks**3)/sqrt_s**2
    return Gamma_phiS

def Gamma_f1_med(Gamma_phi_med,f_pi):
    return Gamma_f1_vac*(f_pi/f_pi_0)**2+Gamma_phi_med*(1.0-(f_pi/f_pi_0)**2)

Gamma_phi_List = []
Gamma_f1_List = []
density_rate_List = []

print("Gamma_phi at nuclear density = ",Gamma_phi_med(510,380))

for density_rate_1000 in range(0,3000): 
    
    density_rate = density_rate_1000/1000
    f_pi = -93.0*0.3*density_rate+93
    #f_pi = 93.0*np.sqrt((-density_rate+1.0/0.51)/0.51)

    mK_med_at_thisDensity = mK_med(density_rate)
    mKbar_med_at_thisDensity = mKbar_med(density_rate)
    
    Gamma_phi_at_thisDensity = Gamma_phi_med(mK_med_at_thisDensity, mKbar_med_at_thisDensity)
    Gamma_f1_at_thisDensity = Gamma_f1_med(Gamma_phi_at_thisDensity, f_pi)
    
    density_rate_List.append(density_rate)
    Gamma_phi_List.append(Gamma_phi_at_thisDensity)
    Gamma_f1_List.append(Gamma_f1_at_thisDensity)

fig, ax = plt.subplots()
ax.plot(density_rate_List,Gamma_phi_List,label=r"$\Gamma_{\phi}^{\rm{med}}(s=m_{\phi}^2)$")
ax.plot(density_rate_List,Gamma_f1_List,label=r"$\Gamma_{f_1(1420)}^{\rm{med}}(s=m_{f_1(1420)}^2)$")
ax.legend()
ax.set_xlabel(r"$\rho/\rho_0$")
ax.set_ylabel("Width")
#plt.yscale('log')
ax.set_xlim(0.0, 3.0)
#ax.set_ylim(0.0, 70.0)
plt.grid(True)
plt.show()


#mass

mass_phi = 1020
mass_f1 = 1420
g_gauge = 6.61

def mass_A_med(massV, f_pi):
    return massV*massV / np.sqrt(massV*massV - g_gauge*g_gauge*f_pi*f_pi)

mass_phi_List = []
mass_f1_List = []
f_pi_rate_List = []

for f_pi_rate_1000 in range(0,1000): 
    
    f_pi_rate = f_pi_rate_1000/1000
    
    mass_phi_at_thisDensity = 1020
    mass_f1_at_thisDensity = mass_A_med(mass_phi, f_pi_rate*93)
    
    f_pi_rate_List.append(1.0-f_pi_rate)
    mass_phi_List.append(mass_phi_at_thisDensity)
    mass_f1_List.append(mass_f1_at_thisDensity)

fig, ax = plt.subplots()
ax.plot(f_pi_rate_List,mass_phi_List,label=r"$m_{\phi}^{\rm{med}}$")
ax.plot(f_pi_rate_List,mass_f1_List,label=r"$m_{f_1(1420)}^{\rm{med}}$")
ax.legend()
ax.set_xlabel(r"$1-f_\pi/f_\pi^{\rm{vac}}$")
ax.set_ylabel("Mass")
#plt.yscale('log')
ax.set_xlim(0.0, 1.0)
plt.grid(True)
plt.show()