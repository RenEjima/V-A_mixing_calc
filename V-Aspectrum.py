import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

#constants
f_pi = 0.093 #pion's decay constant [GeV]
massK = 0.493 #Kaon [GeV]
alpha = 1/137 #coupling const of ele-mag interaction
gRho = 0.119 #rho meson[GeV^2]
mass_rho = 0.770 #rho meson's mass[GeV/c^2]
mass_phi = 1.020 #phi meson's mass[GeV/c^2]
mass_f1_1420 = 1.420 #f1(1420) [GeV]

#phi meson's self-energy at vacuum
Sigma_VL = 0
Sigma_VT = 0
Sigma_AL = 0
Sigma_AT = 0

#density and temperature
baryon_density = 1.0 #times of nuclear
temperature = 0.010 #[GeV]

#swith of chiral symmetry resotration
withCSR = 0

#see dilepton production rate at p
atP = 1.5 #[GeV]

def QCDsumruleMass(massVac):
    massMed = massVac*(1-0.034*baryon_density)
    return massMed

def decayInsideRate(p):
    a=13.44128233
    b=-10.14421448
    c=0.73734453
    d=8.9882025
    e=-0.80843066
    f=0.11062088
    return c*a**(b*p)+f*d**(e*p)

#Info of Vector/Axial-vector meson
gV = np.sqrt(2)/3*mass_phi**2/mass_rho**2*gRho

if withCSR == 0:
    massV = mass_phi
    massA = mass_f1_1420
elif withCSR == 1:
    massV = QCDsumruleMass(mass_phi)
    massA = QCDsumruleMass(mass_f1_1420)

def DispersionRelation_VT(p):
    p_0 = np.sqrt(0.5*(massV**2+massA**2 - np.sqrt((massA**2-massV**2)**2+16*c**2*p**2)) + p**2)
    return p_0

def DispersionRelation_AT(p):
    p_0 = np.sqrt(0.5*(massV**2+massA**2 + np.sqrt((massA**2-massV**2)**2+16*c**2*p**2)) + p**2)
    return p_0

def DispersionRelation_VL(p):
    p_0 = np.sqrt(massV**2+p**2)
    return p_0

def selfEnergyModification(s):
    mK=massK
    aK = 0.22
    aKbar = 0.45
    bK = 3/(8*f_pi**2)
    rhoS = 0.1
    rhoB = 0.004
    mKstar = np.sqrt(mK**2-aK*rhoS+(bK*rhoB)**2)+bK*rhoB #MeV K's inmedium mass
    mKbarstar = np.sqrt(mK**2-aKbar*rhoS+(bK*rhoB)**2)-bK*rhoB #MeV Kbar's inmedium mass
    print("mKstar = ",mKstar," mKbarstar = ",mKbarstar)
    mKstar = 0.56
    mKbarstar = 0.23
    ks = np.sqrt((s - (mKstar+mKbarstar)**2)*(s - (mKstar-mKbarstar)**2))/(2*np.sqrt(s))
    Gamma_phiS=1.69*4/3*ks**3/s
    print("modified Gamma = ", Gamma_phiS)
    return Gamma_phiS

def ImVector_CurrentCurrentCorrelationFx_L(s,p):
    ImG_VL = (gV/massV)**2*s*massV*Gamma_VL/((s-massV**2+Sigma_VL)**2+massV**2*Gamma_VL**2)
    return ImG_VL

def ImVector_CurrentCurrentCorrelationFx_T(s,p):
    ReunitD = (s-massV**2)*(s-massA**2)-massV*massA*Gamma_VT*Gamma_AT-4*c**2*p**2
    ImunitD = (s-massV**2)*massA*Gamma_AT+(s-massA**2)*massV*Gamma_VT
    ReunitN = -s**2+s*massA**2+4*c**2*p**2
    ImunitN = -s*massA*Gamma_AT
    ImG_VT = (gV/massV)**2*(-ReunitN*ImunitD+ImunitN*ReunitD)/(ReunitD**2+ImunitD**2)
    return ImG_VT

def ImVector_CurrentCurrentCorrelationFx_T_Omega(s,p):
    omegaV = DispersionRelation_VT(p)
    omegaA = DispersionRelation_AT(p)
    ReunitD = (s-omegaV)*(s-omegaA)-massV*massA*Gamma_VT*Gamma_AT-4*c**2*p**2
    ImunitD = (s-omegaV)*massA*Gamma_AT+(s-omegaA)*massV*Gamma_VT
    ReunitN = -s**2+s*omegaA+4*c**2*p**2
    ImunitN = -s*massA*Gamma_AT
    ImG_VT = (gV/massV)**2*(-ReunitN*ImunitD+ImunitN*ReunitD)/(ReunitD**2+ImunitD**2)
    return ImG_VT

spectralFunction_L_noMix = [0]*3000
spectralFunction_T_noMix = [0]*3000
spectralFunction_T_2_noMix = [0]*3000
spectralFunction_T_omega_noMix = [0]*3000
spectralFunction_T_omega_2_noMix = [0]*3000
spectralFunction_vac = [0]*3000

#without mixing
c = 0 #[GeV]
for p_MeV in range(1000,2000): #Integrate dilepton production rate by momentum from 0 to 2GeV
    p = p_MeV/1000 #convert from MeV to GeV
    s_MeV = np.arange(1,3001,1)   #range of s
    s = s_MeV/1000 #convert from MeV to GeV
    '''
    Gamma_VL = selfEnergyModification(s)
    Gamma_VT = selfEnergyModification(s)
    '''
    GammaVac = 0.00426
    #Gamma_VL = 0.00426
    #Gamma_VT = 0.00426
    Gamma_VL = 0.061
    Gamma_VT = 0.061
    Gamma_AL = 0.0549
    Gamma_AT = 0.0549
    #in vacuum
    C_0 = 1e-48
    spectralFunction_vac = C_0*GammaVac/(2*math.pi*(pow(np.sqrt(s)-massV,2)+GammaVac*GammaVac/4))

    ImGv_L = ImVector_CurrentCurrentCorrelationFx_L(s,p)
    ImGv_T = ImVector_CurrentCurrentCorrelationFx_T(s,p)
    ImGv_T_omega = ImVector_CurrentCurrentCorrelationFx_T_Omega(s,p)

    #p_0_L = DispersionRelation_VL(p)
    #p_0_T = DispersionRelation_VT(p)
    #p_0_T_2 = DispersionRelation_AT(p)

    p_0_L = np.sqrt(s+p**2)
    p_0_T = np.sqrt(s+p**2)
    p_0_T_2 = np.sqrt(s+p**2)

    dilepton_productionRate_L_noMix = alpha**2/(math.pi*s)*ImGv_L/(np.exp(p_0_L/temperature)-1)
    spectralFunction_L_noMix = spectralFunction_L_noMix+(dilepton_productionRate_L_noMix)/(2*p_0_L)
    dilepton_productionRate_T_noMix = alpha**2/(math.pi*s)*ImGv_T/(np.exp(p_0_T/temperature)-1)
    spectralFunction_T_noMix = spectralFunction_T_noMix+(dilepton_productionRate_T_noMix)/(2*p_0_T)
    dilepton_productionRate_T_2_noMix = alpha**2/(math.pi*s)*ImGv_T/(np.exp(p_0_T_2/temperature)-1)
    spectralFunction_T_2_noMix = spectralFunction_T_2_noMix+(dilepton_productionRate_T_2_noMix)/(2*p_0_T_2)

    dilepton_productionRate_T_omega_noMix = alpha**2/(math.pi*s)*ImGv_T_omega/(np.exp(p_0_T/temperature)-1)
    spectralFunction_T_omega_noMix = spectralFunction_T_omega_noMix+(dilepton_productionRate_T_omega_noMix)/(2*p_0_T)
    dilepton_productionRate_T_omega_2_noMix = alpha**2/(math.pi*s)*ImGv_T_omega/(np.exp(p_0_T_2/temperature)-1)
    spectralFunction_T_omega_2_noMix = spectralFunction_T_omega_2_noMix+(dilepton_productionRate_T_omega_2_noMix)/(2*p_0_T_2)

    if p == atP:
        dilepton_productionRate_L_noMix_save = dilepton_productionRate_L_noMix
        dilepton_productionRate_T_noMix_save = dilepton_productionRate_T_noMix

spectralFunction_L = [0]*3000
spectralFunction_T = [0]*3000
spectralFunction_T_2 = [0]*3000
spectralFunction_T_omega = [0]*3000
spectralFunction_T_omega_2 = [0]*3000

#mixing strength
c = 0.1*baryon_density #[GeV]

for p_MeV in range(1000,2000): #Integrate dilepton production rate by momentum from 0 to 2GeV
    p = p_MeV/1000 #convert from MeV to GeV
    s_MeV = np.arange(1,3001,1)   #range of s
    s = s_MeV/1000 #convert from MeV to GeV
    '''
    Gamma_VL = selfEnergyModification(s)
    Gamma_VT = selfEnergyModification(s)
    '''
    #Gamma_VL = 0.00426
    #Gamma_VT = 0.00426
    Gamma_VL = 0.061
    Gamma_VT = 0.061
    Gamma_AL = 0.0549
    Gamma_AT = 0.0549

    ImGv_L = ImVector_CurrentCurrentCorrelationFx_L(s,p)
    ImGv_T = ImVector_CurrentCurrentCorrelationFx_T(s,p)
    ImGv_T_omega = ImVector_CurrentCurrentCorrelationFx_T_Omega(s,p)

    #p_0_L = DispersionRelation_VL(p)
    #p_0_T = DispersionRelation_VT(p)
    #p_0_T_2 = DispersionRelation_AT(p)

    p_0_L = np.sqrt(s+p**2)
    p_0_T = np.sqrt(s+p**2)

    dilepton_productionRate_L = alpha**2/(math.pi*s)*ImGv_L/(np.exp(p_0_L/temperature)-1)
    bkg_L = alpha**2/(math.pi*s)/(np.exp(p_0_L/temperature)-1)
    spectralFunction_L = spectralFunction_L+(dilepton_productionRate_L)/(2*p_0_L)

    dilepton_productionRate_T = alpha**2/(math.pi*s)*ImGv_T/(np.exp(p_0_T/temperature)-1)
    bkg_T = alpha**2/(math.pi*s)/(np.exp(p_0_T/temperature)-1)
    spectralFunction_T = spectralFunction_T+(dilepton_productionRate_T)/(2*p_0_T)

    if p==atP:
        fig, ax = plt.subplots()
        ax.plot(np.sqrt(s), ImGv_L,label="Longitudinal")
        ax.plot(np.sqrt(s), ImGv_T,label="Transverse")
        ax.legend()
        ax.set_xticks(np.linspace(0.,1.75,43),minor=True)
        ax.grid(which="major",alpha=0.5)
        ax.grid(which="minor",alpha=0.2)
        ax.set_xlabel(r"$M[{\rm{GeV}}]$")
        ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
        plt.yscale('log')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(np.sqrt(s), dilepton_productionRate_L,label="Longitudinal")
        ax.plot(np.sqrt(s), dilepton_productionRate_T,label="Transverse")
        ax.plot(np.sqrt(s), (dilepton_productionRate_L+2*(dilepton_productionRate_T))/3,label="Average")
        ax.plot(np.sqrt(s), (dilepton_productionRate_L_noMix_save+2*dilepton_productionRate_T_noMix_save)/3, label="w/o V-A mixing")
        ax.legend()
        ax.set_xticks(np.linspace(0.,1.75,43),minor=True)
        ax.grid(which="major",alpha=0.5)
        ax.grid(which="minor",alpha=0.2)
        ax.set_xlabel(r"$M[{\rm{GeV}}]$")
        ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
        plt.yscale('log')
        plt.show()

        rate = 1e-3

        fig, ax = plt.subplots()
        ax.plot(np.sqrt(s), bkg_L*rate,label="bkg?")
        ax.plot(np.sqrt(s), (dilepton_productionRate_L+2*(dilepton_productionRate_T))/3,label="spectral Fx")
        ax.legend()
        ax.set_xticks(np.linspace(0.,1.75,43),minor=True)
        ax.grid(which="major",alpha=0.5)
        ax.grid(which="minor",alpha=0.2)
        ax.set_xlabel(r"$M[{\rm{GeV}}]$")
        ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
        plt.yscale('log')
        plt.show()

        fig, ax = plt.subplots()
        #ax.plot(np.sqrt(s), dilepton_productionRate_L-bkg_L*rate,label="Longitudinal")
        #ax.plot(np.sqrt(s), dilepton_productionRate_T-bkg_T*rate,label="Transverse")
        ax.plot(np.sqrt(s), (dilepton_productionRate_L+2*(dilepton_productionRate_T))/3-bkg_L*rate,label="Average")
        ax.legend()
        ax.set_xticks(np.linspace(0.,1.75,43),minor=True)
        ax.grid(which="major",alpha=0.5)
        ax.grid(which="minor",alpha=0.2)
        ax.set_xlabel(r"$M[{\rm{GeV}}]$")
        ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
        plt.yscale('log')
        #ax.set_xlim(0.75,1.5)
        plt.show()


        spectralFunction_L_atP = dilepton_productionRate_L/(2*p_0_L)
        spectralFunction_T_atP = dilepton_productionRate_T/(2*p_0_T)
        fig, ax = plt.subplots()
        ax.plot(np.sqrt(s), spectralFunction_L_atP,label="Longitudinal")
        ax.plot(np.sqrt(s), spectralFunction_T_atP,label="Transverse")
        ax.plot(np.sqrt(s), (spectralFunction_L_atP+2*(spectralFunction_T_atP))/3,label="Average")
        ax.legend()
        ax.set_xticks(np.linspace(0.,1.75,43),minor=True)
        ax.grid(which="major",alpha=0.5)
        ax.grid(which="minor",alpha=0.2)
        ax.set_xlabel(r"$M[{\rm{GeV}}]$")
        ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
        plt.yscale('log')
        plt.show()

fig, ax = plt.subplots()
#ax.plot(np.sqrt(s), spectralFunction_vac, label="vacuum")
ax.plot(np.sqrt(s), (spectralFunction_L_noMix+2*spectralFunction_T_noMix),label="w/o V-A mixing")
#ax.plot(np.sqrt(s), spectralFunction_L,label="Longitudinal")
#ax.plot(np.sqrt(s), spectralFunction_T,label="Transverse")
ax.plot(np.sqrt(s), (spectralFunction_L+2*(spectralFunction_T))/3,label="Average")
ax.legend()
ax.set_xticks(np.linspace(0.,1.75,43),minor=True)
ax.grid(which="major",alpha=0.5)
ax.grid(which="minor",alpha=0.2)
ax.set_xlabel(r"$M[{\rm{GeV}}]$")
ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
plt.yscale('log')
plt.show()
