import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
from matplotlib import gridspec

#constants
f_pi_0 = 0.093 #pion's decay constant [GeV]
f_pi = 0.7*f_pi_0 #chiral symmetry's resotration
#alpha = 1/137 #coupling const of ele-mag interaction
gRho = 0.119 #rho meson[GeV^2]
g_gauge = 6.61 #gauge coupling
mass_rho = 0.770 #rho meson's mass[GeV/c^2]
mass_phi = 1.020 #phi meson's mass[GeV/c^2]
mass_f1_1420 = 1.420 #f1(1420)'s mass [GeV/c^2]
Gamma_phi = 0.00426 #phi meson's width[GeV/c^2]
Gamma_f1_1420 = 0.0549 #f1(1420)'s width[GeV/c^2]
A_target = 63 #Cu target
#A_target = 206 #Pb

#radius of Atom
r_A = 1.2*pow(A_target,1/3) #[fm]
print("r_A=",r_A)

#phi meson's self-energy at vacuum
Sigma_VL = 0
Sigma_VT = 0
Sigma_AL = 0
Sigma_AT = 0

#density and temperature
baryon_density = 1.0 #times of nuclear
#temperature = 0.010 #[GeV]

#see dilepton production rate at p
atP = 1.0 #[GeV]

def woodsSaxonDist(r):
    return 1/(1+np.exp((r-r_A)/0.5))

def decayInsideRate(p):
    a=2.67700347
    b=-4.30764956
    c=0.78604455
    d=2.33265816
    e=-0.62388036
    f=0.24677148
    return c*a**(b*p)+f*d**(e*p)

def momentumDist(p):
    a=160.0
    b=0.75
    c=6.0
    d=1.5
    e=5.1
    momentumDist = a*p/(b+(p/c)**d)**e
    return momentumDist

#Info of Vector/Axial-vector meson
gV = np.sqrt(2)/3*mass_phi**2/mass_rho**2*gRho

#massV = mass_phi*(1-0.034*baryon_density/1) #phi meson's mass shift from QCD sum rule
massV = mass_phi
massA = massV*massV / np.sqrt(massV*massV - g_gauge*g_gauge*f_pi*f_pi)
#massA = mass_f1_1420

def DispersionRelation_VT(p):
    p_0 = np.sqrt(0.5*(massV**2+massA**2 - np.sqrt((massA**2-massV**2)**2+16*c**2*p**2)) + p**2)
    return p_0

def DispersionRelation_AT(p):
    p_0 = np.sqrt(0.5*(massV**2+massA**2 + np.sqrt((massA**2-massV**2)**2+16*c**2*p**2)) + p**2)
    return p_0

def DispersionRelation_VL(p):
    p_0 = np.sqrt(massV**2+p**2)
    return p_0

def selfEnergyModification(sqrt_s):
    mKstar = 0.51
    mKbarstar = 0.38
    #mKstar = 0.493
    #mKbarstar = 0.493
    ks = np.sqrt((sqrt_s**2 - (mKstar+mKbarstar)**2)*(sqrt_s**2 - (mKstar-mKbarstar)**2))/(2*sqrt_s)
    Gamma_phiS=((1.69*4/3)*ks**3)/sqrt_s**2
    return Gamma_phiS

def selfEnergyModification_thisDensity(sqrt_s,density):
    mKstar = 0.022*density+0.493
    mKbarstar = -0.113*density+0.493
    ks = np.sqrt((sqrt_s**2 - (mKstar+mKbarstar)**2)*(sqrt_s**2 - (mKstar-mKbarstar)**2))/(2*sqrt_s)
    Gamma_phiS=((1.69*4/3)*ks**3)/sqrt_s**2
    return Gamma_phiS

def ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,p):
    ImG_VL = (gV/massV)**2*sqrt_s**2*massV*Gamma_VL/((sqrt_s**2-massV**2+Sigma_VL)**2+massV**2*Gamma_VL**2)
    return ImG_VL

def ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,p):
    ReunitD = (sqrt_s**2-massV**2)*(sqrt_s**2-massA**2)-massV*massA*Gamma_VT*Gamma_AT-4*c**2*p**2
    ImunitD = (sqrt_s**2-massV**2)*massA*Gamma_AT+(sqrt_s**2-massA**2)*massV*Gamma_VT
    ReunitN = -sqrt_s**4+sqrt_s**2*massA**2+4*c**2*p**2
    ImunitN = -sqrt_s**2*massA*Gamma_AT
    ImG_VT = (gV/massV)**2*(-ReunitN*ImunitD+ImunitN*ReunitD)/(ReunitD**2+ImunitD**2)
    Nume = massV*(4*c**2*p**2*massA*massV*Gamma_AT+sqrt_s**2*(-4*c**2*p**2+sqrt_s**4)*Gamma_VT+sqrt_s**2*massA**4*Gamma_VT+massA**2*(4*c**2*p**2-2*sqrt_s**4+sqrt_s**2*Gamma_AT**2)*Gamma_VT)
    Deno = (massA*(sqrt_s**2-massV**2)*Gamma_AT+(sqrt_s**2-massA**2)*massV*Gamma_VT)**2+(-4*c**2*p**2+(sqrt_s**2-massA**2)*(sqrt_s**2-massV**2)-massA*massV*Gamma_AT*Gamma_VT)**2
    return ImG_VT
    #return (gV/massV)**2*Nume/Deno

def Gamma_0(sqrt_s):
    return selfEnergyModification(sqrt_s)*sqrt_s/mass_phi
def Gamma_ee(sqrt_s):
    return selfEnergyModification(sqrt_s)*pow(mass_phi,3)/pow(sqrt_s,3)
def nonRelativisticBWdist(sqrt_s):
    return selfEnergyModification(sqrt_s)/(2*math.pi*(pow(sqrt_s-mass_phi,2)+selfEnergyModification(sqrt_s)*selfEnergyModification(sqrt_s)/4))
    #return Gamma_phi/(2*math.pi*(pow(sqrt_s-mass_phi,2)+Gamma_phi*Gamma_phi/4))
def relativisticBWdist(sqrt_s):
    return sqrt_s*Gamma_0(sqrt_s)*Gamma_ee(sqrt_s)/(pow(sqrt_s**2-mass_phi*mass_phi,2)+mass_phi*mass_phi*Gamma_0(sqrt_s)*Gamma_0(sqrt_s))
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

DileptonProductionRate_L_noMix = [0]*3111
DileptonProductionRate_T_noMix = [0]*3111
DileptonProductionRate_vac = [0]*3111

Gamma_V_eff = selfEnergyModification(1.02)
Gamma_A_eff = Gamma_f1_1420*(f_pi/f_pi_0)**2 + Gamma_V_eff*(1-(f_pi/f_pi_0)**2)
print("eff gamma: V = ", Gamma_V_eff, " : A = ", Gamma_A_eff)

#without mixing
c = 0 #[GeV]
for p_MeV in range(1000,2000): #Integrate dilepton production rate by momentum from 0 to 2GeV
    p = p_MeV/1000 #convert from MeV to GeV
    sqrt_s_MeV = np.arange(890,4001,1)   #range of s
    sqrt_s = sqrt_s_MeV/1000 #convert from MeV to GeV
    #GammaVac = Gamma_phi
    #Gamma_VL = 0.0042
    #Gamma_VT = 0.0042
    Gamma_VL = selfEnergyModification(sqrt_s)
    Gamma_VT = selfEnergyModification(sqrt_s)
    Gamma_AL = 0.0549
    Gamma_AT = 0.0549

    ImGv_L = ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,p)
    ImGv_T = ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,p)

    p_0_L = np.sqrt(sqrt_s**2+p**2)
    p_0_T = np.sqrt(sqrt_s**2+p**2)

    #DileptonProductionRate_vac = Gamma_phi/(2*math.pi*(pow(sqrt_s-massV,2)+Gamma_phi*Gamma_phi/4))
    SpectralFx_vac = nonRelativisticBWdist(sqrt_s)/sum(nonRelativisticBWdist(sqrt_s))
    DileptonProductionRate_vac = DileptonProductionRate_vac+SpectralFx_vac/(2*p_0_L)*momentumDist(p)*(1.0-decayInsideRate(p))

    #SpectralFx_L_noMix = alpha**2/(math.pi*sqrt_s**2)*ImGv_L/(np.exp(p_0_L/temperature)-1)
    SpectralFx_L_noMix = ImGv_L
    DileptonProductionRate_L_noMix = DileptonProductionRate_L_noMix+(SpectralFx_L_noMix)/(2*p_0_L)/sum(SpectralFx_L_noMix)*momentumDist(p)*decayInsideRate(p)
    #SpectralFx_T_noMix = alpha**2/(math.pi*sqrt_s**2)*ImGv_T/(np.exp(p_0_T/temperature)-1)
    SpectralFx_T_noMix = ImGv_T
    DileptonProductionRate_T_noMix = DileptonProductionRate_T_noMix+(SpectralFx_T_noMix)/(2*p_0_T)/sum(SpectralFx_L_noMix)*momentumDist(p)*decayInsideRate(p)
    '''
    if p == atP:
        SpectralFx_L_noMix_save = SpectralFx_L_noMix
        SpectralFx_T_noMix_save = SpectralFx_T_noMix
    '''

DileptonProductionRate_L = [0]*3111
DileptonProductionRate_T = [0]*3111
DileptonProductionRate_L_wo_thermal = [0]*3111
DileptonProductionRate_T_wo_thermal = [0]*3111

#mixing strength
c = 0.1*baryon_density #[GeV]
for p_MeV in range(1000,3000): #Integrate dilepton production rate by momentum from 0 to 2GeV
    p = p_MeV/1000 #convert from MeV to GeV
    sqrt_s_MeV = np.arange(890,4001,1)   #range of s
    sqrt_s = sqrt_s_MeV/1000 #convert from MeV to GeV
    Gamma_VL = selfEnergyModification(sqrt_s)
    Gamma_VT = selfEnergyModification(sqrt_s)
    #Gamma_VL = Gamma_phi
    #Gamma_VT = Gamma_phi
    #Gamma_VL = 0.061
    #Gamma_VT = 0.061
    #Gamma_AL = 0.0549
    Gamma_AL = Gamma_f1_1420*(f_pi/f_pi_0)**2 + Gamma_VL*(1-(f_pi/f_pi_0)**2)
    #Gamma_AT = 0.0549
    #Gamma_AL = 0.02
    #Gamma_AT = 0.02
    #C_gamma = 0.0019
    Gamma_AT = Gamma_AL
    #print("Gamma_A = ", C_gamma*g_gauge**2/1.42**2*(g_gauge**2*f_pi_0**2+2*(1.42**2-1.02**2)/1.42**2))
    #print("Gamma_A_eff = ",Gamma_A_eff)
    #Gamma_AL = Gamma_VL
    #Gamma_AT = Gamma_VT

    ImGv_L = ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,p)
    ImGv_T = ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,p)

    p_0_L = np.sqrt(sqrt_s**2+p**2)
    p_0_T = np.sqrt(sqrt_s**2+p**2)

    #SpectralFx_L = alpha**2/(math.pi*sqrt_s**2)*ImGv_L/(np.exp(p_0_L/temperature)-1)
    #DileptonProductionRate_L = DileptonProductionRate_L+(SpectralFx_L)/(2*p_0_L)
    #SpectralFx_T = alpha**2/(math.pi*sqrt_s**2)*ImGv_T/(np.exp(p_0_T/temperature)-1)
    #DileptonProductionRate_T = DileptonProductionRate_T+(SpectralFx_T)/(2*p_0_T)

    #without thermal dilepton but with combinatorial bkg
    SpectralFx_L_wo_thermal = ImGv_L
    DileptonProductionRate_L_wo_thermal = DileptonProductionRate_L_wo_thermal+(SpectralFx_L_wo_thermal)/(2*p_0_L)/sum(SpectralFx_L_wo_thermal)*momentumDist(p)*decayInsideRate(p)
    SpectralFx_T_wo_thermal = ImGv_T
    DileptonProductionRate_T_wo_thermal = DileptonProductionRate_T_wo_thermal+(SpectralFx_T_wo_thermal)/(2*p_0_T)/sum(SpectralFx_T_wo_thermal)*momentumDist(p)*decayInsideRate(p)
    sqrt_s_MeV = np.arange(890,4001,1)   #range of s
'''
fig, ax = plt.subplots()
#plt.rcParams["font.size"] = 15
ax.plot(sqrt_s,Gamma_VT,label=r"$\Gamma_{\phi}$")
ax.legend()
ax.set_xlabel(r"$\sqrt{s}[{\rm{GeV}}]$")
ax.set_ylabel(r"$\Gamma_{\phi}[{\rm{GeV}}]$")
ax.set_xlim(0.8, 1.1)
plt.grid(True)
plt.show()
#plt.hist(Gamma_VT, bins=100)
#plt.hist(Gamma_AT, bins=100)
'''
#各運動量でのスペクトル関数のプロット
sqrt_s = sqrt_s_MeV/1000 #convert from MeV to GeV
#print(sqrt_s)
Gamma_VL = selfEnergyModification(sqrt_s)
Gamma_VT = selfEnergyModification(sqrt_s)
#Gamma_VL = Gamma_phi
#Gamma_VT = Gamma_phi
#Gamma_VL = 0.061
#Gamma_VT = 0.061
#Gamma_AL = 0.0549
Gamma_AL = Gamma_f1_1420*(f_pi/f_pi_0)**2 + Gamma_VL*(1-(f_pi/f_pi_0)**2)
Gamma_AT = Gamma_AL
fig, ax = plt.subplots()
#plt.rcParams["font.size"] = 15
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,0.5)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,0.5))/3,label="p=0.5GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,1.0)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,1.0))/3,label="p=1.0GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,1.5)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,1.5))/3,label="p=1.5GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,2.0)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,2.0))/3,label="p=2.0GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,2.5)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,2.5))/3,label="p=2.5GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,3.0)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,3.0))/3,label="p=3.0GeV/c")
ax.legend()
ax.set_xlabel(r"$M[{\rm{GeV}}]$")
ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
plt.yscale('log')
ax.set_xlim(0.0, 3.0)
plt.grid(True)
plt.show()

fig, ax = plt.subplots()
#plt.rcParams["font.size"] = 15
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,0.5)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,0.5))/3,label="p=0.5GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,1.0)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,1.0))/3,label="p=1.0GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,1.5)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,1.5))/3,label="p=1.5GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,2.0)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,2.0))/3,label="p=2.0GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,2.5)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,2.5))/3,label="p=2.5GeV/c")
ax.plot(sqrt_s, (ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,3.0)+2*ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,3.0))/3,label="p=3.0GeV/c")
ax.legend()
ax.set_xlabel(r"$M[{\rm{GeV}}]$")
ax.set_ylabel(r"${\rm{Im}}G_\phi[{\rm{GeV}}^{2}]$")
ax.set_xlim(0.0, 3.0)
plt.yscale('log')
plt.grid(True)
plt.show()

#畳み込み積分のための検出器分解能を模したガウスカーネルを生成
massResolution = 0.008 #GeV/c^2
mu = 0
sigma = massResolution
x_convolution = np.linspace(-0.035, 0.035, 70)
gaussian_kernel = gaussian(x_convolution, mu, sigma)

#レプトン対生成率のスピン平均を取る
spin_averaged_DileptonProductionRate = (DileptonProductionRate_L_wo_thermal+2*DileptonProductionRate_T_wo_thermal)/3
spin_averaged_DileptonProductionRate_woMixing = (DileptonProductionRate_L_noMix+2*DileptonProductionRate_T_noMix)/3

#畳み込み積分を実行
#convolved_BWdist = np.convolve(nonRelativisticBWdist(sqrt_s), gaussian_kernel, mode='same')
convolved_Vacdist = np.convolve(DileptonProductionRate_vac, gaussian_kernel, mode='same')
convolved_InMediumdist = np.convolve(spin_averaged_DileptonProductionRate, gaussian_kernel, mode='same')
convolved_InMediumdist_woMixing = np.convolve(spin_averaged_DileptonProductionRate_woMixing, gaussian_kernel, mode='same')

#核内崩壊割合と運動量分布と質量分解能を加味した不変質量分布のプロット
C_0 = 0.00013 #S/N比
C_1 = 10 #統計量E325の何倍か？

# プロット
fig = plt.figure()
# 縦に3分割する
gs = gridspec.GridSpec(3,1)
# そのうち上2つを上のグラフに
ax1 = plt.subplot(gs[:2])
# 残り1つを下のグラフに
ax3 = plt.subplot(gs[2], sharex=ax1) # 上のグラフとx軸のスケールは共通

ax1.plot(sqrt_s, C_1*C_0*DileptonProductionRate_vac, label="Decay at Vacuum")
ax1.plot(sqrt_s, C_1*C_0*spin_averaged_DileptonProductionRate_woMixing,label="Decay in Medium(w/o Chiral Mixing)")
ax1.plot(sqrt_s, C_1*C_0*spin_averaged_DileptonProductionRate,label="Decay in Medium(w/ Chiral Mixing, 30%CSR)")
ax1.plot(sqrt_s, C_1*(C_0*(convolved_InMediumdist_woMixing+convolved_Vacdist)+73165*np.exp(-6.5*sqrt_s)),label="convolved Med+Vac+Bkg(w/o Chiral Mixing)")
ax1.plot(sqrt_s, C_1*(C_0*(convolved_InMediumdist+convolved_Vacdist)+73165*np.exp(-6.5*sqrt_s)),label="convolved Med+Vac+Bkg(w/ Chiral Mixing, 30%CSR)")
ax1.legend(loc='upper center')
ax1.set_xlim(0.9, 3)
ax1.set_ylim(0.0001, 1000000000)
ax1.set_ylabel(r"$dN/ds$")
#ax1.set_ylim(0,100000)
ax1.set_yscale('log')
# 上のグラフのx軸の表記を消去
#plt.rcParams["font.size"] = 15
plt.setp(ax1.get_xticklabels(), visible=False)

# 下のグラフをプロット
Difference1 = C_1*(C_0*(convolved_InMediumdist+convolved_Vacdist)+73165*np.exp(-6.5*sqrt_s))
Difference2 = C_1*(C_0*(convolved_InMediumdist_woMixing+convolved_Vacdist)+73165*np.exp(-6.5*sqrt_s))
Difference = Difference1 - Difference2
ax3.fill_between(sqrt_s, Difference+np.sqrt(np.abs(Difference1))+np.sqrt(np.abs(Difference2)), Difference-np.sqrt(np.abs(Difference1))-np.sqrt(np.abs(Difference2)), alpha=0.2, color='blue')
ax3.plot(sqrt_s, Difference,label="w/ Mixing - w/o Mixing")
ax3.grid(axis="y")
ax3.legend()
ax3.set_xlabel(r"$M[{\rm{GeV}}]$")

# 上下のグラフの隙間をなくす
plt.subplots_adjust(hspace=.0)
plt.show()

# phiの幅の分布
fig, ax = plt.subplots()
#plt.rcParams["font.size"] = 15
ax.plot(Gamma_VT*1000,spin_averaged_DileptonProductionRate,label=r"$\Gamma_{\phi}$")
ax.legend()
ax.set_xlabel(r"$\Gamma_{\phi}[{\rm{MeV}}]$")
ax.set_ylabel(r"$dN/d\Gamma_{\phi}$")
ax.set_xlim(0.0, 200.0)
plt.grid(True)
plt.show()

#dN/dpdr分布の読み込み
def file_open(file):
    import sys
    data = []
    try:
        f = open(file, 'r', encoding='utf-8')
    except Exception:
        print("open error. not found file:", str(file))
        sys.exit(1)
    for line in f:
        line = line.strip() #前後空白削除
        line = line.replace('\n','') #末尾の\nの削除
        line = line.split(",") #分割
        data.append(line)
    f.close()
    return data

data=file_open("../p_rad_RES_2_0xbr_0xshift.txt")
eebkg_data = file_open("../E16SIM/bkg_ee_betaGamma.txt")
epibkg_data = file_open("../E16SIM/bkg_epi_betaGamma.txt")
pipibkg_data = file_open("../E16SIM/bkg_pipi_betaGamma.txt")

momentum=[]
radius=[]
n_phi=[]

eebkg_mass=[]
epibkg_mass=[]
pipibkg_mass=[]
eebkg_betaGamma=[]
epibkg_betaGamma=[]
pipibkg_betaGamma=[]

for i in range(0,len(data)):
    #print("i=",i)
    if len(data[i])==3:
        momentum.append(data[i][0])
        radius.append(data[i][1])
        n_phi.append(data[i][2])

for i in range(0,len(eebkg_data)):
    #print("i=",i)
    if len(eebkg_data[i])==3:
        eebkg_mass.append(eebkg_data[i][1])
        eebkg_betaGamma.append(eebkg_data[i][2])

for i in range(0,len(epibkg_data)):
    #print("i=",i)
    if len(epibkg_data[i])==3:
        epibkg_mass.append(epibkg_data[i][1])
        epibkg_betaGamma.append(epibkg_data[i][2])

for i in range(0,len(pipibkg_data)):
    #print("i=",i)
    if len(pipibkg_data[i])==3:
        pipibkg_mass.append(pipibkg_data[i][1])
        pipibkg_betaGamma.append(pipibkg_data[i][2])

npmomentum=np.array(momentum,dtype=float)
npradius=np.array(radius,dtype=float)
npn_phi=np.array(n_phi,dtype=float)

npeebkg_mass=np.array(eebkg_mass,dtype=float)
npeebkg_betaGamma=np.array(eebkg_betaGamma,dtype=float)
npepibkg_mass=np.array(epibkg_mass,dtype=float)
npepibkg_betaGamma=np.array(epibkg_betaGamma,dtype=float)
nppipibkg_mass=np.array(pipibkg_mass,dtype=float)
nppipibkg_betaGamma=np.array(pipibkg_betaGamma,dtype=float)

print(npeebkg_betaGamma)

#3dグラフを作成
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(npmomentum, npradius, npn_phi, color='blue',s=1,alpha=0.3)
ax.set_xlabel('momentum')
ax.set_ylabel('radius')
ax.set_zlabel('n_phi')
plt.show()

DileptonProductionRate_L_IntPR = [0]*3000
DileptonProductionRate_T_IntPR = [0]*3000
DileptonProductionRate_L_woMixing_IntPR = [0]*3000
DileptonProductionRate_T_woMixing_IntPR = [0]*3000

sqrt_s_MeV = np.arange(1,3001,1)   #range of s
sqrt_s = sqrt_s_MeV/1000 #convert from MeV to GeV

for p_start in range(0,8):
    for deltaP in range(1,9):
        p_end = p_start+deltaP
        if p_end<9:
            print("momentum range: ",p_start," ~ ",p_end)
            ax_ee = plt.subplot()
            n_ee, bins_ee, _ = ax_ee.hist(npeebkg_mass[np.where((p_start/1.02<npeebkg_betaGamma) & (npeebkg_betaGamma<p_end/1.02))]/1000, bins=3000, range=(0,3))
            ax_epi = plt.subplot()
            n_epi, bins_epi, _ = ax_epi.hist(npepibkg_mass[np.where((p_start/1.02<npepibkg_betaGamma) & (npepibkg_betaGamma<p_end/1.02))]/1000, bins=3000, range=(0,3))
            ax_pipi = plt.subplot()
            n_pipi, bins_pipi, _ = ax_pipi.hist(nppipibkg_mass[np.where((p_start/1.02<nppipibkg_betaGamma) & (nppipibkg_betaGamma<p_end/1.02))]/1000, bins=3000, range=(0,3))

            for p_i in range(p_start*20,p_end*20):
                p = p_i*0.05 #0to8GeV, step=0.05GeV
                #sqrt_s_MeV = np.arange(0,3001,1)   #range of s
                #sqrt_s = sqrt_s_MeV/1000 #convert from MeV to GeV
                p_0_L = np.sqrt(sqrt_s**2+p**2)
                p_0_T = np.sqrt(sqrt_s**2+p**2)

                for r_i in range(0,400):
                    r = r_i*0.5 #0to200fm, step=0.5fm
                    density_thisR = woodsSaxonDist(r)
                    c = 0.1*density_thisR #[GeV]
                    f_pi_thisR=(-0.3*density_thisR+1.0)*f_pi_0
                    Gamma_VL = selfEnergyModification_thisDensity(sqrt_s,density_thisR)
                    Gamma_VT = selfEnergyModification_thisDensity(sqrt_s,density_thisR)
                    #Gamma_VL = 0.04
                    #Gamma_VT = 0.04
                    Gamma_AL = Gamma_f1_1420*(f_pi_thisR/f_pi_0)**2 + Gamma_VL*(1-(f_pi_thisR/f_pi_0)**2)
                    Gamma_AT = Gamma_AL
                    ImGv_L = ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,p)
                    ImGv_T = ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,p)
                    ImGv_L=[0 if np.isnan(i) else i for i in ImGv_L]
                    ImGv_T=[0 if np.isnan(i) else i for i in ImGv_T]
                    #print("p = ",p," : r = ",r," : index of n_phi = ",r_i+p_i*400," : n_phi = ",npn_phi[r_i+p_i*400])
                    #print("p = ",p," : r = ",r," : index of n_phi = ",r_i+p_i*400)
                    SpectralFx_L_thisPR = ImGv_L
                    #DileptonProductionRate_L_IntPR = DileptonProductionRate_L_IntPR+(SpectralFx_L_thisPR)/(2*p_0_L)/sum(SpectralFx_L_thisPR)*npn_phi[r_i+p_i*400]
                    DileptonProductionRate_L_IntPR = DileptonProductionRate_L_IntPR+(SpectralFx_L_thisPR)/(2*p_0_L)*npn_phi[r_i+p_i*400]
                    SpectralFx_T_thisPR = ImGv_T
                    #DileptonProductionRate_T_IntPR = DileptonProductionRate_T_IntPR+(SpectralFx_T_thisPR)/(2*p_0_T)/sum(SpectralFx_T_thisPR)*npn_phi[r_i+p_i*400]
                    DileptonProductionRate_T_IntPR = DileptonProductionRate_T_IntPR+(SpectralFx_T_thisPR)/(2*p_0_T)*npn_phi[r_i+p_i*400]
                    c = 0 #[GeV]
                    f_pi_thisR=(-0.3*density_thisR+1.0)*f_pi_0
                    Gamma_VL = selfEnergyModification_thisDensity(sqrt_s,density_thisR)
                    Gamma_VT = selfEnergyModification_thisDensity(sqrt_s,density_thisR)
                    #Gamma_VL = 0.04
                    #Gamma_VT = 0.04
                    Gamma_AL = Gamma_f1_1420*(f_pi_thisR/f_pi_0)**2 + Gamma_VL*(1-(f_pi_thisR/f_pi_0)**2)
                    Gamma_AT = Gamma_AL
                    ImGv_L = ImVector_CurrentCurrentCorrelationFx_L(sqrt_s,p)
                    ImGv_T = ImVector_CurrentCurrentCorrelationFx_T(sqrt_s,p)
                    ImGv_L=[0 if np.isnan(i) else i for i in ImGv_L]
                    ImGv_T=[0 if np.isnan(i) else i for i in ImGv_T]
                    SpectralFx_L_thisPR = ImGv_L
                    #DileptonProductionRate_L_woMixing_IntPR = DileptonProductionRate_L_woMixing_IntPR+(SpectralFx_L_thisPR)/(2*p_0_L)/sum(SpectralFx_L_thisPR)*npn_phi[r_i+p_i*400]
                    DileptonProductionRate_L_woMixing_IntPR = DileptonProductionRate_L_woMixing_IntPR+(SpectralFx_L_thisPR)/(2*p_0_L)*npn_phi[r_i+p_i*400]
                    SpectralFx_T_thisPR = ImGv_T
                    #DileptonProductionRate_T_woMixing_IntPR = DileptonProductionRate_T_woMixing_IntPR+(SpectralFx_T_thisPR)/(2*p_0_T)/sum(SpectralFx_T_thisPR)*npn_phi[r_i+p_i*400]
                    DileptonProductionRate_T_woMixing_IntPR = DileptonProductionRate_T_woMixing_IntPR+(SpectralFx_T_thisPR)/(2*p_0_T)*npn_phi[r_i+p_i*400]

            #print(DileptonProductionRate_T_IntPR)
            fig, ax = plt.subplots()
            #plt.rcParams["font.size"] = 15
            ax.plot(sqrt_s, DileptonProductionRate_L_IntPR,label="Longitudinal")
            ax.plot(sqrt_s, DileptonProductionRate_T_IntPR,label="Transverse")
            ax.plot(sqrt_s, (DileptonProductionRate_L_IntPR+2*DileptonProductionRate_T_IntPR)/3,label="spin average")
            ax.legend()
            ax.set_xlabel(r"$M[{\rm{GeV}}]$")
            ax.set_ylabel(r"$dN/ds$")
            ax.set_xlim(0.0, 3.0)
            plt.yscale('log')
            plt.grid(True)
            plt.show()
            file_name = str(p_start)+"-"+str(p_end)+"_Dilepton"
            plt.savefig("file_name")
            plt.close()
            
            fig, ax = plt.subplots()
            #plt.rcParams["font.size"] = 15
            ax.plot(sqrt_s, n_ee,label="ee Bkg")
            ax.legend()
            ax.set_xlabel(r"$M[{\rm{GeV}}]$")
            ax.set_ylabel(r"$dN/ds$")
            ax.set_xlim(0.0, 3.0)
            #plt.yscale('log')
            plt.grid(True)
            plt.show()
            file_name = str(p_start)+"-"+str(p_end)+"_ee_bkg"
            plt.savefig("file_name")
            plt.close()

            fig, ax = plt.subplots()
            #plt.rcParams["font.size"] = 15
            ax.plot(sqrt_s, n_epi,label="epi Bkg")
            ax.legend()
            ax.set_xlabel(r"$M[{\rm{GeV}}]$")
            ax.set_ylabel(r"$dN/ds$")
            ax.set_xlim(0.0, 3.0)
            #plt.yscale('log')
            plt.grid(True)
            plt.show()
            file_name = str(p_start)+"-"+str(p_end)+"_epi_bkg"
            plt.savefig("file_name")
            plt.close()

            fig, ax = plt.subplots()
            #plt.rcParams["font.size"] = 15
            ax.plot(sqrt_s, n_pipi,label="pipi Bkg")
            ax.legend()
            ax.set_xlabel(r"$M[{\rm{GeV}}]$")
            ax.set_ylabel(r"$dN/ds$")
            ax.set_xlim(0.0, 3.0)
            #plt.yscale('log')
            plt.grid(True)
            plt.show()
            file_name = str(p_start)+"-"+str(p_end)+"_pipi_bkg"
            plt.savefig("file_name")
            plt.close()
            
            #畳み込み積分のための検出器分解能を模したガウスカーネルを生成
            massResolution = 0.008 #GeV/c^2
            mu = 0
            sigma = massResolution
            x_convolution = np.linspace(-0.035, 0.035, 70)
            gaussian_kernel = gaussian(x_convolution, mu, sigma)

            #レプトン対生成率のスピン平均を取る
            C_2 = 0.1/3
            #C_3 = 0.055
            C_3 = 1
            spin_averaged_DileptonProductionRate_IntPR = C_3*(DileptonProductionRate_L_IntPR+2*DileptonProductionRate_T_IntPR)/3+C_2*(n_ee+n_epi+n_pipi)
            spin_averaged_DileptonProductionRate_woMixing_IntPR = C_3*(DileptonProductionRate_L_woMixing_IntPR+2*DileptonProductionRate_T_woMixing_IntPR)/3+C_2*(n_ee+n_epi+n_pipi)
            #spin_averaged_DileptonProductionRate_IntPR = (DileptonProductionRate_L_IntPR+2*DileptonProductionRate_T_IntPR)/3
            #spin_averaged_DileptonProductionRate_woMixing_IntPR = (DileptonProductionRate_L_woMixing_IntPR+2*DileptonProductionRate_T_woMixing_IntPR)/3


            #畳み込み積分を実行
            convolved_InMediumdist_IntPR = np.convolve(spin_averaged_DileptonProductionRate_IntPR, gaussian_kernel, mode='same')
            convolved_InMediumdist_woMixing_IntPR = np.convolve(spin_averaged_DileptonProductionRate_woMixing_IntPR, gaussian_kernel, mode='same')

            #核内崩壊割合と運動量分布と質量分解能を加味した不変質量分布のプロット
            C_0 = 1e-5*6.5/3
            C_1 = 100 #統計量E325の何倍か？

            # プロット
            fig = plt.figure()
            # 縦に3分割する
            gs = gridspec.GridSpec(3,1)
            # そのうち上2つを上のグラフに
            ax1 = plt.subplot(gs[:2])
            # 残り1つを下のグラフに
            ax3 = plt.subplot(gs[2], sharex=ax1) # 上のグラフとx軸のスケールは共通
            #ax1.plot(sqrt_s, C_1*(C_0*(convolved_InMediumdist_woMixing_IntPR)+73165*np.exp(-6.5*sqrt_s)),label="convolved Med+Vac+Bkg(w/o Chiral Mixing)")
            #ax1.plot(sqrt_s, C_1*(C_0*(convolved_InMediumdist_IntPR)+73165*np.exp(-6.5*sqrt_s)),label="convolved Med+Vac+Bkg(w/ Chiral Mixing, 30%CSR)")
            ax1.plot(sqrt_s, C_1*C_0*convolved_InMediumdist_woMixing_IntPR,label="convolved Med+Vac+Bkg(w/o Chiral Mixing)")
            ax1.plot(sqrt_s, C_1*C_0*convolved_InMediumdist_IntPR,label="convolved Med+Vac+Bkg(w/ Chiral Mixing, 30%CSR)")
            ax1.legend(loc='lower center')
            ax1.set_xlim(0.9, 3)
            #ax1.set_ylim(0.0001, 1000000000)
            ax1.set_ylabel(r"$dN/ds$")
            #ax1.set_ylim(0,100000)
            ax1.set_yscale('log')
            # 上のグラフのx軸の表記を消去
            #plt.rcParams["font.size"] = 15
            plt.setp(ax1.get_xticklabels(), visible=False)

            # 下のグラフをプロット
            Difference1 = C_1*(C_0*(convolved_InMediumdist_IntPR))
            Difference2 = C_1*(C_0*(convolved_InMediumdist_woMixing_IntPR))
            Difference = Difference1 - Difference2
            ax3.fill_between(sqrt_s, Difference+np.sqrt(np.abs(Difference1))+np.sqrt(np.abs(Difference2)), Difference-np.sqrt(np.abs(Difference1))-np.sqrt(np.abs(Difference2)), alpha=0.2, color='blue')
            ax3.plot(sqrt_s, Difference,label="w/ Mixing - w/o Mixing")
            ax3.grid(axis="y")
            ax3.legend()
            ax3.set_xlabel(r"$M[{\rm{GeV}}]$")

            # 上下のグラフの隙間をなくす
            plt.subplots_adjust(hspace=.0)
            plt.show()
            file_name = str(p_start)+"-"+str(p_end)+"_InvMass"
            plt.savefig("file_name")
            plt.close()

            print("Momentum range = ",p_start," ~ ",p_end," : index = ",1020+np.argmax(Difference[1020:])," : sigma = ",Difference[1020+np.argmax(Difference[1020:])]/(np.sqrt(np.abs(Difference1[1020+np.argmax(Difference[1020:])]))+np.sqrt(np.abs(Difference2[1020+np.argmax(Difference[1020:])]))))