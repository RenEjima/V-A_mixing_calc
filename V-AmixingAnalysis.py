import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import curve_fit

#constant
rho_0 = 1.0
#GammaVac = 4.3 #MeV 4.3
GammaVac = 40.0
massVac = 1020 #MeV
A_target = 63 #Cu target
#A_target = 206 #Pb

#radius of Atom
r_A = 1.2*pow(A_target,1/3) #[fm]
print("r_A=",r_A)

#1. montecalro simulation to investigate
#the momentum dependence of the ratio of phi meson decay inside.

#woods saxon distribution
def woodsSaxonDist(r):
    return 1/(1+np.exp((r-r_A)/0.5))
#phi meson generation
def randomgen_in_r_direction():
    while(1):
        x_rand, y_rand = np.random.random(2)
        x = r_A*x_rand
        if (y_rand <= woodsSaxonDist(x) ):
            return x
            break

# generate random number list
r_generated_list = []
phi_generated_list = []
theta_generated_list = []
for i in range(100000):
    r_generated_list.append(randomgen_in_r_direction())
n_phi = len(r_generated_list)
for j in range(n_phi):
    phi_generated_list.append(random.uniform(0, math.pi*2))
    theta_generated_list.append(random.uniform(0, math.pi))

# plot result
ax = plt.subplot()
ax.hist(r_generated_list, range=[0,r_A+1], bins=300,  color='green', alpha=0.5)
ax.set_xlabel("r_generated[fm]")
plt.show()
'''
ax = plt.subplot()
ax.hist(phi_generated_list, range=[0,math.pi*2], bins=300,  color='red', alpha=0.5)
ax.set_xlabel("phi_generated[rad]")
plt.show()
ax = plt.subplot()
ax.hist(theta_generated_list, range=[0,math.pi], bins=300,  color='red', alpha=0.5)
ax.set_xlabel("theta_generated[rad]")
plt.show()
'''

def selfEnergyModification(s):
    mKstar = 0.51
    mKbarstar = 0.38
    ks = np.sqrt((s - (mKstar+mKbarstar)**2)*(s - (mKstar-mKbarstar)**2))/(2*np.sqrt(s))
    Gamma_phiS=((1.69*4/3)*ks**3)/s
    #print("modified Gamma = ", np.median(Gamma_phiS*1000))
    return Gamma_phiS

def sDist(s):
    return (0.04/(2*math.pi*(pow(np.sqrt(s)-massVac,2)+0.04**2/4)))

#random pT
def randomgen_s():
    while(1):
        x_rand, y_rand = np.random.random(2)
        x = 3*x_rand
        #print("random num = ",y_rand*1e-8," dist = ",sDist(x))
        if (y_rand*1e-9 <= sDist(x) ):
            return x
            break

s_generated_list = []
while len(s_generated_list)<n_phi:
    s_generated_list.append(randomgen_s())
ax = plt.subplot()
ax.hist(np.sqrt(s_generated_list), range=[0,10], bins=300,  color='red', alpha=0.5)
ax.set_xlabel("sqrt(s)[GeV]")
plt.show()
#momentum distribution
#pT distribution
def PtDist(p):
    p_0 = 48
    p_1 = 0.62
    p_2 = 5.0
    p_3 = 0.74
    p_4 = 14
    return p_0*p/pow((p_1+pow(p/p_2,p_3)),p_4)

#random pT
def randomgen_Pt():
    while(1):
        x_rand, y_rand = np.random.random(2)
        x = 10*x_rand
        if (y_rand*300 <= PtDist(x) ):
            return x
            break

pT_generated_list = []
while len(pT_generated_list)<n_phi:
    pT_generated_list.append(randomgen_Pt())
ax = plt.subplot()
ax.hist(pT_generated_list, range=[0,10], bins=300,  color='red', alpha=0.5)
ax.set_xlabel("pT[GeV]")
plt.show()

#random pZ
#pZ分布がどんな形してるかわからないのでとりあえずpT分布と同じ分布を過程


'''
def PzDist(p):
    Pz =
    return Pz

#random pZ
def randomgen_Pz():
    while(1):
        x_rand, y_rand = np.random.random(2)
        x = 100*x_rand #何GeVくらいまでいるのか...
        if (y_rand*300 <= PzDist(x) ):
            return x

pZ_generated_list = []
while len(pZ_generated_list)<n_phi:
    pZ_generated_list.append(randomgen_Pz())
ax.hist(pZ_generated_list, range=[0,10], bins=300,  color='red', alpha=0.5)
plt.show()
'''

pZ_generated_list = []
pZ_generated_list = pT_generated_list

#decay inside or outside
#life time distribution
lifeTime_generated_list = []
def lifetime_dist(t):
    tau = np.exp(-t*GammaVac/1000)
    return tau

def randomgen_lifeTime():
    while(1):
        x_rand, y_rand = np.random.random(2)
        x = 50000*x_rand
        if (y_rand <= lifetime_dist(x) ):
            return x
tau_generated_list = []
while len(tau_generated_list)<n_phi:
    tau_generated_list.append(randomgen_lifeTime())

lorentz_gamma_phi = []
lifeTime_generated_list = []
for k in range(n_phi):
    lorentz_gamma_phi.append(np.sqrt(1+np.sqrt(pT_generated_list[k]**2+pZ_generated_list[k]**2)**2/((massVac/1000)**2)))
    lifeTime_generated_list.append(tau_generated_list[k]*lorentz_gamma_phi[k])
ax = plt.subplot()
ax.hist(tau_generated_list, range=[0,2000], bins=300,  color='red', alpha=0.5)
ax.set_xlabel("tau")
plt.show()
ax = plt.subplot()
ax.hist(lifeTime_generated_list, range=[0,200], bins=300,  color='red', alpha=0.5)
ax.set_xlabel("life time[fm/c]")
plt.show()

#phi's decay point (r-direction)
def yogenTeiri(b,c,theta):
    return math.sqrt(b**2+c**2-2*b*c*math.cos(theta))
momentum_list = []
momentum_decayInside = []
momentum_decayOutside = []
r_decay_list = []
for l in range (n_phi):
    r_decay_phi = yogenTeiri(np.sqrt(pT_generated_list[l]**2+pZ_generated_list[l]**2)/(lorentz_gamma_phi[l]*massVac/1000)*(lifeTime_generated_list[l]),r_generated_list[l],math.pi*np.random.rand())
    momentum_list.append(np.sqrt(pT_generated_list[l]**2+pZ_generated_list[l]**2))
    r_decay_list.append(r_decay_phi)
    #print("r_decay = ",r_decay_phi," r_A = ",r_A)
    if r_decay_phi<r_A:
        momentum_decayInside.append(np.sqrt(pT_generated_list[l]**2+pZ_generated_list[l]**2))
    else:
        momentum_decayOutside.append(np.sqrt(pT_generated_list[l]**2+pZ_generated_list[l]**2))
print("N_phi = ",n_phi," N_inside = ",len(momentum_decayInside))
ax = plt.subplot()
ax.hist(momentum_list, range=[0,10], bins=300,  color='red', alpha=0.5)
ax.set_xlabel("all "+r"$\phi$"+"'s p[GeV/c]")
plt.show()

print("r_A = ", r_A)
print("r_decay_list")
print(r_decay_list)
ax = plt.subplot()
ax.hist(r_decay_list, range=[0,r_A+10], bins=300, color='green', alpha=0.5)
ax.set_xlabel("decay point[fm]")
plt.show()

r_decay_hist, r_decay_bins = np.histogram(r_decay_list, bins=300)
plt.step(r_decay_bins[:-1], r_decay_hist, where='post')
plt.show()

ax = plt.subplot()
ax.hist(momentum_decayInside, range=[0,10], bins=300,  color='red', alpha=0.5)
ax.set_xlabel(r"$\phi$"+" decay inside p[GeV/c]")
plt.show()

ax = plt.subplot()
decayInsideOnThisBin, bins, patches = ax.hist(momentum_decayInside, bins=300, range=(0, 3))
plt.show()
ax = plt.subplot()
momentumOnThisBin, bins, patches = ax.hist(momentum_list, bins=300, range=(0, 3))
plt.show()
print("len(inside) = ",len(decayInsideOnThisBin)," len(all) = ",len(momentumOnThisBin))
decayInsideRateList = []
for m in range(300):
    #print("Inside = ",decayInsideOnThisBin[m]," all = ",momentumOnThisBin[m]," ratio = ",decayInsideOnThisBin[m]/momentumOnThisBin[m])
    decayInsideRateList.append(decayInsideOnThisBin[m]/momentumOnThisBin[m])
decayInsideRateListCu = decayInsideRateList

fig, ax = plt.subplots()
p_phi = np.arange(0,3,0.01)
ax.plot(p_phi, decayInsideRateList,label="Cu target")
ax.set_xlabel(r"p_\phi[{\rm{GeV}}]")
ax.set_xlim(0,3)
ax.set_ylim(0,1)
ax.legend()
ax.set_ylabel("decay inside ratio")
plt.show()
def fitting_function(x,a,b,c,d,e,f):
    return c*a**(b*x)+f*d**(e*x)
def fit(func, data, param_init):
    """
    func:データxに近似したい任意の関数
    data:データ
    param_init:パラメータの初期値
    popｔ:最適化されたパラメータ
    pocv:パラメータの共分散
    """
    X = p_phi
    Y = data
    popt,pocv=curve_fit(func, X, Y, p0=param_init)
    perr = np.sqrt(np.diag(pocv)) #対角成分が各パラメータの標準誤差に相当
    y=func(X, *popt)
    print(popt)
    return y, popt, perr
paramList = [2,-3,1,2,-0.5,1]
result = fit(fitting_function, decayInsideRateList, paramList)
fig, ax = plt.subplots()
ax.plot(p_phi, decayInsideRateList,label="Cu target")
ax.plot(p_phi, result[0])
ax.set_xlabel(r"$p_\phi[{\rm{GeV}}]$")
ax.set_xlim(0,3)
ax.set_ylim(0,1)
ax.legend()
ax.set_ylabel("decay inside ratio")
plt.show()