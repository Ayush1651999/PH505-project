import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy
# import mpmath

# define the parametrs used

A1 = 60
A2 = 4
A3 = 64
reduced_mass = A1*A2/(A1+A2)

Z1 = 28
Z2 = 2
Z3 = 30

pi = np.pi
e = np.sqrt(1.44) # sqrt(MeV-fm)
b_surf = 17 # MeV
h = 197.32 # Mev-fm
hcut = 1 # h/(2*pi)



# Defining function to solve with the Newton raphson method

# source for this code : 
# https://stackoverflow.com/questions/42449242/newton-raphsons-method-user-input-and-numerical-output-problems

def f(symx):
    tmp = sp.sympify(symx)
    return tmp

def fprime(symx):
    tmp = sp.diff(f(symx), r)
    return tmp;

def newtons_method(symx):   
    guess = sp.sympify(0.5) # Convert to an int immediately.
    div = f(symx)/fprime(symx)

    for i in range(1, 100):
        nextGuess = guess - div.subs(r, guess)
        guess = nextGuess
    return guess.evalf()

# Code for part 3

r = sp.Symbol('r')

V_coulomb = Z1*Z2*(e**2)/r   # MeV

R1 = (1.128)*(A1**(1/3))*(1 - 0.786*(A1**(-2/3))) # fm
R2 = (1.128)*(A2**(1/3))*(1 - 0.786*(A2**(-2/3))) # fm
V_0 = b_surf*(A1**(2/3) + A2**(2/3) - (A1+A2)**(2/3))
a = (0.356)*((R1+R2)*(A1**(2/3) + A2**(2/3) - (A1+A2)**(2/3)))/(R1*R2)

V_nuclear = -V_0/(1+sp.exp((r-R1-R2)/a))

V_net = V_coulomb + V_nuclear    # net potential is coulomb + nuclear potential

dV_dr = sp.diff(V_net)     # derivative of net potential

R_l = newtons_method(dV_dr)     
# r at which derivative of net potential vanishes


d2V_dr2 = sp.diff(dV_dr)     
# double derivative of net potential w.r.t r

omega = ((d2V_dr2.subs(r, R_l))/(reduced_mass))**(0.5)    
# evaluating omega, by substituting R_l in the double derivative


r_B = 1.4  # units = fm
r_C = 1    # units = fm

R_B = r_B*((A1**(1/3)) + (A2**(1/3)))
R_C = r_C*((A1**(1/3)) + (A2**(1/3)))

V_B = V_net.subs(r, R_B)
V_C = V_net.subs(r, R_C)


E = sp.Symbol('E')

hcut_omega = hcut*omega

num = 1 + sp.exp(2*pi*(E-V_B)/hcut_omega)
den = 1 + sp.exp(2*pi*(E - V_B - ((R_C/R_B)**2)*(E - V_C))/hcut_omega)
sigma_fusion = (0.5)*(hcut_omega)*(R_B**2)*(1/E)*(sp.log(num/den))   
# Final Glas-Mosel equation


E_values = np.linspace(10,38)    
# range of energy values considered in the paper

sigma_fusion_values = np.zeros(50)

for i in range(50):
    sigma_fusion_values[i] = sigma_fusion.subs(E, E_values[i])
#calculating sigma_fusion for all the energy values


    
# Code for part 2 starts here

reduced_masses = np.array([0.984375, 0.96875, 0.96875])
#reduced mass of all the exit channels

spin_multiplicity = np.array([2, 4, 4])
# spin multiplicities of the all the exit channel partciles 

Q_values = np.array([-7.906, -17.022, -14.621])    # in MeV
# Q values of the exit channels

Separation_energy = np.array([11.861, 20.978, 18.576])    # in MeV
# Separation energy of the outgoing particles

T = 1.255 # MeV, 
# nuclear temperature for the compound nucleus Zn-64

E_excitation = np.zeros([50,3]) 
# number of incident energy values vs no of exit channels

for i in range(50):
    for j in range(3):
        E_excitation[i][j] = Q_values[j] + (E_values[i])*(15/16)
# Calculating excitation energies for all the values
        
a_values = (E_excitation+T)/(T**2)
# calculating the level density parameter

threshold = np.array([0, 0, -9])  # MeV
# threshold for charged particle emission
# we have approximated with the coulomb barrier

R_values = E_excitation - Separation_energy - threshold
# maximum exciation energy for the residual nucleus

tau_values = np.zeros_like(E_excitation)

for i in range(tau_values.shape[0]):
    for j in range(tau_values.shape[1]):
        ax_Rx = (R_values[i][j])*(a_values[i][j])
        ax_Rx = np.abs(ax_Rx)
        tau_values[i][j] = (spin_multiplicity[j])*\
        (reduced_masses[j])*(ax_Rx)*(np.exp(2*((ax_Rx)**0.5)))
        
        
for i in range(tau_values.shape[0]):
    tau = tau_values[i][0]+tau_values[i][1]+tau_values[i][2]
    for j in range(tau_values.shape[1]):
        tau_values[i][j] = tau_values[i][j]/(tau)
# tau_values containes the probabilities of all the residual reactions

residual_cross_sections = np.zeros_like(tau_values)

for i in range(residual_cross_sections.shape[0]):
    for j in range(residual_cross_sections.shape[1]):
        residual_cross_sections[i][j] = sigma_fusion_values[i]\
        *tau_values[i][j]
# residual cross sections represent the cross sections
# of residual reactions w.r.t energy values 


# Plotting part

plt.plot(E_values, sigma_fusion_values, label='fusion_cross_section')
plt.plot(E_values, residual_cross_sections[:,0], label='first_cross_section')
plt.plot(E_values, residual_cross_sections[:,1], label='second_cross_section')
plt.plot(E_values, residual_cross_sections[:,2], label='third_cross_section')

plt.xlabel('Energy in MeV')
plt.ylabel('Crosss section in $fm^2$')

plt.legend(loc='center right', bbox_to_anchor=(0.70, -0.35))
plt.show()