# %% Import modules
from diffv4 import StepDiffusion
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import pi



"""

So – does this make sense? Boundary conditions are F = 200, Cl = 5000, Br = 50, exterior = 10 for all; Start T = 1500 C, cooling rate = 10 deg/min, total time 3600 minutes. It would be great to have this calculation. I’m very curious about how it will affect the profiles of halogen ratios (= Fig. 7c).


"""
F_initial = 200
Cl_initial = 5000
Br_initial = 50

figsize = (5,3)
figsize1by2 = (6,8)
mpl.rcParams.update({'font.size': 8})
# %% Create diffusion models for F,Cl,Br and I and save output to DataFrames
F_model = StepDiffusion(
                        R=5000e-6,
                        R_steps=101,
                        element='F',
                        Tc=1500,
                        t_steps='Auto',
                        t_max=3600*60,
                        C0=200,
                        Cout=10,
                        plot=10,
                        delT=-600,
                        legend=False,
                        sphericFactor=True,
                        plotDetectionLimit=False
                        )

F_Ci = F_model.runModel()

Cl_model = StepDiffusion(
                        R=5000e-6,
                        R_steps=101,
                        element='Cl',
                        Tc=1500,
                        t_steps='Auto',
                        t_max=3600*60,
                        C0=5000,
                        Cout=10,
                        plot=10,
                        delT=-600,
                        legend=False,
                        sphericFactor=True,
                        plotDetectionLimit=False
                        )

Cl_Ci = Cl_model.runModel()

Br_model = StepDiffusion(
                        R=5000e-6,
                        R_steps=101,
                        element='Br',
                        Tc=1500,
                        t_steps='Auto',
                        t_max=3600*60,
                        C0=50,
                        Cout=10,
                        plot=10,
                        delT=-600,
                        legend=False,
                        sphericFactor=True,
                        plotDetectionLimit=False
                        )

Br_Ci = Br_model.runModel()

# %% plot diffusion related fractionation

F_comp = F_Ci[0]
Cl_comp = Cl_Ci[0]
Br_comp = Br_Ci[0]

F_comp_final = F_comp[-1,:-1]
Cl_comp_final = Cl_comp[-1,:-1]
Br_comp_final = Br_comp[-1,:-1]

# %% plot F/Cl and Br/Cl ratios

ratio_fig, r_axes = plt.subplots(2,1,figsize=figsize1by2,sharex = True,constrained_layout=True)

x = np.linspace(0,500,14)

r_axes[0].plot(F_comp_final,label='F')
r_axes[0].plot(Cl_comp_final,label='Cl')
r_axes[0].plot(Br_comp_final,label='Br')
r_axes[0].axhline(y=10,linestyle='--',color='k',label='Baseline')
r_axes[0].set_ylabel('Concentraiton after 15 min ($\u03BCg.g^{-1}$)')
r_axes[0].legend()

r_axes[1].plot(F_comp_final/Cl_comp_final,label='F/Cl')
r_axes[1].plot(Br_comp_final/Cl_comp_final,label='Br/Cl',color='C2')
r_axes[1].set_ylabel('Ratio')
r_axes[1].set_xlabel('Distance from Centre ($\u03BCm$)')
r_axes[1].legend()

# ratio_fig.savefig(str('G:\My Drive\whymanchesterwhy\HalogensInEnstatiteChondrites\Figs'+'\F_Cl_Br_ratios.png'),dpi=300)

# %%
x = np.linspace(0,5000,100)
fig,ax=plt.subplots(figsize=figsize)
ax.plot(x,(F_comp_final/F_initial)/(Cl_comp_final/Cl_initial),label='F/Cl')
ax.plot(x,(Br_comp_final/Br_initial)/(Cl_comp_final/Cl_initial),label='Br/Cl',color='C2')
ax.set_ylabel('Ratio (Initial Normalised)')
ax.set_xlabel('Distance from Centre ($\u03BCm$)')
ax.set_yscale('log')
ax.set_ylim(0.5,50)
ax.axhline(y=1,linestyle='--',color='k')
ax.legend()
# %%
