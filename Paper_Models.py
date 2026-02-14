# %% Import modules
from Diffusion_Model import StepDiffusion
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import pi


F_initial = 200
Cl_initial = 5000
Br_initial = 50
C_out = 10
chondrule_radius = 500e-6
r_steps = 101
T_initial = 1500 # C
T_change = -600 # 10 deg/min cooling rate, so -600 K over 3600 minutes
t_max = 60*60 # 3600 minutes in seconds

figsize = (5,3)
figsize1by2 = (6,8)
mpl.rcParams.update({'font.size': 8})
# %% Create diffusion models for F,Cl,Br and I and save output to DataFrames
F_model = StepDiffusion(
                        R=chondrule_radius,
                        R_steps=r_steps,
                        element='F',
                        Tc=T_initial,
                        t_steps='Auto',
                        t_max=t_max,
                        C0=F_initial,
                        Cout=C_out,
                        plot=10,
                        delT=T_change,
                        legend=False,
                        sphericFactor=True,
                        plotDetectionLimit=False
                        )

F_Ci = F_model.runModel()

Cl_model = StepDiffusion(
                        R=chondrule_radius,
                        R_steps=r_steps,
                        element='Cl',
                        Tc=T_initial,
                        t_steps='Auto',
                        t_max=t_max,
                        C0=Cl_initial,
                        Cout=C_out,
                        plot=10,
                        delT=T_change,
                        legend=False,
                        sphericFactor=True,
                        plotDetectionLimit=False
                        )

Cl_Ci = Cl_model.runModel()

Br_model = StepDiffusion(
                        R=chondrule_radius,
                        R_steps=r_steps,
                        element='Br',
                        Tc=T_initial,
                        t_steps='Auto',
                        t_max=t_max,
                        C0=Br_initial,
                        Cout=C_out,
                        plot=10,
                        delT=T_change,
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

ratio_fig, r_axes = plt.subplots(2,1,figsize=figsize1by2,constrained_layout=True)

x = np.linspace(0,chondrule_radius*1e6,r_steps)

r_axes[0].plot(F_comp_final,label='F')
r_axes[0].plot(Cl_comp_final,label='Cl')
r_axes[0].plot(Br_comp_final,label='Br')
r_axes[0].axhline(y=10,linestyle='--',color='k',label='Baseline')
r_axes[0].set_ylabel('Concentraiton after 60 min ($\u03BCg.g^{-1}$)')
r_axes[0].set_xlabel(f'Distance from Centre ($\u03BCm$)')

r_axes[0].legend()
r_axes[1].plot(F_comp_final/Cl_comp_final,label='F/Cl')
r_axes[1].plot(Br_comp_final/Cl_comp_final,label='Br/Cl',color='C2')
r_axes[1].set_ylabel('Ratio')
r_axes[1].set_xlabel(f'Distance from Centre ($\u03BCm$)')
r_axes[1].set_yscale('log')
r_axes[1].legend()

# ratio_fig.savefig(str('G:\My Drive\whymanchesterwhy\HalogensInEnstatiteChondrites\Figs'+'\F_Cl_Br_ratios.png'),dpi=300)

# %%
x = np.linspace(0,chondrule_radius*1e6,100)
fig,ax=plt.subplots(figsize=figsize)
ax.plot(x,(F_comp_final/F_initial)/(Cl_comp_final/Cl_initial),label='F/Cl')
ax.plot(x,(Br_comp_final/Br_initial)/(Cl_comp_final/Cl_initial),label='Br/Cl',color='C2')
ax.set_ylabel('Ratio (Initial Normalised)')
ax.set_xlabel(f'Distance from Centre ($\u03BCm$)')
ax.set_yscale('log')
#ax.set_ylim(0.5,50)
ax.axhline(y=1,linestyle='--',color='k')
ax.legend()
# %%
