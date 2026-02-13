# %% Import modules
from diffv4 import CoupledModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import pi


# %% Create diffusion models for F,Cl,Br and I and save output to DataFrames
# plotting settings
figsize = (8/2.54,6/2.54)
figsize1by2 = (8/2.54,10/2.54)
mpl.rcParams.update({'font.size': 8})
#constrained_layout=True # to be put into plt.subplots()

# %% Create a model for F
Fmodel = CoupledModel(x1=0.0001,x2=500*pi,Bi=1,
              R_steps=15,R=500e-6,
              C0=200,Cout=10,
              Ti=1500+273,Tout=900+273,
              alpha=0.3e-6,element='F',
              t_steps='Auto',t_max=1*1e3,
              plot=10,legend=True,plotDetectionLimit=True,
              sphericFactor=True)

F_Ci =  Fmodel.runModel()

# %% Create a model for Cl

Clmodel = CoupledModel(x1=0.0001,x2=500*pi,Bi=1,
                R_steps=15,R=500e-6,
                C0=5000,Cout=10,
                Ti=1500+273,Tout=900+273,
                alpha=0.3e-6,element='Cl',
                t_steps='Auto',t_max=1*1e3,
                plot=10,legend=False,plotDetectionLimit=True,
                sphericFactor=True)

Cl_Ci =  Clmodel.runModel()

# %% Create a model for Br

Brmodel = CoupledModel(x1=0.0001,x2=500*pi,Bi=1,
                R_steps=15,R=500e-6,
                C0=50,Cout=10,
                Ti=1500+273,Tout=900+273,
                alpha=0.3e-6,element='Br',
                t_steps='Auto',t_max=1*1e3,
                plot=10,legend=True,plotDetectionLimit=True,
                sphericFactor=True)

Br_Ci =  Brmodel.runModel()

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

r_axes[0].plot(x,F_comp_final,label='F')
r_axes[0].plot(x,Cl_comp_final,label='Cl')
r_axes[0].plot(x,Br_comp_final,label='Br')
r_axes[0].axhline(y=10,linestyle='--',color='k',label='Baseline')
r_axes[0].set_ylabel('Concentraiton after 15 min ($\u03BCg.g^{-1}$)')
r_axes[0].legend()

r_axes[1].plot(x,F_comp_final/Cl_comp_final,label='F/Cl')
r_axes[1].plot(x,Br_comp_final/Cl_comp_final,label='Br/Cl',color='C2')
r_axes[1].set_ylabel('Ratio')
r_axes[1].set_xlabel('Distance from Centre ($\u03BCm$)')
r_axes[1].legend()

# ratio_fig.savefig(str('G:\My Drive\whymanchesterwhy\HalogensInEnstatiteChondrites\Figs'+'\F_Cl_Br_ratios.png'),dpi=300)


# %%
