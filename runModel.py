# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:57:15 2022

@author: r11403eb
"""

#%%

import diffv2
import numpy as np
import matplotlib.pyplot as plt
t_steps = 100000+1
r = 500e-6
r_steps = 100
tmax = 60*30
delT = -300

starting_comp = {
    'F': 200,
    'Cl': 5000,
    'Br': 50,
}

F = diffv2.StepDiffusion(R=r,t_steps=t_steps,t_max=tmax,R_steps=r_steps,
                        element='F',Tc=1500,C0=starting_comp['F'],Cout=10,plot=10,
                        delT=delT,legend=False,plotDetectionLimit=False)
print(str('Min temperature reached = '+str(F.Tc+F.delT)+' \N{DEGREE SIGN}C'))
Fdata,_,Ffig,Fax,Fj = F.runModel()
Fax.set_ylim(0,None)
Fax.set_xlim(0,None)
Ffig.savefig('F.png')
np.savetxt('Fdata.txt',Fdata)

Cl = diffv2.StepDiffusion(R=r,t_steps=t_steps,t_max=tmax,R_steps=r_steps,
                            element='Cl',Tc=1500,C0=starting_comp['Cl'],Cout=10,plot=10,
                            delT=delT,legend=False,plotDetectionLimit=False)
print(str('Min temperature reached = '+str(Cl.Tc+Cl.delT)+' \N{DEGREE SIGN}C'))
Cldata,_,Clfig,Clax,Cj = Cl.runModel()
Clax.set_ylim(0,None)
Clax.set_xlim(0,None)
Clfig.savefig('Cl.png')
np.savetxt('Cldata.txt',Cldata)
    
Br = diffv2.StepDiffusion(R=r,t_steps=t_steps,t_max=tmax,R_steps=r_steps,
                        element='Br',Tc=1500,C0=starting_comp['Br'],Cout=10,plot=10,
                        delT=delT,legend=True,plotDetectionLimit=False)
print(str('Min temperature reached = '+str(Br.Tc+Br.delT)+' \N{DEGREE SIGN}C'))
Brdata,_,Brfig,Brax,Bj = Br.runModel()
Brax.set_ylim(0,None)
Brax.set_xlim(0,None)
Brfig.savefig('Br.png')
np.savetxt('Brdata.txt',Brdata)

#%% ratioplot

F_final = Fdata[-1,:]
Cl_final = Cldata[-1,:]
Br_final = Brdata[-1,:]

FCl = F_final/Cl_final
BrCl = Br_final/Cl_final

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(np.linspace(0,1e6*r,r_steps),FCl,label='F/Cl')
ax.plot(np.linspace(0,1e6*r,r_steps),BrCl,label='Br/Cl')
ax.legend()
ax.axhline(y=1, color='k', linestyle='--')
#ax.set_ylim(0.5,2.5)
ax.set_xlim(0,None)
ax.set_xlabel('Radius ($\\mu$m)')
ax.set_ylabel('Relative Concentration')

fig, ax = plt.subplots(figsize=(5,4), layout='constrained')
ax.plot(np.linspace(0,tmax,11),(Brdata[1:,0]/starting_comp['Br'])/(Cldata[1:,0]/starting_comp['Cl']), label = "Br/Cl (r=0 um)")

ax.plot(np.linspace(0,tmax,11),(Fdata[1:,0]/starting_comp['F'])/(Cldata[1:,0]/starting_comp['Cl']), label = "F/Cl (r=0 um)")

ax.legend()
ax.axhline(y=1, color='k', linestyle='--')
ax.set_xticks(np.linspace(0,tmax,6))
ax.set_xlabel('Simulation time (s)')
ax.set_ylabel('Halogen Ratio (norm)')

#%%