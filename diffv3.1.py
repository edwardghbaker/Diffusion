# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 22:13:44 2022

@author: Ed
"""
import numpy as np
import pandas as pd
import scipy.special as spec
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from sympy import Float
from tqdm import tqdm
pi = np.pi
from line_profiler import LineProfiler
import time as time

from multiprocessing import Process
#from mpmath import mpNsum, mpPi, mpSin, mpExp, mpInf

#%% constants, other data and Basic Laws

'''    data from Alletti, M., Baker, D. R., & Freda, C. (2007). Halogen diffusion in a basaltic melt. Geochimica et Cosmochimica Acta, 71(14), 3570–3580. https://doi.org/10.1016/j.gca.2007.04.018
'''


dataD0 = pd.DataFrame({'D0': [5.9e-4, 3.3e-2, 7.5e-5],
                        'Ea': [218.2e3, 277.2e3, 199.1e3], 
                        'Ea_Err': [33.5e3, 8.1e3, 33.3e3]},
                      index=['F','Cl','Br'])

SIMS_detectionLimit = pd.DataFrame(columns=['F[ppm]','Cl[ppm]','Br[ppm]','S[ppm]'],data=[[42,6.9,0.019,6.7]])


class Ficks:
    
    def f1L(D,dc_dx):
        #J = -D*((c1-c0)*(x1-x0))
        J = -D*dc_dx
        return J
    
    def f2L(D,dc2_dx2):
        dc_dt = D*dc2_dx2
        return dc_dt

   
class DiffLaws:
    
    def pecletNumber_Time(L2,D):
        return (L2**2)/(D)
        
    def pecletNumber_Distance(time,D):
        return np.sqrt(time*D)
    
    def characteristicTime_Sphere(R,D):
        return 0.5*(R**2)/D
    
    def stabilityCriterion(D,dt,dx):
        A = (D*dt)/(dx**2)
        return A
    
    def arrheniusD(D0,Ea,Tk):
        R = scipy.constants.R
        D = D0*np.exp(-Ea/(R*Tk))
        return D
    
    def arrheniusD_ele(ele,Tc):
        Tk = Tc+273.15
        D0 = dataD0.loc[ele]['D0']
        Ea = dataD0.loc[ele]['Ea']
        R = scipy.constants.R
        D = D0*np.exp(-Ea/(R*Tk))
        return D
        
    def constantConcentration(n0,D,x,t):
        n = n0*spec.erfc((x)/(2*np.sqrt(D*t)))   
        return n
    
    def D_1(x,h,t,C0,D0,Ea,Tk):
        
        ''' 
        
        Equation from Crank, J. (1975). the Mathematics of Diffusion Clarendon Press Oxford 1975. 3–4. https://books.google.cl/books?hl=es&lr=&id=eHANhZwVouYC&oi=fnd&pg=PA1&dq=J.+Crank+The+Mathematics+of+Diffusion+(second+ed.),+Oxford+University&ots=fz40zXhgS1&sig=JK5-NVgPC18crrrnDqZJhZLw_Is%0Ahttp://www-eng.lbl.gov/~shuman/NEXT/MATERIALS%26COMPONENTS/Xe_d
        
        '''
        D = DiffLaws.arrheniusD(D0,Ea,Tk)
        C = 0.5*C0*(spec.erfc((h-x)/(2*np.sqrt(D*t)))+spec.erfc((h+x)/(2*np.sqrt(D*t))))
        return C,D
    
    def makeDiffCoeffPlots():
        fig, ax = plt.subplots(figsize=(5,5),dpi=960)

        print(DiffLaws.arrheniusD_ele('Cl',1500))
        hal = ['F','Cl','Br']
        Ts = np.linspace(1000,1600)

        for i in hal:
            ax.plot(10000/(Ts+273.15),DiffLaws.arrheniusD_ele(i,Ts),label=i)

        ax2 = ax.twiny()
        ax.set_xlim((10000/(1600+273.15),10000/(1000+273.15)))
        ax2.set_xlim((1600,1000))
        ax2.set_xlabel('Temperature ($^\circ$C)')
        ax.set_xlabel('10000/T ($T_k$)')
        ax.set_ylabel('Log (D) ($m^2 s^{-1}$)')
        ax.legend()
        ax.set_yscale('log')
        








#%%Thermal diffusion profiles 
'''
Thermal Diffusion


'''

class ThermalDiff:
    
    def __init__(self,x1=0.0001,x2=500*pi,Bi_num=10,n_search=10000,
                 Ti=2000,Tout=1000,R=5000e-6,R_steps=100,alpha=0.3e-6,plot=15,t_max=10,t_steps=5):
        
        self.x1 = x1
        self.x2 = x2
        self.Bi_num = Bi_num
        self.n_search = n_search
        self.Ti = Ti
        self.Tout = Tout
        self.R = R
        self.R_steps = R_steps
        self.alpha = alpha
        self.plot = plot
        self.t_max = t_max
        self.t_steps = t_steps
    
    def Bi(zeta_n,self):
        Bi_num = self.Bi_num # need to make the search 1000 times bigger than the Bi
        return 1-Bi_num-zeta_n*((np.cos(zeta_n))/(np.sin(zeta_n)))
        
    def findNRoots(self):
        x = np.linspace(self.x1,self.x2*np.pi,self.n_search)
        values = ThermalDiff.Bi(x,self)
        bounds_pos = np.array([[]])
        bounds_neg = np.array([[]])
        roots=np.array([])
    
        sign = np.zeros((1,len(values)))
        for i,j in enumerate(values):
            if j > 0:
                sign[0,i] = 1
            elif j < 0:
                sign[0,i] = -1
            else:
                sign[0,i] = 0
        for i in range(len(sign[0,:])):
            if sign[0,i] != sign[0,i-1]:# and abs(values[i]) < 5:
                bounds_pos = np.append(arr=bounds_pos,values=(x[i]))
                bounds_neg = np.append(arr=bounds_neg,values=(x[i-1]))
            else: next
        
        for p,n in zip(bounds_pos,bounds_neg):
            root = bisect(ThermalDiff.Bi,p,n,args=(self))
            roots = np.append(roots,root)
        
        roots = np.delete(roots,0)
        return roots
    
    def Tprofile_sphere(self,zeta,t):
        R = self.R
        R_steps = self.R_steps
        alpha = self.alpha
        Ti = self.Ti
        Tout = self.Tout
        
        r = np.linspace(0.0001,R,R_steps)
        output = np.zeros((0,len(r)))
        for zeta_n in zeta:
            output = np.vstack([output,(2*((np.sin(zeta_n)-zeta_n*np.cos(zeta_n))/(2*zeta_n-np.sin(2*zeta_n)))*np.exp((-zeta_n**2)*((alpha*t)/(R**2)))*((np.sin(zeta_n*(r/R)))/(zeta_n*(r/R))))])
            
        fullOutput = output
        output = np.sum(output,axis=0)
        profile = Tout + (Ti-Tout)*output
        
        return profile,fullOutput
    
    def runModel(self):
        
        roots = ThermalDiff.findNRoots(self)
        fig1,ax1 = plt.subplots(figsize=(6,8))
        ax1.set_ylabel('Temperature ($^\circ$C)')
        ax1.set_xlabel('Radius ($\u03BCm$)')


        for i in tqdm(np.linspace(0,self.t_max,self.t_steps)):
            x,fullOutput = ThermalDiff.Tprofile_sphere(self,zeta=roots,t=i)
            ax1.plot(1e6*np.linspace(0,self.R,self.R_steps),x,label=str(str(i)+' secs'))
        ax1.legend()


#%% Step by step 

'''
Stepwise diffusion


'''


class StepDiffusion:

    def __init__(self,R=5000e-6,R_steps=10,element='Cl',Tc=1500,t_steps='Auto',t_max=2e4,C0=100,Cout=10,plot=10,
                 delT=None,legend=True,sphericFactor=True,plotDetectionLimit=True):
        
        self.R = R
        self.t_max = t_max
        self.R_steps = R_steps
        self.element = element
        self.plotDetectionLimit = plotDetectionLimit
        if t_steps == 'Auto':
            D = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],Tc+273.15)
            dt = (0.5*((R/R_steps)**2))/(D)
            t_steps = int(t_max/dt)+1
            self.t_steps = t_steps

        elif type(t_steps) == int:
            self.t_steps = t_steps
            
        else:
            print('t_steps entered in wrong format, must be "Auto" or int')
            
        self.dt = t_max/t_steps
        
        self.C0 = C0
        self.Cout = Cout
        
        self.plot = plot
        
        self.delT = delT
        self.Tc = Tc
        
        self.legend = legend
        self.sphericFactor = sphericFactor
        
        
        if delT == None:
            self.D = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],Tc+273.15)
            print(self.D)
            print(np.shape(self.D))
        elif type(delT) is int or type(delT) is float:
            Ts = np.linspace(Tc,Tc+delT,t_steps)
            self.D = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],Ts+273.15)
            print(self.D)
            print(np.shape(self.D))
        else:
            print('delT entered in wrong format, must be "None", int or float')
    
    def makeInitial(self):
        
        R=self.R
        C0=self.C0
        Cout=self.Cout
        R_steps=self.R_steps
        
        X = np.linspace(0,R,R_steps)
        Ci = np.ones(len(X))*C0
        Ci[-1]=Cout
        return X,Ci
    
    def sphericFactor(self,X):
        factor = np.ones(len(X)-1)
        for i in range(len(X)-1):
            r = X[i]
            dr = X[i+1]-X[i]
            f = ((3*r**2)+(3*r*dr)+(dr**2))/((3*r**2)+(9*r*dr)+(7*dr**2))
            factor[i] = f
        
        self.factor = factor
        return factor
    
    def applyF1L(self,X,C,dt,factor,j):    
        D = self.D
        Cout = self.Cout
        
        try: J = [-D[j]*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]
        except: J = [-D*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]
        
        if self.sphericFactor == True:
            Jb = J*factor
        else:
            Jb = J*factor
        
        C[:-1] = C[:-1]-J
        C[1:] = C[1:]+Jb 
        C[-1] = Cout # reapply boundary condition
        
        return C
    
    def F1L_diffworkshop(self,X,C,dt,factor,j):
        
        D = self.D
        Cout = self.Cout
        
        #if type(self.T_grad) is not int or type(self.T_grad) is not float:
        D = np.ones(len(X))*D[j]
        #J = [((dt/((X[i+1]-X[i])))*-D[i]*(C[i+1]-2*C[i]+C[i-1])) for i in range(1,len(X)-1)] list comprehension
        J = np.array([((dt/((X[1:-1]-X[0:-2])**2))*-D[1:-1]*(C[2:]-2*C[1:-1]+C[:-2]))]) # vectorised

        J = np.insert(J,0,J[0,0])


        if self.sphericFactor == True:
            Jb = J*factor
        else:
            Jb = J

        #apply diffusive flux
        C[:-1] = C[:-1]-J
        C[1:] = C[1:]+Jb
        
        #boundary conditions
        C[-1] = Cout
        C[1],C[0] = C[2],C[2]
        C[0] = C[1]
        
        return C        

    
    def runModel(self):
        
        plot = self.plot
        D = self.D 
        dt = self.dt
        R = self.R
        R_steps = self.R_steps
        t_steps = self.t_steps
        ele = self.element
        plotDetectionLimit = self.plotDetectionLimit
        

        if DiffLaws.stabilityCriterion(D,dt,(R/R_steps)).min() > 0.5:
            print(str('Fail - Stability criteria not met '+'- '+ str(DiffLaws.stabilityCriterion(D,dt,(R/R_steps)).max())))

        X,Ci = StepDiffusion.makeInitial(self)
        factor = StepDiffusion.sphericFactor(self,X)
        
        C = Ci
        
        for j in tqdm(range(t_steps)):
            C = StepDiffusion.applyF1L(self,X,C,dt,factor=factor,j=j)
            if j%(int(t_steps/plot)) == 0:
                Ci = np.vstack((Ci,C))       
            if j == t_steps:
                Ci = np.vstack((Ci,C))
        
        if plot != None:
            fig2,ax2 = plt.subplots(figsize=(6,6))
            for i in range(Ci.shape[0]):
                ax2.plot(1e6*X,Ci[i,:],label=str(str(int(i*dt*int(t_steps/plot)))+' secs'))
            
            ax2.set_ylabel('Concentraiton ($\u03BCg.g^{-1}$)')
            ax2.set_xlabel('Radius ($\u03BCm$)')
            ax2.set_title(str(f'{self.element} for '+str(int(self.t_max/(60)))+' minutes'))
            
            if plotDetectionLimit == True:
                ax2.axhline(SIMS_detectionLimit[str(ele+'[ppm]')][0],c='k')
                
            if self.legend == True:
                ax2.legend()
            
            fig2.savefig(str(str(self.element)+'R_steps_'+str(self.R_steps)+'t_steps_'+str(self.t_steps)+'.png'))
            return Ci,factor,fig2,ax2
        
        return Ci,factor

# SD_constantT = StepDiffusion(t_max=3e2,t_steps='Auto',element='Cl',R_steps=25,R=500e-6,
#                     sphericFactor=True,C0=100,Cout=0,plot=10,legend=False,
#                     Tc=1500,delT=None)

# Ci1,factor1,fig1,ax1 = SD_constantT.runModel()

##################
# SD_dt0 = StepDiffusion(t_max=5e3,t_steps='Auto',element='Cl',R_steps=25,R=500e-6,
#                     sphericFactor=True,C0=100,Cout=0,plot=10,legend=False,
#                     Tc=1500,delT=-500)

# Ci2,factor2,fig2,ax2 = SD_dt0.runModel()

# SD_dt1 = StepDiffusion(t_max=1e3,t_steps='Auto',element='Cl',R_steps=25,R=500e-6,
#                     sphericFactor=True,C0=100,Cout=0,plot=10,legend=False,
#                     Tc=1500,delT=-500)

# Ci2,factor2,fig2,ax2 = SD_dt1.runModel()

# SD_dt2 = StepDiffusion(t_max=5e2,t_steps='Auto',element='Cl',R_steps=25,R=500e-6,
#                     sphericFactor=True,C0=100,Cout=0,plot=10,legend=False,
#                     Tc=1500,delT=-500)

# Ci2,factor2,fig2,ax2 = SD_dt2.runModel()

# SD_dt3 = StepDiffusion(t_max=1e2,t_steps='Auto',element='Cl',R_steps=25,R=500e-6,
#                     sphericFactor=True,C0=100,Cout=0,plot=10,legend=False,
#                     Tc=1500,delT=-500)

# Ci2,factor2,fig2,ax2 = SD_dt3.runModel()

#%% CoupledModel

class CoupledModel:
    
    """
    
    This is the model for coupled thermal and material diffusion from a finite sperical object into an infinite space with a constant dispurtion
    
    Thermal diffusivity from: Nabelek, P.I., Hofmeister, A.M. and Whittington, A.G., 2012. The influence of temperature-dependent thermal diffusivity on the conductive cooling rates of plutons and temperature-time paths in contact aureoles. Earth and Planetary Science Letters, 317, pp.157-164.
    0.3e-6
    
    Material Diffusivity data from: Baker, D. R., & Balcone-Boissard, H. (2009). Halogen diffusion in magmatic systems: Our current state of knowledge. Chemical Geology, 263 (1–4), 82–88. https://doi.org/10.1016/j.chemgeo.2008.10.010</div>
    
    Diffusion Equations from: Crank, J. (1975). the Mathematics of Diffusion Clarendon Press Oxford 1975. 3–4. https://books.google.cl/books?hl=es&lr=&id=eHANhZwVouYC&oi=fnd&pg=PA1&dq=J.+Crank+The+Mathematics+of+Diffusion+(second+ed.),+Oxford+University&ots=fz40zXhgS1&sig=JK5-NVgPC18crrrnDqZJhZLw_Is%0Ahttp://www-eng.lbl.gov/~shuman/NEXT/MATERIALS%26COMPONENTS/Xe_d
    
    ***How to use code***
    
    
    ***FAQs***
    
    -Why not use parralel processing?
    
    I've tried but because one step is dependent on the output of the second step each must be done in order. If you want to run several simulations at once it is possible to simply impliment multi core processing for that. It may be easier to simply run overnight.

  
    """ 
    
    dataD0 = pd.DataFrame({'D0': [5.9e-4, 3.3e-2, 7.5e-5], 
                           'Ea': [218.2e3, 277.2e3, 199.1e3], 
                           'Ea_Err': [33.5e3, 8.1e3, 33.3e3]},
                          index=['F','Cl','Br'])
       
        
    def __init__(self,x1=0.0001,x2=500*pi,Bi=1,
                 R_steps=25,R=500e-6,
                 C0=200,Cout=10,
                 Ti=2500,Tout=1500,
                 alpha=0.3e-6,element='F',
                 t_steps='Auto',t_max=1*1e4,
                 plot=10,legend=True,plotDetectionLimit=True,
                 sphericFactor=True):
        
        self.x1 = x1
        self.x2 = x2
        self.Bi = Bi
        
        self.R_steps = R_steps
        self.R = R
        
        self.C0 = C0
        self.Cout = Cout
        
        self.Ti = Ti
        self.Tout = Tout
                
        self.alpha = alpha
        self.element = element
        
        self.t_max = t_max
        if type(t_steps) == int or type(t_steps) == float:
            self.t_steps = int(t_steps)
        else:
            D = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],Ti+273.15)
            dt = (0.5*((R/R_steps)**2))/(D)
            self.dt = dt
            t_steps = t_max/dt
            self.t_steps = int(t_steps)+1
        print(str('# time steps = '+str(self.t_steps)))
        
        self.plot = plot
        self.legend = legend
        self.plotDetectionLimit = plotDetectionLimit
        self.sphericFactor = sphericFactor
        
    def findNRoots(self):
        
        Bi = self.Bi
        x1 = self.x1
        x2 = self.x2
        
        f = lambda x: 1-Bi-x*((np.cos(x))/(np.sin(x)))
        
        x = np.linspace(x1,x2*np.pi,Bi*1000)
        values = f(x)
        bounds_pos = np.array([])
        bounds_neg = np.array([])
        zeta=np.array([])
        sign = np.zeros((1,len(values)))
        
        for i,j in enumerate(values):
            if j > 0:
                sign[0,i] = 1
            elif j < 0:
                sign[0,i] = -1
            else:
                sign[0,i] = 0
        for i in range(len(sign[0,:])):
            if sign[0,i] != sign[0,i-1]:# and abs(values[i]) < 5:
                bounds_pos = np.append(arr=bounds_pos,values=(x[i]))
                bounds_neg = np.append(arr=bounds_neg,values=(x[i-1]))
            else: next
        for p,n in zip(bounds_pos,bounds_neg):
            root = bisect(f,p,n)
            zeta = np.append(zeta,root)
        
        np.delete(zeta,0)
        return zeta

    def makeInitial(self):
        
        R_steps = self.R_steps
        R = self.R
        C0 = self.C0
        Cout = self.Cout
        
        r = np.linspace(0.000001,R,R_steps)
        X = np.linspace(0,R,R_steps)
        factor = np.ones(R_steps-1)
        Ci = np.ones(R_steps)*C0
        Ci[-1]=Cout
        
        #### section to make the step diffusion factor
        for i in range(len(r)-1):
            dr = r[i+1]-r[i]
            factor_i = ((3*r[i]**2)+(3*r[i]*dr)+(dr**2))/((3*r[i]**2)+(9*r[i]*dr)+(7*dr**2))
            factor[i] = factor_i
        ####
        
        self.X = X
        self.r = r
        self.Ci = Ci
        self.factor = factor
        
        return X,r,Ci,factor
    
    def Tprofile_sphere(self,zeta,t,r,Ti,Tout):
        
        alpha = self.alpha       
        R = max(r)
        output = np.zeros((0,len(r)))
        
        for zeta_n in zeta:
            output = np.vstack([output,(2*((np.sin(zeta_n)-zeta_n*np.cos(zeta_n))/(2*zeta_n-np.sin(2*zeta_n)))*np.exp((-zeta_n**2)*((alpha*t)/(R**2)))*((np.sin(zeta_n*(r/R)))/(zeta_n*(r/R))))])
        output = np.sum(output,axis=0)
        profile = Tout + (Ti-Tout)*output
        
        #print(np.shape(profile))
        return profile
    
    def DTMatrix(self):
        element = self.element
        t_steps = self.t_steps
        t_max = self.t_max
        Ti = self.Ti
        Tout = self.Tout
        plot = self.plot
        
        zeta = CoupledModel.findNRoots(self)
        X,r,Ci,factor = CoupledModel.makeInitial(self)
        TempProfiles = np.ones((t_steps,len(r)))
        
        for i,t in enumerate(tqdm(np.linspace(0,t_max,t_steps))):
            profile = CoupledModel.Tprofile_sphere(self,zeta=zeta, t=t, r=r, Ti=Ti, Tout=Tout)
            TempProfiles[i,:] = profile
            
        #To get rid of the gaussian problems with t=0, i will just impliment the starting conditions
        TempProfiles[0,:] = Ti
        TempProfiles[0,-1] = Tout
        
        DMatrix = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],TempProfiles+273.15)
        
        if plot != 0:
            fig3,[ax3_1,ax3_2]=plt.subplots(2,1,figsize=(5,10),sharex=True)
            fig3.tight_layout()
            for i in range(0,len(TempProfiles[:,0]),int(len(TempProfiles[:,0])/plot)):
                ax3_1.plot(r*1e6,TempProfiles[i,:])
                ax3_2.plot(r*1e6,DMatrix[i,:])
            
            ax3_1.set_ylabel('Temperature ($^\circ$C)')
            ax3_2.set_ylabel('Diffusivity $M^2s^{-1}$')
            ax3_2.set_xlabel('Radius ($\u03BCm$)')
        else: next
        self.TempProfiles = TempProfiles
        self.DMatrix = DMatrix
    
    def f1L(self,X,C,D,j):    
        
        dt = self.dt
        factor = self.factor

        Cout = self.Cout

        try: J = [-D[j,i]*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]
        except: J = [-D*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]

        if self.sphericFactor == True:
            Jb = J*factor
        else:
            Jb = J*factor
        
        C[:-1] = C[:-1]-J
        C[1:] = C[1:]+Jb 
        C[-1] = Cout # reapply boundary condition
        
        return C
    
    def runModel(self):
        
        X,r,Ci,factor = CoupledModel.makeInitial(self)
        CoupledModel.DTMatrix(self)
        
        plot = self.plot
        dt = self.t_max/self.t_steps
        R = self.R
        R_steps = self.R_steps
        t_steps = self.t_steps
        DMatrix = self.DMatrix
        
        ele = self.element
        plotDetectionLimit = self.plotDetectionLimit
        Dmax = np.max(DMatrix)

        
        if DiffLaws.stabilityCriterion(Dmax,dt,(R/R_steps)).min() > 0.5:
            print(str('Fail - Stability criteria not met '+'- '+ str(DiffLaws.stabilityCriterion(Dmax,dt,(R/R_steps)).max())))
        
        C = Ci

        for j in range(t_steps):
            
            C = CoupledModel.f1L(self,X,C,D=DMatrix,j=j)
            if j%(int(t_steps/plot)) == 0:
                Ci = np.vstack((Ci,C))       
            if j == t_steps:
                Ci = np.vstack((Ci,C))
        
        if plot != None:
            fig2,ax2 = plt.subplots(figsize=(6,6))
            for i in range(Ci.shape[0]):
                ax2.plot(1e6*X,Ci[i,:],label=str(str(int(i*dt*int(t_steps/plot)))+' secs'))
            
            ax2.set_ylabel('Concentraiton ($\u03BCg.g^{-1}$)')
            ax2.set_xlabel('Radius ($\u03BCm$)')
            ax2.set_title(str(f'{self.element} for '+str(int(self.t_max/(60)))+' minutes'))
            
            if plotDetectionLimit == True:
                ax2.axhline(SIMS_detectionLimit[str(ele+'[ppm]')][0],c='k')
                
            if self.legend == True:
                ax2.legend()
            
            fig2.savefig(str(str(self.element)+'R_steps_'+str(self.R_steps)+'t_steps_'+str(self.t_steps)+'.png'))
            return Ci,factor,fig2,ax2
        
        return Ci,factor
    
    
CM = CoupledModel(x1=0.0001,x2=500*pi,Bi=1,
              R_steps=15,R=500e-6,
              C0=200,Cout=10,
              Ti=2500,Tout=1000,
              alpha=0.3e-6,element='Cl',
              t_steps='Auto',t_max=1*1e2,
              plot=10,legend=True,plotDetectionLimit=True,
              sphericFactor=True)

CM.runModel()
    

#%%  Look at the MELTS files 


MELTS = pd.read_table(r"C:/Users/r11403eb/OneDrive - The University of Manchester\MELTS\windows_alphamelts_1-9\EC\Phase_mass_tbl.txt",header=2,sep=' ')

#MELTS.plot(['olivine_0','spinel_0'])


fig_crystal, [ax_crystal,ax_melt] = plt.subplots(2,1,figsize=(6,10))

ax_crystal.plot(MELTS.iloc[:,4:9],MELTS['Temperature'],label=MELTS.columns[4:9])
ax_crystal.set_xlim((-5,100))

ax_melt.plot(MELTS['liquid_0'],MELTS['Temperature'],label=MELTS.columns[3])

fit = np.polyfit(MELTS['Temperature'][48:],MELTS['liquid_0'][48:],deg=3)

ax_melt.plot((fit[3]+fit[2]*MELTS['Temperature']+fit[1]*MELTS['Temperature']**2+fit[0]*MELTS['Temperature']**3),MELTS['Temperature'])
ax_melt.set_xlim((-5,100))

ax_crystal.legend()
ax_melt.legend()




