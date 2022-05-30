# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 22:13:44 2022

@author: Ed
"""
import math as m
import numpy as np
import pandas as pd
import scipy.optimize as sciop
import scipy.special as spec 
import scipy
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from scipy.optimize import bisect 
from sympy import Float
from mpmath import sin as mpSin, cos as mpCos, exp as mpExp, nsum as mpNsum, inf as mpInf, pi as mpPi
from tqdm import tqdm 
from multiprocessing import Process
pi = np.pi



#%% constants and other data

'''    data from Alletti, M., Baker, D. R., & Freda, C. (2007). Halogen diffusion in a basaltic melt. Geochimica et Cosmochimica Acta, 71(14), 3570–3580. https://doi.org/10.1016/j.gca.2007.04.018
'''


dataD0 = pd.DataFrame({'D0': [5.9e-4, 3.3e-2, 7.5e-5], 
                       'Ea': [218.2e3, 277.2e3, 199.1e3], 
                       'Ea_Err': [33.5e3, 8.1e3, 33.3e3]},
                      index=['F','Cl','Br'])


#%% Basic Laws

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
    
    def arrheniusD(D0,Ea,T):
        R = scipy.constants.R
        D = D0*np.exp(-Ea/(R*T))
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
    
    def Balluffi_thinFilm(L=150e-6,C0=1,time=[1e1],Tk=1500,steps=100,element='Br',Plot=True):
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        D = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],Tk)    

        for t in time:
            C = []
            for x in np.linspace(0,L,steps):
                c = Float(((4*C0)/(pi))*(mpNsum(lambda j: ((1/(2*j+1))*
                                                   (mpSin((2*j+1)*mpPi*x/L))*
                                                   (mpExp((D*t*mpPi**2)*(-(2*j+1)**2)/(L**2)))),
                                        [0, mpInf])))
                C.append(c)
                
            if Plot == True:
                #note x data multiplied by 1e6 to convert to microns
                ax.plot(np.linspace(L/2,L,steps//2)*1e6,C[len(C)//2:],label=str(str(t)+' sec'))
                ax.set_ylabel('Concentration')
                ax.set_xlabel('Distance (\u03BCm)')
                ax.legend()
            else: next
        return C,ax

#%%Thermal diffusion profiles 


class ThermalDiff:
    
    def __init__(self,x1=0.0001,x2=500*pi,Bi_num=10,n_search=10000,
                 Ti=2000,Tout=100,R=5000e-6,R_steps=1000,alpha=0.3e-6,plot=15,t_max=10,t_steps=5):
        
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

# TD = ThermalDiff()
# TD.runModel()



#%% Step by step 


class StepDiffusion:
    
    dataD0 = pd.DataFrame({'D0': [5.9e-4, 3.3e-2, 7.5e-5], 
                           'Ea': [218.2e3, 277.2e3, 199.1e3], 
                           'Ea_Err': [33.5e3, 8.1e3, 33.3e3]},
                          index=['F','Cl','Br'])
    
    def __init__(self,R=5000e-6,R_steps=10,element='F',Tc=1500,t_steps=10000,dt=2e2,C0=100,Cout=10,plot=10):
        
        self.R = R
        self.R_steps = R_steps
        self.element = element
        
        self.D = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],Tc+273.15)
        
        self.t_steps = t_steps
        self.dt = dt
        self.C0 = C0
        self.Cout = Cout
        self.plot = plot
    
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
            
        return factor
    
    def applyF1L(self,X,C,dt,factor):    
        D = self.D
        Cout = self.Cout
        
        J = [-D*dt*((C[i+1]-C[i])/(X[i+1]-X[i])) for i in range(len(X)-1)]    
        
        Jb = J*factor
        C[:-1] = C[:-1]-J
        C[1:] = C[1:]+Jb 
        C[-1] = Cout # reapply boundary condition
        
        return C
    
    def runModel(self):
        
        plot = self.plot
        D = self.D
        dt = self.dt
        R = self.R
        R_steps = self.R_steps
        t_steps = self.t_steps
        
        if DiffLaws.stabilityCriterion(D,dt,(R/R_steps)) > 0.5:
            print('Fail - Stability criteria not met')
        
        X,Ci = StepDiffusion.makeInitial(self)
        factor = StepDiffusion.sphericFactor(self,X)
        
        C = Ci
        
        for i in tqdm(range(t_steps)):
            C = StepDiffusion.applyF1L(self,X,C,dt,factor=factor)
            Ci = np.vstack((Ci,C))            
        
        if plot != None:
            fig2,ax2 = plt.subplots(figsize=(6,8))
            for i in range(0,len(Ci[:,:]),int(len(Ci[:,:])/plot)):
                ax2.plot(1e6*X,Ci[i,:],label=str(str(int(i*dt))+' secs'))
            ax2.legend()
            ax2.set_ylabel('Concentraiton ($\u03BCg.g^{-1}$)')
            ax2.set_xlabel('Radius ($\u03BCm$)')
    
        return Ci


SD = StepDiffusion(R=5000e-6,R_steps=1000,element='Cl',Tc=1500,t_steps=10000,dt=2e3,C0=100,Cout=10,plot=10)
Ci = SD.runModel()


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
    
    
    
    
    ***TO DO***
    1. Add in output parameter for proportion of substance lost
    2. Check and see if the model can accept a variable external temperature
    3. Check why the R_steps changes the output
    
    """ 
    
    dataD0 = pd.DataFrame({'D0': [5.9e-4, 3.3e-2, 7.5e-5], 
                           'Ea': [218.2e3, 277.2e3, 199.1e3], 
                           'Ea_Err': [33.5e3, 8.1e3, 33.3e3]},
                          index=['F','Cl','Br'])
    
    
    def __init__(self,x1=0.0001,x2=500*pi,Bi=1,
                 R_steps=25,R=2500e-6,
                 C0=200,Cout=10,
                 alpha=0.3e-6,element='F',
                 t_steps='Auto',t_max=1*1e4,
                 plot=10,
                 Ti=2500,Tout=1500):
        
        self.x1 = x1
        self.x2 = x2
        self.Bi = Bi
        
        self.R_steps = R_steps
        self.R = R
        
        self.C0 = C0
        self.Cout = Cout
        
        self.alpha = alpha
        self.element = element
        
        self.t_max = t_max
        if type(t_steps) == int or type(t_steps) == float:
            self.t_steps = int(t_steps)
        else:
            D = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],Ti+273.15)
            dt = (0.5*((R/R_steps)**2))/(D)
            t_steps = t_max/dt
            self.t_steps = int(t_steps)+1
        print(str('# time steps = '+str(self.t_steps)))
        
        self.plot = plot
        
        self.Ti = Ti
        self.Tout = Tout
        
        
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
        factor = np.ones(R_steps-1)
        Ci = np.ones(R_steps)*C0
        Ci[-1]=Cout
        #### section to make the step diffusion factor
        for i in range(len(r)-1):
            dr = r[i+1]-r[i]
            factor_i = ((3*r[i]**2)+(3*r[i]*dr)+(dr**2))/((3*r[i]**2)+(9*r[i]*dr)+(7*dr**2))
            factor[i] = factor_i
        ####
        return r,Ci,factor
    
    def Tprofile_sphere(self,zeta,t,r,Ti,Tout):
        
        alpha = self.alpha       
        R = max(r)
        output = np.zeros((0,len(r)))
        
        for zeta_n in zeta:
            output = np.vstack([output,(2*((np.sin(zeta_n)-zeta_n*np.cos(zeta_n))/(2*zeta_n-np.sin(2*zeta_n)))*np.exp((-zeta_n**2)*((alpha*t)/(R**2)))*((np.sin(zeta_n*(r/R)))/(zeta_n*(r/R))))])
        output = np.sum(output,axis=0)
        profile = Tout + (Ti-Tout)*output
        
        return profile
    
    def DTMatrix(self):
        element = self.element
        t_steps = self.t_steps
        t_max = self.t_max
        Ti = self.Ti
        Tout = self.Tout
        plot = self.plot
        
        zeta = CoupledModel.findNRoots(self)
        r,Ci,factor = CoupledModel.makeInitial(self)
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
                    
        return TempProfiles,DMatrix,r,Ci,factor,t_steps,t_max
    
    def f1L(r,C,dt,factor,Darray):   
        Cout = C[-1]
        J = [-Darray[i]*dt*((C[i+1]-C[i])/(r[i+1]-r[i])) for i in range(len(r)-1)]
        Jb = J*factor
        C[:-1] = C[:-1]-J
        C[1:] = C[1:]+Jb 
        C[-1] = Cout # reapply boundary condition
        return C
    
    def runModel(self):
        plot = self.plot   
        TempProfiles,DMatrix,r,Ci,factor,t_steps,t_max = CoupledModel.DTMatrix(self)
        C = Ci
        dt = t_max/t_steps
        
        if DiffLaws.stabilityCriterion(
                DiffLaws.arrheniusD(dataD0.loc[self.element]['D0'],dataD0.loc[self.element]['Ea'],self.Ti+273.15),
                self.t_max/self.t_steps,
                (self.R/self.R_steps)) > 0.5:
            
            print(str('Fail - Stability criteria not met  -  ' + str(DiffLaws.stabilityCriterion(
                    DiffLaws.arrheniusD(dataD0.loc[self.element]['D0'],dataD0.loc[self.element]['Ea'],self.Ti+273.15),
                    self.t_max/self.t_steps,
                    (self.R/self.R_steps)))))
        
        if plot != 0:
            fig4,ax4=plt.subplots(figsize=(6,8))
            ax4.plot(r*1e6,Ci)
            ax4.plot(r*1e6,C)
            ax4.set_xlabel('Radius ($\u03BCm$)')
            ax4.set_ylabel('Concentraiton ($\u03BCg.g^{-1}$)')
            
        C = np.vstack([Ci,C])
        
        for i in tqdm(range(1,t_steps)):
            C = np.vstack([C,CoupledModel.f1L(r=r,C=C[-1,:],dt=dt,factor=factor,Darray=DMatrix[i,:])])
        
        if plot != 0:
            for i in range(0,t_steps,int(t_steps/plot)):
                ax4.plot(r*1e6,C[i,:],label=str(dt*i))
            ax4.plot(r*1e6,C[-1,:],label=str(t_max))
            ax4.axvline((r[-1]+r[-2])*1e6/2,label='Boundary')
            ax4.legend()
            ax4.set_ylim((self.Cout-5,self.C0+5))
        return r,C,TempProfiles,DMatrix,t_steps,t_max


# CM = CoupledModel(plot=15,t_steps='Auto',t_max=1e5,R_steps=5)
# CM1 = CoupledModel(plot=15,t_steps='Auto',t_max=1e5,R_steps=10)
# CM2 = CoupledModel(plot=15,t_steps='Auto',t_max=1e5,R_steps=15)

# CM.runModel()
# CM1.runModel()
# CM2.runModel()



