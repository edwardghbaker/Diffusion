# -*- coding: utf-8 -*-
"""
Diffusion Model for Halogen Diffusion in Magmatic Systems

This module simulates diffusion processes in spherical geometries, particularly:
1. Thermal diffusion - cooling of a hot sphere
2. Material diffusion - diffusion of halogens (F, Cl, Br) from an initial state
3. Coupled diffusion - simultaneous thermal and material diffusion

The code solves the diffusion equation (Fick's laws) using finite difference methods
and analytical solutions (eigenvalue expansion for thermal cooling).

Created on Tue Mar  8 22:13:44 2022

@author: Ed
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import scipy.special as spec
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import bisect
from tqdm import tqdm
pi = np.pi
import time as time

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================
figsize = (5,3)
figsize1by2 = (6,8)
mpl.rcParams.update({'font.size': 8})

# ============================================================================
# PHYSICAL CONSTANTS AND REFERENCE DATA
# ============================================================================
# Data source: Alletti et al. (2007) - Halogen diffusion in basaltic melt

# DataFrame containing pre-exponential diffusion coefficients (D0), activation energies (Ea), and their uncertainties for halogens
# D0 is in m^2/s, Ea (activation energy) is in J/mol
dataD0 = pd.DataFrame({'D0': [5.9e-4, 3.3e-2, 7.5e-5],
                        'Ea': [218.2e3, 277.2e3, 199.1e3], 
                        'Ea_Err': [33.5e3, 8.1e3, 33.3e3]},
                      index=['F','Cl','Br'])

dataD_na = pd.DataFrame({'D0': [None],
                        'Ea': [None], 
                        'Ea_Err': [None]},
                      index=['Na'])


SIMS_detectionLimit = pd.DataFrame(columns=['F[ppm]','Cl[ppm]','Br[ppm]','S[ppm]'],data=[[42,6.9,0.019,6.7]])


class Ficks:
    
    def f1L(D,dc_dx):
        """Fick's First Law - Calculate diffusive flux.
        
        Flux is proportional to concentration gradient with diffusion coefficient.
        Relates diffusive flux to concentration gradient.
        
        Args:
            D: Diffusion coefficient (m²/s)
            dc_dx: Concentration gradient dC/dx (1/m)
            
        Returns:
            J: Diffusive flux (mol/(m²·s)) = -D * dC/dx
        """
        # Negative sign indicates flux flows opposite to concentration gradient
        J = -D*dc_dx
        return J
    
    def f2L(D,dc2_dx2):
        """Fick's Second Law - Calculate concentration change over time.
        
        Relates temporal concentration change to spatial curvature of concentration.
        This is the fundamental diffusion equation: dC/dt = D*d²C/dx²
        
        Args:
            D: Diffusion coefficient (m²/s)
            dc2_dx2: Second derivative of concentration d²C/dx² (1/m²)
            
        Returns:
            dc_dt: Rate of concentration change (ppm/s) = D * d²C/dx²
        """
        # Positive second derivative (concentration "bulges up") causes C to increase
        dc_dt = D*dc2_dx2
        return dc_dt
        dc_dt = D*dc2_dx2
        return dc_dt

   
# ============================================================================
# DIFFUSION LAWS - Analytical and Numerical Solutions
# ============================================================================
    
    def pecletNumber_Time(L2,D):
        """Calculate characteristic diffusion time based on distance and diffusion coefficient.
        
        Args:
            L2: Characteristic length scale squared (L²)
            D: Diffusion coefficient
            
        Returns:
            Characteristic time scale (L²/D)
        """
        return (L2**2)/(D)
        
    def pecletNumber_Distance(time,D):
        """Calculate characteristic diffusion distance for a given time.
        
        Args:
            time: Time duration
            D: Diffusion coefficient
            
        Returns:
            Characteristic diffusion distance (sqrt(time * D))
        """
        return np.sqrt(time*D)
    
    def characteristicTime_Sphere(R,D):
        """Calculate characteristic diffusion time for a spherical system.
        
        Args:
            R: Sphere radius
            D: Diffusion coefficient
            
        Returns:
            Characteristic time scale (0.5 * R²/D)
        """
        return 0.5*(R**2)/D
    
    def stabilityCriterion(D,dt,dx):
        """Calculate numerical stability criterion for finite difference scheme.
        
        Args:
            D: Diffusion coefficient
            dt: Time step
            dx: Spatial step
            
        Returns:
            Stability number (D*dt/dx²). Should be < 0.5 for stability.
        """
        A = (D*dt)/(dx**2)
        return A
    
    def arrheniusD(D0,Ea,Tk):
        """Calculate temperature-dependent diffusion coefficient using Arrhenius equation.
        
        The Arrhenius equation models how diffusion increases exponentially with temperature.
        Formula: D(T) = D0 * exp(-Ea / (R*T))
        
        Args:
            D0: Pre-exponential diffusion coefficient (m²/s)
            Ea: Activation energy (J/mol) - energy barrier to diffusion
            Tk: Temperature (Kelvin)
            
        Returns:
            D: Temperature-dependent diffusion coefficient (m²/s)
        """
        R = scipy.constants.R  # Gas constant: 8.314 J/(mol·K)
        # At higher T, exponential term increases, so D increases
        D = D0*np.exp(-Ea/(R*Tk))
        return D
    
    def arrheniusD_ele(ele,Tc):
        """Calculate diffusion coefficient for a specific element at given temperature.
        
        Args:
            ele: Element name ('F', 'Cl', or 'Br')
            Tc: Temperature in Celsius
            
        Returns:
            D: Temperature-dependent diffusion coefficient for the element
        """
        Tk = Tc+273.15
        D0 = dataD0.loc[ele]['D0']
        Ea = dataD0.loc[ele]['Ea']
        R = scipy.constants.R
        D = D0*np.exp(-Ea/(R*Tk))
        return D
        
    def constantConcentration(n0,D,x,t):
        """Calculate concentration profile at constant boundary condition.
        
        Solves diffusion equation with semi-infinite domain and fixed concentration
        boundary. Uses the complementary error function (erfc) solution.
        
        Args:
            n0: Initial/boundary concentration
            D: Diffusion coefficient (m²/s)
            x: Distance from boundary (m)
            t: Time (s)
            
        Returns:
            n: Concentration at distance x and time t
        """
        # Analytical solution: n(x,t) = n0 * erfc(x / (2*sqrt(D*t)))
        # The argument sqrt(D*t) represents the characteristic diffusion length
        n = n0*spec.erfc((x)/(2*np.sqrt(D*t)))   
        return n
    
    def D_1(x,h,t,C0,D0,Ea,Tk):
        """Calculate concentration profile for diffusion from finite slab.
        
        Solves diffusion equation for material diffusing from a finite-thickness slab
        into surrounding material. Uses analytical solution from Crank (1975).
        
        Reference:
            Equation from Crank, J. (1975). The Mathematics of Diffusion. 
            Clarendon Press, Oxford.
        
        Args:
            x: Position in the domain (m)
            h: Half-thickness of the slab (m)
            t: Time (s)
            C0: Initial concentration in slab (ppm)
            D0: Pre-exponential diffusion coefficient (m²/s)
            Ea: Activation energy (J/mol)
            Tk: Temperature (Kelvin)
            
        Returns:
            C: Concentration profile at position x and time t (ppm)
            D: Temperature-dependent diffusion coefficient (m²/s)
        """
        # Calculate temperature-dependent diffusion coefficient
        D = DiffLaws.arrheniusD(D0,Ea,Tk)
        # Use complementary error function solution for slab geometry
        # Sum erfc of symmetric distances to account for diffusion from both surfaces
        C = 0.5*C0*(spec.erfc((h-x)/(2*np.sqrt(D*t)))+spec.erfc((h+x)/(2*np.sqrt(D*t))))
        return C,D
    
    def makeDiffCoeffPlots():
        """Generate Arrhenius plot of diffusion coefficients for halogens.
        
        Creates a plot showing temperature-dependent diffusion coefficients
        for F, Cl, and Br with both Celsius and inverse temperature scales.
        """
        fig, ax = plt.subplots(figsize=figsize,dpi=960)

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
        








#%% Thermal diffusion profiles 

"""
Thermal Diffusion Class

Solves transient heat conduction in a sphere with fixed outer temperature boundary.
Uses analytical eigenvalue expansion method for efficiency.
"""


class ThermalDiff:
    
    def __init__(self,x1=0.0001,x2=500*pi,Bi_num=10,n_search=10000,
                 Ti=2000,Tout=1000,R=5000e-6,R_steps=100,alpha=0.3e-6,plot=15,t_max=10,t_steps=5):
        """Initialize thermal diffusion model parameters.
        
        Args:
            x1: Starting point for root search
            x2: Ending point for root search
            Bi_num: Biot number
            n_search: Number of search points for root finding
            Ti: Initial temperature (K)
            Tout: Outer boundary temperature (K)
            R: Sphere radius (m)
            R_steps: Number of radial discretization steps
            alpha: Thermal diffusivity (m²/s)
            plot: Number of time steps to plot
            t_max: Maximum time (s)
            t_steps: Number of time steps
        """
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
        """Calculate Biot number eigenvalue equation.
        
        Args:
            zeta_n: Eigenvalue candidate
            self: ThermalDiff instance
            
        Returns:
            Value of the Biot eigenvalue equation at zeta_n
        """
        Bi_num = self.Bi_num # need to make the search 1000 times bigger than the Bi
        return 1-Bi_num-zeta_n*((np.cos(zeta_n))/(np.sin(zeta_n)))
        
    def findNRoots(self):
        """Find eigenvalues of the Biot equation for spherical symmetry.
        
        Uses bisection method to find roots of the transcendental equation
        defining thermal diffusion eigenvalues in a sphere.
        
        Returns:
            roots: Array of eigenvalues (zeta_n values)
        """
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
        """Calculate temperature profile in a cooling sphere at given time.
        
        Args:
            zeta: Array of eigenvalues from findNRoots
            t: Time (s)
            
        Returns:
            profile: Temperature profile from center to surface
            fullOutput: Contributions from each eigenvalue
        """
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
        """Execute thermal diffusion model and plot temperature evolution.
        
        Calculates and plots temperature profiles at multiple time steps
        as the sphere cools following initial thermal conditions.
        
        Uses eigenvalue expansion method to solve heat equation in spherical geometry.
        """
        # Find eigenvalues that satisfy the thermal boundary condition (Biot number)
        roots = ThermalDiff.findNRoots(self)
        # Create figure with single plot for temperature vs radius
        fig1,ax1 = plt.subplots(figsize=figsize)
        ax1.set_ylabel('Temperature ($^\circ$C)')
        ax1.set_xlabel('Radius ($\u03BCm$)')


        # Calculate and plot temperature profiles at regular time intervals
        for i in tqdm(np.linspace(0,self.t_max,self.t_steps)):
            # Get temperature profile at current time
            x,fullOutput = ThermalDiff.Tprofile_sphere(self,zeta=roots,t=i)
            # Plot with time as label
            ax1.plot(1e6*np.linspace(0,self.R,self.R_steps),x,label=str(str(i)+' secs'))
        ax1.legend()


#%% Step by step 

"""
Step-by-Step Material Diffusion Class

Simulates diffusion of elemental species (F, Cl, Br) from initial concentration
using explicit finite difference scheme. Temperature can be constant or vary linearly.

Uses Fick's first law with finite differences to discretize and time-march the solution.
"""


class StepDiffusion:

    def __init__(self,R=5000e-6,R_steps=10,element='Cl',Tc=1500,t_steps='Auto',t_max=2e4,C0=100,Cout=10,plot=10,
                 delT=None,legend=True,sphericFactor=True,plotDetectionLimit=True):
        """Initialize step-by-step diffusion model parameters.
        
        Args:
            R: Sphere radius (m)
            R_steps: Number of radial discretization steps
            element: Element to diffuse ('F', 'Cl', or 'Br')
            Tc: Temperature in Celsius
            t_steps: Number of time steps ('Auto' or int)
            t_max: Maximum time (s)
            C0: Initial concentration (ppm)
            Cout: Outer boundary concentration (ppm)
            plot: Number of profiles to plot
            delT: Temperature gradient if isothermal (None for isothermal)
            legend: Whether to include legend in plots
            sphericFactor: Whether to apply spherical geometry corrections
            plotDetectionLimit: Whether to show detection limit on plots
        """
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
        """Create initial concentration profile for diffusion simulation.
        
        Sets uniform concentration throughout the sphere with fixed concentration
        at the boundary (constant outer boundary condition).
        
        Returns:
            X: Radial distance array (m)
            Ci: Initial concentration array (ppm)
        """
        R=self.R
        C0=self.C0  # Initial/core concentration
        Cout=self.Cout  # Boundary concentration
        R_steps=self.R_steps
        
        # Create uniform radial grid
        X = np.linspace(0,R,R_steps)
        # Initialize with uniform concentration
        Ci = np.ones(len(X))*C0
        # Set boundary condition (at sphere surface)
        Ci[-1]=Cout
        return X,Ci
    
    def sphericFactor(self,X):
        """Calculate geometric correction factor for spherical flux geometry.
        
        Accounts for the change in surface area with radius in spherical coordinates.
        The factor adjusts the flux based on the surface area at each node.
        
        Args:
            X: Radial distance array
            
        Returns:
            factor: Array of geometric correction factors for each radial step
        """
        factor = np.ones(len(X)-1)
        # Calculate factor for each shell interface
        for i in range(len(X)-1):
            r = X[i]  # Current radial position
            dr = X[i+1]-X[i]  # Radial spacing
            # Geometric factor accounts for spherical surface area change (4*pi*r^2)
            f = ((3*r**2)+(3*r*dr)+(dr**2))/((3*r**2)+(9*r*dr)+(7*dr**2))
            factor[i] = f
        
        self.factor = factor
        return factor
    
    def applyF1L(self,X,C,dt,factor,j):    
        """Apply Fick's first law to update concentration profile (explicit single step).
        
        Args:
            X: Radial distance array
            C: Current concentration profile
            dt: Time step
            factor: Spherical geometry correction factors
            j: Current time step index
            
        Returns:
            C: Updated concentration profile
        """
        D = self.D
        Cout = self.Cout
        
        # Calculate flux at each radial interface using Fick's first law: J = -D*dC/dx
        # Handle both time-varying D (array) and constant D (scalar)
        try: J = [-D[j]*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]
        except: J = [-D*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]
        
        # Apply spherical geometry correction factor to outward fluxes
        if self.sphericFactor == True:
            Jb = J*factor
        else:
            Jb = J*factor
        
        # Update concentrations: remove flux leaving node, add flux entering node
        C[:-1] = C[:-1]-J  # Concentration decreases by outgoing flux
        C[1:] = C[1:]+Jb   # Concentration increases by incoming flux (with geometry factor)
        C[-1] = Cout # Enforce fixed boundary condition at surface
        
        return C
    
    def F1L_diffworkshop(self,X,C,dt,factor,j):
        """Apply Fick's first law with centered finite differences (alternative implementation).
        
        This version uses vectorized operations for better performance.
        
        Args:
            X: Radial distance array
            C: Current concentration profile
            dt: Time step
            factor: Spherical geometry correction factors
            j: Current time step index
            
        Returns:
            C: Updated concentration profile
        """
        D = self.D
        Cout = self.Cout
        
        # Expand D to single time step value across all spatial nodes
        D = np.ones(len(X))*D[j]
        # Vectorized calculation of diffusive flux using centered differences
        # dC/dx ≈ (C[i+1] - C[i-1]) / 2dx for interior points
        J = np.array([((dt/((X[1:-1]-X[0:-2])**2))*-D[1:-1]*(C[2:]-2*C[1:-1]+C[:-2]))]) # vectorised
        
        # Prepend first boundary flux
        J = np.insert(J,0,J[0,0])
        
        # Apply spherical geometry correction factor to outward fluxes
        if self.sphericFactor == True:
            Jb = J*factor
        else:
            Jb = J

        # Apply diffusive flux to update concentrations
        C[:-1] = C[:-1]-J
        C[1:] = C[1:]+Jb
        
        # Enforce boundary conditions at both ends
        C[-1] = Cout  # Fixed concentration at surface
        C[1],C[0] = C[2],C[2]  # Symmetry boundary at center
        C[0] = C[1]  # Ensure center continuity
        
        return C        

    
    def runModel(self):
        """Execute step-by-step diffusion model simulation.
        
        Runs finite difference diffusion simulation with stability checks
        and generates concentration vs radius plot.
        
        Returns:
            Ci: Concentration profiles at selected time steps
            factor: Spherical geometry correction factors
            fig2 (optional): Matplotlib figure object
            ax2 (optional): Matplotlib axes object
        """
        # Get model parameters
        plot = self.plot
        D = self.D  # Diffusion coefficient (temperature-dependent if specified)
        dt = self.dt  # Time step size
        R = self.R  # Sphere radius
        R_steps = self.R_steps  # Number of radial steps
        t_steps = self.t_steps  # Total number of time steps
        ele = self.element  # Element being diffused
        plotDetectionLimit = self.plotDetectionLimit  # Flag for SIMS detection limit
        
        # Check numerical stability: Stability criterion (Courant number) must be < 0.5
        if DiffLaws.stabilityCriterion(D,dt,(R/R_steps)).min() > 0.5:
            print(str('Fail - Stability criteria not met '+'- '+ str(DiffLaws.stabilityCriterion(D,dt,(R/R_steps)).max())))

        # Initialize spatial grid and concentration profile
        X,Ci = StepDiffusion.makeInitial(self)  # Create initial profile
        factor = StepDiffusion.sphericFactor(self,X)  # Calculate geometry factors
        
        # Start with initial concentration profile
        C = Ci
        
        # Main time-stepping loop
        for j in tqdm(range(t_steps)):
            # Update concentration profile using Fick's law
            C = StepDiffusion.applyF1L(self,X,C,dt,factor=factor,j=j)
            # Store concentration profile at regular intervals for plotting
            if j%(int(t_steps/plot)) == 0:
                Ci = np.vstack((Ci,C))  # Store profile
            if j == t_steps:
                Ci = np.vstack((Ci,C))  # Store final profile
        
        # Generate plot of concentration evolution
        if plot != None:
            fig2,ax2 = plt.subplots(figsize=figsize)
            # Plot each stored concentration profile with time label
            for i in range(Ci.shape[0]):
                ax2.plot(1e6*X,Ci[i,:],label=str(str(int(i*dt*int(t_steps/plot)))+' secs'))
            
            ax2.set_ylabel('Concentraiton ($\u03BCg.g^{-1}$)')
            ax2.set_xlabel('Radius ($\u03BCm$)')
            ax2.set_title(str(f'{self.element} for '+str(int(self.t_max/(60)))+' minutes'))
            
            # Overlay SIMS detection limit threshold if requested
            if plotDetectionLimit == True:
                ax2.axhline(SIMS_detectionLimit[str(ele+'[ppm]')][0],c='k')
                
            if self.legend == True:
                ax2.legend()
            
            # fig2.savefig(str(str(self.element)+'R_steps_'+str(self.R_steps)+'t_steps_'+str(self.t_steps)+'.png'))
            return Ci,factor,fig2,ax2
        
        return Ci,factor


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
        """Initialize coupled thermal and material diffusion model.
        
        This model simulates coupled cooling and halogen diffusion from a 
        finite spherical object into infinite space with constant boundary.
        
        Args:
            x1: Starting point for eigenvalue root search
            x2: Ending point for eigenvalue root search
            Bi: Biot number for thermal problem
            R_steps: Number of radial discretization steps
            R: Sphere radius (m)
            C0: Initial concentration (ppm)
            Cout: Outer boundary concentration (ppm)
            Ti: Initial sphere temperature (K)
            Tout: Outer boundary temperature (K)
            alpha: Thermal diffusivity (m²/s), typically 0.3e-6
            element: Diffusing element ('F', 'Cl', or 'Br')
            t_steps: Number of time steps ('Auto' or int)
            t_max: Maximum simulation time (s)
            plot: Number of profiles to plot
            legend: Whether to include legend in plots
            plotDetectionLimit: Whether to show SIMS detection limit
            sphericFactor: Whether to apply spherical geometry corrections
        """
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
        """Find eigenvalues of the thermal diffusion problem in a sphere.
        
        Solves the transcendental Biot equation for the given Biot number
        using bisection method on a discretized search domain.
        
        Returns:
            zeta: Array of eigenvalues (zeta_n) for thermal problem
        """
        x1 = self.x1
        x2 = self.x2
        Bi = self.Bi
        
        f = lambda x: 1-self.Bi-x*((np.cos(x))/(np.sin(x)))
        
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
        """Initialize concentration and spatial grids for simulation.
        
        Creates radial position arrays and applies initial and boundary
        concentration conditions. Calculates spherical geometry factors.
        
        Returns:
            X: Radial position array (uniform spacing)
            r: Radial position array (for geometric calculations)
            Ci: Initial concentration profile
            factor: Spherical geometry correction factors
        """
        R_steps = self.R_steps
        R = self.R
        C0 = self.C0  # Initial concentration
        Cout = self.Cout  # Boundary concentration
        
        # Create radial grids for calculations
        r = np.linspace(0.000001,R,R_steps)  # Near-center node slightly offset to avoid singularity
        X = np.linspace(0,R,R_steps)  # Uniform grid
        factor = np.ones(R_steps-1)  # Spherical geometry factors
        # Initialize with uniform concentration
        Ci = np.ones(R_steps)*C0
        Ci[-1]=Cout  # Set boundary condition
        
        # Calculate spherical geometry correction factors for each shell interface
        # These account for the change in surface area from r to r+dr
        for i in range(len(r)-1):
            dr = r[i+1]-r[i]  # Shell thickness
            # Geometric factor corrects flux for spherical surface area (proportional to r^2)
            factor_i = ((3*r[i]**2)+(3*r[i]*dr)+(dr**2))/((3*r[i]**2)+(9*r[i]*dr)+(7*dr**2))
            factor[i] = factor_i
        
        # Store for later use
        self.X = X
        self.r = r
        self.Ci = Ci
        self.factor = factor
        
        return X,r,Ci,factor
    
    def Tprofile_sphere(self,zeta,t,r,Ti,Tout):
        """Calculate temperature profile in cooling sphere at given time.
        
        Uses analytical solution to heat diffusion equation with
        Eigenvalue expansion method.
        
        Args:
            zeta: Array of eigenvalues from findNRoots
            t: Current time (s)
            r: Radial position array (m)
            Ti: Initial temperature (K)
            Tout: Outer boundary temperature (K)
            
        Returns:
            profile: Temperature profile at current time
        """
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
        """Calculate temperature and diffusion coefficient matrices over time.
        
        Solves the transient heat diffusion in a sphere and calculates
        the resulting spatially and temporally varying diffusion coefficients.
        Stores results in self.TempProfiles and self.DMatrix.
        """
        element = self.element
        t_steps = self.t_steps
        t_max = self.t_max
        Ti = self.Ti  # Initial temperature
        Tout = self.Tout  # Outer boundary temperature
        plot = self.plot
        
        # Solve thermal problem: find eigenvalues and calculate temperature profiles
        zeta = CoupledModel.findNRoots(self)  # Eigenvalues for thermal diffusion
        X,r,Ci,factor = CoupledModel.makeInitial(self)  # Initialize spatial grid
        # Create matrix to store temperature at each time step and radial position
        TempProfiles = np.ones((t_steps,len(r)))
        
        # Calculate temperature profile at each time step
        for i,t in enumerate(tqdm(np.linspace(0,t_max,t_steps))):
            profile = CoupledModel.Tprofile_sphere(self,zeta=zeta, t=t, r=r, Ti=Ti, Tout=Tout)
            TempProfiles[i,:] = profile
            
        # Fix initial condition (t=0) with starting values to avoid numerical artifacts
        TempProfiles[0,:] = Ti
        TempProfiles[0,-1] = Tout
        
        # Calculate temperature-dependent diffusion coefficients using Arrhenius equation
        # D = D0 * exp(-Ea/(R*T))
        DMatrix = DiffLaws.arrheniusD(dataD0.loc[element]['D0'],dataD0.loc[element]['Ea'],TempProfiles+273.15)
        
        # Plot temperature and diffusion coefficient profiles at selected time steps
        if plot != 0:
            fig3,[ax3_1,ax3_2]=plt.subplots(2,1,figsize=figsize1by2,sharex=True)
            fig3.tight_layout()
            # Evenly spaced time steps for plotting
            for i in range(0,len(TempProfiles[:,0]),int(len(TempProfiles[:,0])/plot)):
                ax3_1.plot(r*1e6,TempProfiles[i,:])  # Plot in micrometers
                ax3_2.plot(r*1e6,DMatrix[i,:])  # Diffusion coefficient varies with temperature
            
            ax3_1.set_ylabel('Temperature ($^\circ$C)')
            ax3_2.set_ylabel('Diffusivity $M^2s^{-1}$')
            ax3_2.set_xlabel('Radius ($\u03BCm$)')
        else: next
        self.TempProfiles = TempProfiles
        self.DMatrix = DMatrix
    
    def f1L(self,X,C,D,j):    
        """Apply Fick's first law with temperature-dependent diffusion.
        
        Updates concentration profile using explicit finite difference
        method with spatially varying diffusion coefficient.
        
        Args:
            X: Radial position array
            C: Current concentration profile
            D: Diffusion coefficient matrix (from DTMatrix)
            j: Current time step index
            
        Returns:
            C: Updated concentration profile
        """
        dt = self.dt  # Time step size
        factor = self.factor  # Spherical geometry factors
        Cout = self.Cout  # Boundary concentration

        # Calculate diffusive flux at each interface: J = -D*dC/dx
        # Try to use time-varying D[j,i], fall back to constant D if needed
        try: J = [-D[j,i]*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]
        except: J = [-D*dt*((C[i+1]-C[i])/((X[i+1]-X[i])**2)) for i in range(len(X)-1)]

        # Apply spherical geometry corrections to outward fluxes
        if self.sphericFactor == True:
            Jb = J*factor  # Scale by surface area factor
        else:
            Jb = J*factor
        
        # Update concentrations using finite difference method
        C[:-1] = C[:-1]-J  # Decrease concentration by outgoing flux
        C[1:] = C[1:]+Jb   # Increase concentration by incoming flux
        C[-1] = Cout # Enforce boundary condition at surface
        
        return C
    
    def runModel(self):
        """Execute full coupled thermal-material diffusion simulation.
        
        Integrates thermal cooling in the sphere with element diffusion
        using temperature-dependent diffusion coefficients. Generates
        concentration vs radius plots over simulation time.
        
        Returns:
            Ci: Concentration profiles at selected time steps
            factor: Spherical geometry correction factors
            fig2 (optional): Matplotlib figure object if plot != None
            ax2 (optional): Matplotlib axes object if plot != None
        """
        # Initialize spatial grid and calculate thermal/diffusion matrices
        X,r,Ci,factor = CoupledModel.makeInitial(self)
        CoupledModel.DTMatrix(self)  # Solve coupling: thermal problem then diffusion
        
        # Get simulation parameters
        plot = self.plot
        dt = self.t_max/self.t_steps
        R = self.R
        R_steps = self.R_steps
        t_steps = self.t_steps
        DMatrix = self.DMatrix  # Temperature-dependent diffusion coefficients
        
        ele = self.element
        plotDetectionLimit = self.plotDetectionLimit
        Dmax = np.max(DMatrix)  # Maximum diffusion coefficient for stability check

        # Check numerical stability: Courant number must be < 0.5 for stability
        if DiffLaws.stabilityCriterion(Dmax,dt,(R/R_steps)).min() > 0.5:
            print(str('Fail - Stability criteria not met '+'- '+ str(DiffLaws.stabilityCriterion(Dmax,dt,(R/R_steps)).max())))
        
        C = Ci.copy()  # Current concentration profile (start with initial profile)

        # Time stepping loop - calculate concentration evolution
        for j in range(t_steps):
            # Update concentration using Fick's law with temperature-dependent D
            C = CoupledModel.f1L(self,X,C,D=DMatrix,j=j)
            # Store profile at regular intervals for plotting
            if j%(int(t_steps/plot)) == 0:
                Ci = np.vstack((Ci,C))       
            if j == t_steps:
                Ci = np.vstack((Ci,C))
        
        # Generate plot of concentration profiles
        if plot != None:
            fig2,ax2 = plt.subplots(figsize=figsize)
            # Plot each stored concentration profile
            for i in range(Ci.shape[0]):
                ax2.plot(1e6*X,Ci[i,:],label=str(str(int(i*dt*int(t_steps/plot)))+' secs'))
            
            ax2.set_ylabel('Concentraiton ($\u03BCg.g^{-1}$)')
            ax2.set_xlabel('Radius ($\u03BCm$)')
            ax2.set_title(str(f'{self.element} for '+str(int(self.t_max/(60)))+' minutes'))
            
            # Overlay SIMS detection limit if requested
            if plotDetectionLimit == True:
                ax2.axhline(SIMS_detectionLimit[str(ele+'[ppm]')][0],c='k')
                
            if self.legend == True:
                ax2.legend()
            
            # fig2.savefig(str(str(self.element)+'R_steps_'+str(self.R_steps)+'t_steps_'+str(self.t_steps)+'.png'))
            return Ci,factor,fig2,ax2
        
        return Ci,factor
    
    


# %%
