#%%
# -*- coding: utf-8 -*-
"""
Paper Models - Halogen Diffusion in Chondrules

This module models the diffusion and fractionation of halogens (F, Cl, Br) in
a cooling chondrule (spherical magmatic object) during thermal metamorphism.

Physical scenario:
  - Initial uniform halogen distribution in a 500 μm radius chondrule
  - Linear cooling from 1500°C at a rate of ~1.67°C/s
  - Material diffuses at temperature-dependent ratesx
  - Boundary condition: constant low concentration at surface
  - Tracks differential diffusion rates of F, Cl, and Br

Output:
  - Panel plots showing concentration profiles vs radius
  - Elemental ratios (F/Cl, Br/Cl) - useful for geochemical signatures
  - Normalized ratios relative to initial composition
"""

# ============================================================================
# IMPORTS AND CONSTANTS
# ============================================================================
from Diffusion_Model import StepDiffusion
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import pi

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
# Initial halogen concentrations (ppm)
F_initial = 200       # Fluorine concentration
Cl_initial = 5000     # Chlorine concentration (much higher)
Br_initial = 50       # Bromine concentration
C_out = 10            # Boundary concentration at chondrule surface (ppm)

# Geometric parameters
chondrule_radius = 500e-6  # Chondrule radius in meters (500 micrometers)
r_steps = 101              # Number of radial grid points

# Thermal parameters
T_initial = 1500    # Initial temperature in Celsius
T_change = -600     # Temperature change over simulation (-600°C cooling)
                    # Corresponds to ~1.67°C/s cooling rate
t_max = 60*60       # Total simulation time = 3600 seconds (60 minutes)

figsize = (5,3)
figsize1by2 = (6,8)
mpl.rcParams.update({'font.size': 8})


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    """
    Main script execution for halogen diffusion analysis.
    
    This section runs when the script is executed directly (not imported as a module).
    """
    
    # ======================================================================
    # HALOGEN DIFFUSION MODELS - Create and run simulations
    # ====================================================================== 
    # Each model simulates diffusion of a single halogen element with:
    # - Temperature-dependent diffusion coefficient (Arrhenius equation)
    # - Linear cooling from 1500°C over 60 minutes
    # - Constant boundary concentration at surface
    # - Automatic time-stepping for numerical stability

    print(f"Running halogen diffusion simulations...")
    print(f"Cooling rate: {-T_change/t_max*60:.2f}°C/min over 60 minutes")
    print(f"Chondrule radius: {chondrule_radius*1e6:.0f} μm")
    print(f"Temperature range: {T_initial}°C → {T_initial + T_change}°C")
    print()

    # FLUORINE DIFFUSION MODEL
    # Fluorine is the fastest diffuser among the halogens
    print("Modeling F diffusion...")
    F_model = StepDiffusion(
                            R=chondrule_radius,
                            R_steps=r_steps,
                            element='F',  # Element
                            Tc=T_initial,  # Initial temperature
                            t_steps='Auto',  # Auto-calculate for numerical stability
                            t_max=t_max,    # Total simulation time
                            C0=F_initial,   # Initial concentration
                            Cout=C_out,     # Boundary concentration
                            plot=10,        # Plot 10 time snapshots
                            delT=T_change,  # Temperature change (enables cooling)
                            legend=False,   # No legend
                            sphericFactor=True,  # Account for spherical geometry
                            plotDetectionLimit=False  # Don't show SIMS detection limit
                            )

    F_Ci = F_model.runModel()

    # CHLORINE DIFFUSION MODEL
    # Chlorine is the middle diffuser - much slower than F
    print("Modeling Cl diffusion...")
    Cl_model = StepDiffusion(
                            R=chondrule_radius,
                            R_steps=r_steps,
                            element='Cl',  # Element
                            Tc=T_initial,  # Initial temperature
                            t_steps='Auto',  # Auto-calculate for numerical stability
                            t_max=t_max,    # Total simulation time
                            C0=Cl_initial,  # Initial concentration (high)
                            Cout=C_out,     # Boundary concentration
                            plot=10,        # Plot 10 time snapshots
                            delT=T_change,  # Temperature change (enables cooling)
                            legend=False,   # No legend
                            sphericFactor=True,  # Account for spherical geometry
                            plotDetectionLimit=False  # Don't show SIMS detection limit
                            )

    Cl_Ci = Cl_model.runModel()

    # BROMINE DIFFUSION MODEL
    # Bromine is the slowest diffuser and least soluble
    print("Modeling Br diffusion...")
    Br_model = StepDiffusion(
                            R=chondrule_radius,
                            R_steps=r_steps,
                            element='Br',  # Element
                            Tc=T_initial,  # Initial temperature
                            t_steps='Auto',  # Auto-calculate for numerical stability
                            t_max=t_max,    # Total simulation time
                            C0=Br_initial,  # Initial concentration
                            Cout=C_out,     # Boundary concentration
                            plot=10,        # Plot 10 time snapshots
                            delT=T_change,  # Temperature change (enables cooling)
                            legend=False,   # No legend
                            sphericFactor=True,  # Account for spherical geometry
                            plotDetectionLimit=False  # Don't show SIMS detection limit
                            )

    Br_Ci = Br_model.runModel()

    print("✓ All diffusion models completed")
    print()

    # ============================================================================
    # EXTRACT FINAL CONCENTRATION PROFILES
    # ============================================================================
    # Extract the final concentration profile (last time step) for each halogen
    # The profile shows how much each element has been depleted from center to edge

    # Get the concentration matrix from each model
    # F_Ci, Cl_Ci, Br_Ci are 2D arrays: [time_steps, radial_positions]
    F_comp = F_Ci[0]          # Full F concentration matrix
    Cl_comp = Cl_Ci[0]        # Full Cl concentration matrix
    Br_comp = Br_Ci[0]        # Full Br concentration matrix

    # Extract final concentration profile (last time step, all radii)
    # [:-1] removes the boundary condition point (kept constant at C_out)
    F_comp_final = F_comp[-1,:-1]   # F profile at end of cooling
    Cl_comp_final = Cl_comp[-1,:-1] # Cl profile at end of cooling
    Br_comp_final = Br_comp[-1,:-1] # Br profile at end of cooling

    # ============================================================================
    # ANALYSIS: ELEMENTAL RATIOS AND FRACTIONATION
    # ============================================================================
    print("Creating fractionation analysis plots...")
    print()

    # ============================================================================
    # PLOT 1: Absolute Concentrations and Elemental Ratios
    # ============================================================================
    # Two-panel figure showing:
    # Top: Concentration profiles vs radius for F, Cl, Br after cooling
    # Bottom: F/Cl and Br/Cl ratios showing halogen fractionation

    ratio_fig, r_axes = plt.subplots(2,1,figsize=figsize1by2,constrained_layout=True)

    # Radial distance array for plotting
    x = np.linspace(0,chondrule_radius*1e6,r_steps)

    # TOP PANEL: Absolute concentrations
    r_axes[0].plot(F_comp_final,label='F')
    r_axes[0].plot(Cl_comp_final,label='Cl')
    r_axes[0].plot(Br_comp_final,label='Br')
    r_axes[0].axhline(y=10,linestyle='--',color='k',label='Baseline')  # Baseline = surface concentration
    r_axes[0].set_ylabel('Concentraiton after 60 min ($\u03BCg.g^{-1}$)')
    r_axes[0].set_xlabel(f'Distance from Centre ($\u03BCm$)')

    r_axes[0].legend()

    # BOTTOM PANEL: Elemental ratios (F/Cl and Br/Cl)
    # These ratios show how much each halogen has been preferentially lost
    # F/Cl decreases faster (F diffuses faster) → core becomes Cl-rich and F-poor
    # Br/Cl increases (Br diffuses slower) → core becomes Br-rich and Cl-poor
    r_axes[1].plot(F_comp_final/Cl_comp_final,label='F/Cl')
    r_axes[1].plot(Br_comp_final/Cl_comp_final,label='Br/Cl',color='C2')
    r_axes[1].set_ylabel('Ratio')
    r_axes[1].set_xlabel(f'Distance from Centre ($\u03BCm$)')
    r_axes[1].set_yscale('log')  # Log scale to show ratio variations clearly
    r_axes[1].legend()

    # ============================================================================
    # PLOT 2: Normalized Ratios (Relative to Initial Composition)
    # ============================================================================
    # Normalize ratios relative to their initial values
    # This shows how much fractionation has occurred from the original composition
    # Values < 1: element has been preferentially lost (diffused out)
    # Values > 1: element has been preferentially retained (diffused in slower)

    x = np.linspace(0,chondrule_radius*1e6,100)
    fig,ax=plt.subplots(figsize=figsize)

    # Normalized F/Cl ratio: (F_final/F_initial) / (Cl_final/Cl_initial)
    ax.plot(x,(F_comp_final/F_initial)/(Cl_comp_final/Cl_initial),label='F/Cl')
    # Normalized Br/Cl ratio
    ax.plot(x,(Br_comp_final/Br_initial)/(Cl_comp_final/Cl_initial),label='Br/Cl',color='C2')

    ax.set_ylabel('Ratio (Initial Normalised)')
    ax.set_xlabel(f'Distance from Centre ($\u03BCm$)')
    ax.set_yscale('log')
    #ax.set_ylim(0.5,50)
    ax.axhline(y=1,linestyle='--',color='k')  # Ratio = 1 means no fractionation
    ax.legend()

    print("✓ Fractionation plots generated")
