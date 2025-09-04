# Senior Honours Project Semester 1 2025

## Overview

The goal of the project is to test if importance sampling can be effective in speeding up sampling over a large collection of alternative realizations of the number density n(z) in a weak lensing analysis.

## Step List

### Setup
- Install the CosmoSIS software on a laptop or school computer for testing, and on the Cuillin computer for larger runs.
- Dowload the n(z) that Joe's PhD student Yun-Hao has generated

### Baseline
- Launch a run on the Cuillin cluster of a baseline analysis that JZ will provide.
- Make contour plots using the cosmosis-postprocess tool.

### Increasing the temperature
- Add a module to the pipeline that increases the "temperature" of the analysis, modifying the log likelihood by logL -> logL / T.
- Run an analysis with that new module on Cuillin
- Make contour plots showing both this new analysis and the old one; we expect the new one's contours to be broader.

### Importance sampling
- Set up a collection of importance sample runs each of which takes the high-temperature chain as input and importance samples it with the same pipeline but the n(z) changed to a different realization.
- See if the sampling has by looking at the standard deviations of the importance weights.



# Setup steps
Install the code following the instructions here:
https://cosmosis.readthedocs.io/en/latest/intro/installation.html#conda-forge-from-scratch



## Joe's notes to himself

- do cosmic shear alone
- fix baryons, shear calibration
- nautilus analysis
- set 
- just seven parameters; should be fast. 
    omega_m
    h0
    omega_b
    n_s
    sigma_8
    A_1
    eta
