# Senior Honours Project Semester 1 2025

## Overview

The goal of the project is to test if importance sampling can be effective in speeding up sampling over a large collection of alternative realizations of the number density n(z) in a weak lensing analysis.

Here is an overview of the different phases of the project.

### Setup
- Install the CosmoSIS software on a laptop or school computer for testing, and on the Cuillin computer for larger runs.
- Dowload the n(z) that Joe's PhD student Yun-Hao has generated (not reaady yet - 08/09/25)

### Baseline
- Launch a run on the Cuillin cluster of the baseline analysis - JZ will provide a launcher script template.
- Make contour plots using the cosmosis-postprocess tool.

### Increasing the temperature
- Update the module in the "heat" directory to scale the likelihood with logL -> logL / T
- Run an analysis with that new module activate on Cuillin
- Make contour plots showing both this new analysis and the old one; we expect the new one's contours to be broader.

### Importance sampling
- Set up a collection of importance sample runs each of which takes the high-temperature chain as input and importance samples it with the same pipeline but the n(z) changed to a different realization.
- See if the sampling is valid by looking at the standard deviations of the importance weights.


# Setup steps
Install the code from the terminal using the "install.sh" script. It may not work, but just
let me know if so!


## Joe's notes to himself

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
