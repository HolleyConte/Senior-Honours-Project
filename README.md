# Senior Honours Project Semester 1 2025

## Overview

The goal of the project is to test if importance sampling can be effective in speeding up sampling over a large collection of alternative realizations of the number density n(z) in a weak lensing analysis.

Here is an overview of the different phases of the project. I will gradually add the sections below with more detail.

- Getting set up
- Running the baseline analysis
- Increasing the temperature
- Importance sampling
- Combining results

# Getting set up

In these first steps we can get set up with the code.

## Installing on your laptop

First clone this repository using this command in your choice of working directory:
    git clone https://github.com/holleyconte/Senior-Honours-Project

Then install the code from the terminal by using "cd" to get to a directory to the directory where it was cloned, and then running "./install.sh". It may not work, but just let me know if so! 

## Connecting to Cuillin and installing there

You probably need to be on the Uni VPN for it to work, or inside the department. You do this using the command:
    ssh USERNAME@cuillin.roe.ac.uk
where USERNAME is the user name that Eric emailed you. It will ask for the password he sent you.

You can install on cuillin using the same commands as on your laptop.

## Getting the data file

My PhD student Yun-Hao Zhang has passed on a data file containing  500,000 samples of the redshift distributions. This is way more than we need! You can download the file here:

https://www.dropbox.com/scl/fi/9usiawwilmv06rkxhbo6s/ENSEMBLE_Y1.hdf5.zip?rlkey=o96o2l7jr7xc41tm5zi5l1ah0&dl=1

You will need to download it to your laptop first and then use `scp` to copy it to Cuillin later.

Once you've downloaded it then unzip it in the data directory.



# Running the baseline analysis

In this analysis we won't do anything special, just run a basic analysis with a fixed n(z).

## Running on your laptop

Run these commands to run a baseline test of the code once you've installed it, from inside the cloned directory:

    source ./env/bin/activate
    source cosmosis-configure
    cd baseline
    cosmosis baseline.ini

That will take about six hours to run! If you want to stop press ctrl-c. Instead we can run it on the cuillin machine much faster without tying up your computer.

## Examining the analysis

Have a look at the file `params.ini` in the baseline directory. It describes the pipeline that is being run. It also has some hints about what we will be doing later.


## Running on cuillin

Don't run large programs directly on the Cuillin login node that you first connect to!  Instead you'll launch special scripts that will send a job to one of many separate "nodes".

You can do this by going into the "baseline" directory and running the command:
    sbatch baseline.sub

For testing, you can also use this command to connect to a "first-come first-served" node:

    ssh fcfs9

You'll need to use your password again. Once that's done you will be on another machine where it is allowed to run programs.

## Making plots

Use the `cosmosis-postproces` command once the chain is finished to make plots.

# Increasing the temperature

Now we can increase the temperature of the MCMC and compare to our baseline results.

## Writing the new module

In the `heat` directory there is a new module file called `likelihood_temperature.py`. You will need to modify it to change the temperature of the MCMC. It has two functions:

- the `setup` function is run once, at the start of the entire chain. It reads in options that you set in the `params.ini` file. The object it returns (currently a list of three items) represents all the stored configuration information that it needs to run.
- the `execute` function is run at every single step in the MCMC to calculate the likelihood. The argument `block` contains the results of all the previous pipeline steps. The `config` argument is whatever was returned from `setup`.

Update the module to complete the three steps listed in the comments, and update the parameter file to use your new module.

## Running the high temperature chain

Update the parameter file to:
- save the results in a new output file, 
- include the new heat module in the pipeline
- set the temperature

Then you can run the pipeline again.

## Comparing results on the high temperature chain to the baseline.

Use the `cosmosis-postprocess` command on both the old and new files so we can compare the results.  You can use the `--no-fill` command to make it easier to see.  We expect the contours to be bigger in most of the directions in the new heated version (depending on the temperature you chose).

# Importance sampling

Will fill this in later.
