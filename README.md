# Cluster rotation with Intrinsic Scatter Using MCMC

This repository contains a Python script to model the rotation of a cluster. For this we fit a sine function to data with uncertainties using Markov Chain Monte Carlo (MCMC) via the emcee package. The script also models the intrinsic scatter around the sine function.

## Features

Fits a sine function to data with known uncertainties.
Models intrinsic scatter in the data.
Uses MCMC for parameter estimation.
Supports parallel execution for faster computation.
Generates diagnostic plots including a corner plot and residual plots.

## Requirements

Python 3.x
NumPy
pandas
emcee
corner
matplotlib
