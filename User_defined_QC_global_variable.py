#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:27:13 2024

@author: julienballbe
"""



'''
Template file construction for Quality_Criteria_to_apply.py
Each function defined in this file will be applied on each sweep

Each function must be decorated by 
@requires_params(variable_1, variable_2,...)
This decorator ensure that the following function will receive the input variables it needs

Each function parameters must be the same as those entered in @requires_params. The order of appearance doesn't matter
Each function must return 2 object: 
    Obs (str): String defining the test applied
    passed_QC (bool): wether or not the trace should be rejected based on this test

The variables in @requires_params correspond to precise element in the cell or sweep analysis
Here is the exhaustive list of available elements (their type) and: their descritptions

    Bridge_Error_extrapolated (bool): Has the bridge error been determined by extrapolation (True) or directly estimated from the traces(False)
    Bridge_Error_GOhms (float): Bridge Error estimated from the traces
    Cell_Input_Resistance_GOhms (float): Input Resistance of the cell, estimated by averaging the input resistances computed for each sweep
    Cell_Resting_potential_mV (float): Resting potential of the cell, estimated by averaging the resting potentials computed for each sweep
    Cell_Time_constant_ms (float): Time constant of the cell, estimated by averaging the time constants computed for each sweep
    Holding_current_pA (float): Current applied before and after the stimulus application
    Holding_potential_mV (float): Membrane potential resulting of the application of the holding current
    Input_Resistance_GOhms (float): Input Resistance measured in the current sweep
    Raw_current_pA_trace (numpy array of floats) : Trace of the input current applied to the cell
    Raw_potential_mV_trace (numpy array of floats) : Trace of the cell's membrane potential
    Resting_potential_mV (float) : Resting potential of the cell measured in the current sweep
    Sampling_Rate_Hz (float) : Sampling rate of the traces
    SS_potential_mV (float) : Steady state membrane potential, if membrane potential reaches steady state, and do not have spike
    Stim_amp_pA (float) : Difference between the steady state current and the  holding current
    Stim_end_s (float) : Time at which the stimulus current is no longer applied
    Stim_SS_pA (float) : Value of the applied stimulus (between Stim_start_s and Stim_end_s)
    Stim_start_s (float) : Time at which the stimulus current is applied
    Time_constant_ms (float) : Time constant measured in the current sweep
    Time_s_trace (numpy array of floats) :  Time trace of the current sweep
    
    
'''


##### DO NOT REMOVE
import globals_module  # Import the common module
global Bridge_Error_extrapolated, Bridge_Error_GOhms, Cell_Input_Resistance_GOhms, Cell_Resting_potential_mV, Cell_Time_constant_ms, Holding_current_pA, Holding_potential_mV, Input_Resistance_GOhms, Raw_current_pA_trace, Raw_potential_mV_trace, Resting_potential_mV, Sampling_Rate_Hz, SS_potential_mV, Stim_amp_pA, Stim_end_s, Stim_SS_pA, Stim_start_s, Time_constant_ms, Time_s_trace
import numpy as np

Bridge_Error_extrapolated = globals_module.Bridge_Error_extrapolated
Bridge_Error_GOhms = globals_module.Bridge_Error_GOhms
Cell_Input_Resistance_GOhms = globals_module.Cell_Input_Resistance_GOhms
Cell_Resting_potential_mV = globals_module.Cell_Resting_potential_mV
Cell_Time_constant_ms = globals_module.Cell_Time_constant_ms
Holding_current_pA = globals_module.Holding_current_pA
Holding_potential_mV = globals_module.Holding_potential_mV
Input_Resistance_GOhms = globals_module.Input_Resistance_GOhms
Raw_current_pA_trace = globals_module.Raw_current_pA_trace
Raw_potential_mV_trace = globals_module.Raw_potential_mV_trace
Resting_potential_mV = globals_module.Resting_potential_mV
Sampling_Rate_Hz = globals_module.Sampling_Rate_Hz
SS_potential_mV = globals_module.SS_potential_mV
Stim_amp_pA = globals_module.Stim_amp_pA
Stim_end_s = globals_module.Stim_end_s
Stim_SS_pA = globals_module.Stim_SS_pA
Stim_start_s = globals_module.Stim_start_s
Time_constant_ms = globals_module.Time_constant_ms
Time_s_trace = globals_module.Time_s_trace


##### DO NOT REMOVE WHAT IS ABOVE THIS LINE

def check_upper_bound_voltage_trace():
    Obs = "Mean Voltage Trace <= 10mV"
    mean_value = np.nanmean(Raw_potential_mV_trace)
    if mean_value >= -10:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC


def check_upper_bound_current_trace():
    Obs = "Mean Current Trace <= 5000 pA"
    mean_value = np.nanmean(Holding_current_pA)
    if mean_value >= 5000:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC



def check_upper_bound_amplitude_voltage_trace():
    Obs = "Amplitude Voltage Trace <= 190 mV"
    min_voltage = np.nanmin(Raw_potential_mV_trace)
    max_voltage = np.nanmax(Raw_potential_mV_trace)
    amplitude = max_voltage-min_voltage
    if amplitude >= 190.:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC


def check_upper_bound_amplitude_current_trace():
    Obs = "Amplitude Current Trace <= 5000 pA"
    min_voltage = np.nanmin(Holding_current_pA)
    max_voltage = np.nanmax(Holding_current_pA)
    amplitude = max_voltage-min_voltage
    if amplitude >= 5000:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC

def check_input_resistance_is_positive():
    Obs = "Input resistance > 0 GOhms"
    if Input_Resistance_GOhms <= 0.:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC






























        
