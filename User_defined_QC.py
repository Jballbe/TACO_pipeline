#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:27:13 2024

@author: julienballbe
"""
from Sweep_QC_analysis import requires_params
import numpy as np

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
    
    
    
If a Low pass filtered potential of current trace is needed, the parameter name should be written as follow:
    filtered_a_b_c_d
    With :
        a : either "potential" or "current" , indicating if it requires filtered potential trace or filtered current trace
        b : the cutting frequency in kHz as float. Indicate it as 1_0 (1kHz) or 2_3 (2.3kHz) as "." will not be recognized properly
        c : an integer indicating the filter order
        d : either 0 or 1, indicating if the filter should be a zero phase filter (1) or not (0)
    example:
        filtered_potential_5_5_4_0
            Requires potential trace filtered by a non-zero phase Low pass 4th order Butterworth Filter with a cutting frequency of 5.5 kHz
        filtered_current_1_0_2_1
            Requires current trace filtered by a zero phase Low pass 2nd order Butterworth Filter with a cutting frequency of 1 kHz
    
'''


@requires_params('filtered_potential_5_5_4_0', 'Raw_potential_mV_trace')
def function_name(filtered_potential_5_5_4_0, Raw_potential_mV_trace):
    max_voltage_filtered = np.nanmax(filtered_potential_5_5_4_0)
    max_voltage = np.nanmax(Raw_potential_mV_trace)
    Obs = "Is filtering working"
    if max_voltage_filtered == max_voltage:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC

@requires_params("Raw_potentFial_mV_trace")
def check_upper_bound_voltage_trace(Raw_potential_mV_trace):
    Obs = "Mean Voltage Trace <= 10mV"
    mean_value = np.nanmean(Raw_potential_mV_trace)
    if mean_value >= -10:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC

@requires_params('Holding_current_pA')
def check_upper_bound_current_trace(Holding_current_pA):
    Obs = "Mean Current Trace <= 5000 pA"
    mean_value = np.nanmean(Holding_current_pA)
    if mean_value >= 5000:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC


@requires_params("Raw_potential_mV_trace")
def check_upper_bound_amplitude_voltage_trace(Raw_potential_mV_trace):
    Obs = "Amplitude Voltage Trace <= 190 mV"
    min_voltage = np.nanmin(Raw_potential_mV_trace)
    max_voltage = np.nanmax(Raw_potential_mV_trace)
    amplitude = max_voltage-min_voltage
    if amplitude >= 190.:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC

@requires_params('Holding_current_pA')
def check_upper_bound_amplitude_current_trace(Holding_current_pA):
    Obs = "Amplitude Current Trace <= 5000 pA"
    min_voltage = np.nanmin(Holding_current_pA)
    max_voltage = np.nanmax(Holding_current_pA)
    amplitude = max_voltage-min_voltage
    if amplitude >= 5000:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC


@requires_params('Bridge_Error_GOhms')
def check_bridge_error_upper_bound(Bridge_Error_GOhms):
    Obs = "Bridge error <= 20 MOhms"
    
    if Bridge_Error_GOhms >= 20:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC

@requires_params('Bridge_Error_GOhms', "Cell_Input_Resistance_GOhms")
def check_bridge_Error_relative_to_RIn(Cell_Input_Resistance_GOhms, Bridge_Error_GOhms):
    Obs = "BE ≤ 15% * RIn"
    if Bridge_Error_GOhms/Cell_Input_Resistance_GOhms > 0.15:
        passed_QC = False
    else:
        passed_QC = True
    return Obs, passed_QC   

























        
