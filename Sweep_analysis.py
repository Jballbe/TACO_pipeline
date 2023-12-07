#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:22:09 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
import plotnine as p9
import scipy
from lmfit.models import   Model, QuadraticModel, ExponentialModel
from lmfit import Parameters
from sklearn.metrics import mean_squared_error
import importlib

import Ordinary_functions as ordifunc
import Spike_analysis as sp_an





def sweep_analysis_processing(cell_Full_TVC_table, cell_stim_time_table, nb_workers):
    '''
    Create cell_sweep_info_table using parallel processing.
    For each sweep contained in cell_Full_TVC_table, extract different recording information (Bridge Error, sampling frequency),
    stimulus and voltage trace information (holding current, resting voltage...),
    and trace-computed linear properties if possible (membrane time constant, input resistance)

    Parameters
    ----------
    cell_Full_TVC_table : pd.DataFrame
        2 columns DataFrame, cotaining in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time, Current and Potential Traces
        
    cell_stim_time_table : pd.DataFrame
        DataFrame, containing foir each Sweep the corresponding stimulus start and end times.
        
    nb_workers : int
        Number of CPU cores to use for parallel processing.

    Returns
    -------
    cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).

    '''
    cell_sweep_info_table = pd.DataFrame(columns=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_SS_pA', 'Holding_current_pA','Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', 'Sampling_Rate_Hz'])
    cell_Full_TVC_table.index.name = 'Index'
    cell_Full_TVC_table = cell_Full_TVC_table.sort_values(by=['Sweep'])
    cell_stim_time_table.index.name = 'Index'
    cell_stim_time_table = cell_stim_time_table.sort_values(by=['Sweep'])
    sweep_list = np.array(cell_Full_TVC_table['Sweep'], dtype=str)


    TVC_list = []
    
    for x in sweep_list:
        current_TVC = ordifunc.get_filtered_TVC_table(cell_Full_TVC_table, x, do_filter=True, filter=5., do_plot=False)
        TVC_list.append(current_TVC)
    
    stim_start_time_list = list(cell_stim_time_table.loc[:,'Stim_start_s'])
    stim_end_time_list = list(cell_stim_time_table.loc[:,'Stim_end_s'])
    sweep_info_zip = zip(TVC_list,sweep_list,stim_start_time_list,stim_end_time_list)
    sweep_info_list= list(sweep_info_zip)
    for x in sweep_info_list:
        result = get_sweep_info_parallel(x)
        cell_sweep_info_table = pd.concat([
            cell_sweep_info_table,result], ignore_index=True)
    
    
    
    ### create table of trace data
    
    extrapolated_BE = pd.DataFrame(columns=['Sweep', 'Bridge_Error_GOhms'])
    
    cell_sweep_info_table['Bridge_Error_GOhms']=removeOutliers(
        np.array(cell_sweep_info_table['Bridge_Error_GOhms']), 3)
    
    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'

    
    
    ### Create Linear properties table
    
    Linear_table=pd.DataFrame(columns=['Sweep',"Time_constant_ms", 'Input_Resistance_GOhms', 'Holding_potential_mV','SS_potential_mV','Resting_potential_mV'])
    cell_sweep_info_table = cell_sweep_info_table.sort_values(by=['Sweep'])
    stim_start_time_list = list(cell_sweep_info_table.loc[:,'Stim_start_s'])
    stim_end_time_list = list(cell_sweep_info_table.loc[:,'Stim_end_s'])
    BE_extrapolated_list = list(cell_sweep_info_table.loc[:,'Bridge_Error_extrapolated'])
    BE_list = list(cell_sweep_info_table.loc[:,'Bridge_Error_GOhms'])
    stim_amp_list = list(cell_sweep_info_table.loc[:,'Stim_amp_pA'])
    
    sweep_info_zip = zip(TVC_list,sweep_list,stim_start_time_list,stim_end_time_list, BE_extrapolated_list, BE_list, stim_amp_list)
    sweep_info_list= list(sweep_info_zip)
    
    for x in sweep_info_list:
        result = get_sweep_linear_properties(x)
        Linear_table = pd.concat([
            Linear_table,result], ignore_index=True)
    
    
   
    cell_sweep_info_table = cell_sweep_info_table.merge(
        Linear_table, how='inner', on='Sweep')
    
    
    
    ### For sweeps whose Brige Error couldn't be estimated, extrapolate from neighboring traces (protocol_wise)


    for current_Protocol in cell_sweep_info_table['Protocol_id'].unique():

        reduced_cell_sweep_info_table = cell_sweep_info_table[
            cell_sweep_info_table['Protocol_id'] == current_Protocol]
        reduced_cell_sweep_info_table=reduced_cell_sweep_info_table.astype({"Trace_id":"int"})
        reduced_cell_sweep_info_table=reduced_cell_sweep_info_table.sort_values(by=['Trace_id'])
        BE_array = np.array(
            reduced_cell_sweep_info_table['Bridge_Error_GOhms'])
        sweep_array = np.array(reduced_cell_sweep_info_table['Sweep'])

        nan_ind_BE = np.isnan(BE_array)

        x = np.arange(len(BE_array))
        if False in nan_ind_BE:

            BE_array[nan_ind_BE] = np.interp(
                x[nan_ind_BE], x[~nan_ind_BE], BE_array[~nan_ind_BE])

        extrapolated_BE_Series = pd.Series(BE_array)
        sweep_array = pd.Series(sweep_array)
        extrapolated_BE_Series = pd.DataFrame(
            pd.concat([sweep_array, extrapolated_BE_Series], axis=1))
        extrapolated_BE_Series.columns = ['Sweep', 'Bridge_Error_GOhms']
        extrapolated_BE = pd.concat(
            [extrapolated_BE,extrapolated_BE_Series], ignore_index=True)


    cell_sweep_info_table.pop('Bridge_Error_GOhms')

    cell_sweep_info_table = cell_sweep_info_table.merge(
        extrapolated_BE, how='inner', on='Sweep')
    


    cell_sweep_info_table = cell_sweep_info_table.loc[:, ['Sweep',
                                                          "Protocol_id",
                                                          'Trace_id',
                                                          'Stim_amp_pA',
                                                          'Stim_SS_pA',
                                                          'Holding_current_pA',
                                                          'Stim_start_s',
                                                          'Stim_end_s',
                                                          'Bridge_Error_GOhms',
                                                          'Bridge_Error_extrapolated',
                                                          'Time_constant_ms',
                                                          'Input_Resistance_GOhms',
                                                          'Holding_potential_mV',
                                                          "SS_potential_mV",
                                                          "Resting_potential_mV",
                                                          'Sampling_Rate_Hz']]

    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'
    convert_dict = {'Sweep': str,
                    'Protocol_id': str,
                    'Trace_id': int,
                    'Stim_amp_pA': float,
                    'Holding_current_pA':float,
                    'Stim_SS_pA':float,
                    'Stim_start_s': float,
                    'Stim_end_s': float,
                    'Bridge_Error_GOhms': float,
                    'Bridge_Error_extrapolated': bool,
                    'Time_constant_ms': float,
                    'Input_Resistance_GOhms': float,
                    'Holding_potential_mV':float,
                    "SS_potential_mV": float,
                    "Resting_potential_mV":float,
                    'Sampling_Rate_Hz': float
                    }

    cell_sweep_info_table = cell_sweep_info_table.astype(convert_dict)
    
    ### Remove extremes outliers of Time_constant and Input_Resistance (Q1/Q3 ± 3IQR)
    remove_outier_columns = ['Time_constant_ms', 'Input_Resistance_GOhms']

    for current_column in remove_outier_columns:
        cell_sweep_info_table.loc[:, current_column] = removeOutliers(
            np.array(cell_sweep_info_table.loc[:, current_column]), 3)

    
    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'

   
    return cell_sweep_info_table



def get_sweep_info_parallel(sweep_info_list):
    '''
    Function to be used in parallel to extract or compute different sweep related information
    
    Parameters
    ----------
    sweep_info_list : List
        List of parameters required by the function.

    Returns
    -------
    output_line : pd.DataFrame
        One Row DataFrame containing different sweep related information.

    '''

    TVC_table, current_sweep, stim_start_time, stim_end_time = sweep_info_list
    
        
    if len(str(current_sweep).split("_"))==1:
        Protocol_id = str(1)
        Trace_id = current_sweep
        
    elif len(str(current_sweep).split("_")) ==2:
        Protocol_id, Trace_id = str(current_sweep).split("_")
        
    else :
        Protocol_id, Trace_id = str(current_sweep).split("_")[-2:]

        
    Protocol_id=str(Protocol_id)
        
    Holding_current,SS_current = fit_stimulus_trace(
        TVC_table, stim_start_time, stim_end_time,do_plot=False)[:2]
    
        
        
    if np.abs((Holding_current-SS_current))>=20.:
        Bridge_Error= estimate_bridge_error(
            TVC_table, SS_current, stim_start_time, stim_end_time, do_plot=False)
        
    else:
        Bridge_Error=np.nan
    
    if np.isnan(Bridge_Error):
        BE_extrapolated = True
    else:
        BE_extrapolated = False
        
    time_array = np.array(TVC_table.loc[:,"Time_s"])
    sampling_rate = 1/(time_array[1]-time_array[0])
    
    
    stim_amp = SS_current - Holding_current
    output_line = pd.DataFrame([str(current_sweep),
                             str(Protocol_id),
                             Trace_id,
                             SS_current,
                             Holding_current,
                             stim_amp,
                             stim_start_time,
                             stim_end_time,
                             Bridge_Error,
                             BE_extrapolated,
                             sampling_rate]).T
    
    
    output_line.columns=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_SS_pA', 'Holding_current_pA','Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', 'Sampling_Rate_Hz']
    return output_line

def get_sweep_linear_properties(sweep_info_list):
    '''
    Function to be used in parallel to compute different sweep based linear properties 
    
    Parameters
    ----------
    sweep_info_list : List
        List of parameters required by the function.

    Returns
    -------
    sweep_line : pd.DataFrame
        One Row DataFrame containing different sweep based linear properties 


    '''
    
    
    TVC_table, sweep, stim_start, stim_end, BE_extrapolated, BE, stim_amp = sweep_info_list


    
    sub_test_TVC=TVC_table[TVC_table['Time_s']<=stim_start-.005].copy()
    sub_test_TVC=sub_test_TVC=sub_test_TVC[sub_test_TVC['Time_s']>=(stim_start-.200)]
    
    CV_pre_stim=np.std(np.array(sub_test_TVC['Membrane_potential_mV']))/np.mean(np.array(sub_test_TVC['Membrane_potential_mV']))

    
    
    
    voltage_spike_table = sp_an.identify_spike(np.array(TVC_table.Membrane_potential_mV),
                                    np.array(TVC_table.Time_s),
                                    np.array(TVC_table.Input_current_pA),
                                    stim_start,
                                    (stim_start+stim_end)/2,do_plot=False)
    if len(voltage_spike_table['Peak']) != 0:
        is_spike = True
    else:
        is_spike = False
    
    sub_test_TVC=TVC_table[TVC_table['Time_s']<=(stim_end)].copy()
    sub_test_TVC=sub_test_TVC[sub_test_TVC['Time_s']>=((stim_start+stim_end)/2)]
    
    CV_second_half=np.std(np.array(sub_test_TVC['Membrane_potential_mV']))/np.mean(np.array(sub_test_TVC['Membrane_potential_mV']))
    
    

    if (is_spike == True or
        np.abs(stim_amp)<2.0 or
        BE_extrapolated ==True or
        np.abs(CV_pre_stim) > 0.01 or 
        np.abs(CV_second_half) > 0.01):
        

        SS_potential = np.nan
        resting_potential = np.nan
        holding_potential = np.nan
        R_in = np.nan
        time_cst = np.nan
        
    
    else:

        best_A,best_tau,SS_potential,holding_potential,NRMSE = fit_membrane_time_cst(TVC_table,
                                                                         stim_start,
                                                                         (stim_start+.300),do_plot=False)

        if NRMSE > 0.3:
            SS_potential = np.nan
            resting_potential = np.nan
            holding_potential = np.nan
            R_in = np.nan
            time_cst = np.nan
            
        else:
            
            
            R_in=((SS_potential-holding_potential)/stim_amp)-BE
            
            time_cst=best_tau*1e3 #convert s to ms
            resting_potential = holding_potential - (R_in+BE)*stim_amp
            
    sweep_line=pd.DataFrame([str(sweep),time_cst,R_in,holding_potential,SS_potential,resting_potential]).T        
    sweep_line.columns=['Sweep',"Time_constant_ms", 'Input_Resistance_GOhms', 'Holding_potential_mV','SS_potential_mV','Resting_potential_mV']
    
    return sweep_line

def removeOutliers(x, outlierConstant=3):
    '''
    Remove from an array outliers defined by values x<Q1-nIQR or x>Q3+nIQR
    with x a value in array and n the outlierConstant

    Parameters
    ----------
    x : np.array or list
        Values from whihc remove outliers.
        
    outlierConstant : int, optional
        The number of InterQuartile Range to define outleirs. The default is 3.

    Returns
    -------
    np.array
        x array without outliers.

    '''
    a = np.array(x)

    
    upper_quartile = np.nanpercentile(a, 75)
    lower_quartile = np.nanpercentile(a, 25)

    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
        else:
            resultList.append(np.nan)

    return np.array(resultList)


def fit_stimulus_trace(TVC_table_original,stim_start,stim_end,do_plot=False):
    '''
    Fit double Heaviside function to the time-varying trace representing the input current.
    The Trace is low-pass filtered at 1kHz, and the fit is weighted to the inverse of the time derivative of the trace

    Parameters
    ----------
    TVC_table_original : Tpd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    stim_start : Float
        Stimulus start time.
        
    stim_end : Float
        Stimulus end time.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_baseline ,best_stim_amp, best_stim_start, best_stim_end : Float
        Fitting results.
        
    NRMSE_double_Heaviside : Float
        Godness of fit.

    '''
    
    TVC_table=TVC_table_original.copy()
    stim_table=TVC_table.loc[:,['Time_s','Input_current_pA']].copy()
    stim_table = stim_table.reset_index(drop=True)
    x_data=stim_table.loc[:,"Time_s"]
    index_stim_start=next(x for x, val in enumerate(x_data[0:]) if val >= stim_start )
    index_stim_end=next(x for x, val in enumerate(x_data[0:]) if val >= (stim_end) )
    
    
    
    stim_table['Input_current_pA']=np.array(ordifunc.filter_trace(stim_table['Input_current_pA'],
                                                                        stim_table['Time_s'],
                                                                        filter=1.,
                                                                        do_plot=False))
    
    first_current_derivative=ordifunc.get_derivative(np.array(stim_table["Input_current_pA"]),np.array(stim_table["Time_s"]))
    first_current_derivative=np.insert(first_current_derivative,0,first_current_derivative[0])
    
    stim_table['Filtered_Stimulus_trace_derivative_pA/ms']=np.array(first_current_derivative)
    
    
    
    before_stim_start_table=stim_table[stim_table['Time_s']<stim_start-.01]
    baseline_current=np.mean(before_stim_start_table['Input_current_pA'])
    
    during_stimulus_table=stim_table[stim_table['Time_s']<stim_end]
    during_stimulus_table=during_stimulus_table[during_stimulus_table['Time_s']>stim_start]
    estimate_stim_amp=np.median(during_stimulus_table.loc[:,'Input_current_pA'])
    
    
    
    y_data=stim_table.loc[:,"Input_current_pA"]
    stim_table['weight']=np.abs(1/stim_table['Filtered_Stimulus_trace_derivative_pA/ms'])**2
    

    
    
    weight=stim_table.loc[:,"weight"]

    if np.isinf(weight).sum!=0:
        
        if np.isinf(weight).sum() >= (len(weight)/2): # if inf values represent more than half the values
            weight=np.ones(len(weight))

        else: # otherwise replace inf values by the maximum value non inf

            max_weight_without_inf=np.nanmax(weight[weight != np.inf])
            weight.replace([np.inf], max_weight_without_inf, inplace=True)
   
    weight/=np.nanmax(weight) # normalize the weigth to the maximum weight 

    double_step_model=Model(Double_Heaviside_function)
    double_step_model_parameters=Parameters()
    double_step_model_parameters.add('stim_start',value=stim_start,vary=False)
    double_step_model_parameters.add('stim_end',value=stim_end,vary=False)
    double_step_model_parameters.add('baseline',value=baseline_current)
    double_step_model_parameters.add('stim_amplitude',value=estimate_stim_amp)


    #return stim_table
    double_step_out=double_step_model.fit(y_data,double_step_model_parameters,x=x_data,weights=weight)
    

    
    best_baseline=double_step_out.best_values['baseline']
    best_stim_amp=double_step_out.best_values['stim_amplitude']
    best_stim_start=double_step_out.best_values['stim_start']
    best_stim_end=double_step_out.best_values['stim_end']
    
    NRMSE_double_Heaviside=mean_squared_error(y_data.iloc[index_stim_start:index_stim_end], Double_Heaviside_function(x_data, best_stim_start, best_stim_end, best_baseline, best_stim_amp)[index_stim_start:index_stim_end],squared=(False))/(best_stim_amp)

    if do_plot:
        computed_y_data=pd.Series(Double_Heaviside_function(x_data, best_stim_start,best_stim_end, best_baseline, best_stim_amp))
        model_table=pd.DataFrame({'Time_s' :x_data,
                                  'Input_current_pA': computed_y_data})#np.column_stack((x_data,computed_y_data)),columns=['Time_s ',"Stim_amp_pA"])
        
        
        TVC_table['Legend']='Original_Data'
        
        stim_table['Legend']='Filtered_fitted_Data'
        model_table['Legend']='Fit'
        

        data_table = pd.concat([TVC_table,stim_table],ignore_index=True)
        data_table = pd.concat([data_table,model_table],ignore_table = True)
        
        my_plot = p9.ggplot()
        my_plot += p9.geom_line(TVC_table, p9.aes(x='Time_s',y='Input_current_pA'),colour = 'black')
        my_plot += p9.geom_line(stim_table,p9.aes(x='Time_s',y='Input_current_pA'),colour = 'red')
        my_plot += p9.geom_line(model_table,p9.aes(x='Time_s',y='Input_current_pA'),colour='blue')
        my_plot += p9.xlab(str("Time_s"))
        my_plot += p9.xlim((stim_start-0.05), (stim_end+0.05))
        
        print(my_plot)
    return best_baseline,best_stim_amp,best_stim_start,best_stim_end,NRMSE_double_Heaviside


def estimate_bridge_error(original_TVC_table,stim_amplitude,stim_start_time,stim_end_time,do_plot=False):
    '''
    A posteriori estimation of bridge error, by estimating 'very fast' membrane voltage transient around stimulus start and end

    Parameters
    ----------
    original_TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    stim_amplitude : float
        Value of Stimulus amplitude (between stimulus start and end).
        
    stim_start_time : float
        Stimulus start time.
        
    stim_end_time : float
        Stimulus end time.
        
    do_plot : TYPE, optional
        If True, returns Bridge Error Plots. The default is False.

    Returns
    -------
    if do_plot == True: return plots
    
    if do_plot == False: return Bridge Error in GOhms

    '''
    
    
    TVC_table=original_TVC_table.reset_index(drop=True).copy()
    point_table=pd.DataFrame(columns=["Time_s","Membrane_potential_mV","Feature"])
    
    
    start_time_index = np.argmin(abs(np.array(TVC_table['Time_s']) - stim_start_time))
    
    stimulus_baseline=np.mean(TVC_table.loc[:(start_time_index-1),'Input_current_pA'])
    at_least_one_Bridge_Error=False
    
    ########################################################################################
    #Stimulus_start_window
    
    #Consider a 40msec windows centered on the specified stimulus start so +/- 20 msec and compute spline fit for current and voltage
    stimulus_start_table=TVC_table[TVC_table["Time_s"]<(stim_start_time+.020)]
    stimulus_start_table=stimulus_start_table[stimulus_start_table['Time_s']>(stim_start_time-.020)]
   
    
    time_array=np.array(stimulus_start_table.loc[:,'Time_s'])
    
    
    current_start_trace_spline=scipy.interpolate.UnivariateSpline(np.array(stimulus_start_table.loc[:,'Time_s']),np.array(stimulus_start_table.loc[:,'Input_current_pA']))
    current_start_trace_spline.set_smoothing_factor(.999)
    current_trace_filtered=current_start_trace_spline(time_array)
    current_trace_filtered_derivative=ordifunc.get_derivative( current_trace_filtered , time_array )
    current_trace_filtered_derivative=np.insert(current_trace_filtered_derivative,0,current_trace_filtered_derivative[0])
    
    
    stimulus_start_table['Input_current_derivative_pA/s']=current_trace_filtered_derivative
    
    
    #determine T_trans
    T_trans_table=stimulus_start_table[stimulus_start_table["Time_s"]<(stim_start_time+.002)]
    T_trans_table=T_trans_table[T_trans_table['Time_s']>(stim_start_time-.002)]
    
    T_Trans_current_derivative=np.array(T_trans_table['Input_current_derivative_pA/s'])
    
    if stim_amplitude <= stimulus_baseline:
        
        max_abs_dI_dt_index = np.nanargmin(T_Trans_current_derivative)
        
        
    elif stim_amplitude>stimulus_baseline:

        max_abs_dI_dt_index = np.nanargmax(T_Trans_current_derivative)
        
        

    T_trans=np.array(T_trans_table.loc[:,'Time_s'])[max_abs_dI_dt_index]
    
    
    ## determine Pre_Trans_Median and Post_Trans_median; 
    #Median current value in a window between T_trans-0.006 and T_trans-0.002 (and vice-versa)
    Pre_Trans_Median_table=stimulus_start_table[stimulus_start_table['Time_s']<=T_trans-.002]
    Pre_Trans_Median_table=Pre_Trans_Median_table[Pre_Trans_Median_table['Time_s']>=T_trans-.006]
    Pre_Trans_Median=np.median(Pre_Trans_Median_table['Input_current_pA'])
    
    Post_Trans_Median_table=stimulus_start_table[stimulus_start_table['Time_s']>=T_trans+.002]
    Post_Trans_Median_table=Post_Trans_Median_table[Post_Trans_Median_table['Time_s']<=T_trans+.006]
    Post_Trans_Median=np.median(Post_Trans_Median_table['Input_current_pA'])
    
    I_step=Post_Trans_Median-Pre_Trans_Median
    
    #determine DT_trans, the time scale of response transient to avoid contaminated period

    ## get time of the max (min) dV/dt, for a positive (negative) Istep, respectively
    DT_Trans_table=stimulus_start_table[stimulus_start_table['Time_s']>=T_trans].reset_index(drop=True).copy()


    DT_Trans_table=DT_Trans_table[DT_Trans_table['Time_s']<=(T_trans+.003)]
    voltage_derivative_trace=np.array(DT_Trans_table.loc[:,'Potential_first_time_derivative_mV/s'])
    
    if stim_amplitude <= stimulus_baseline:

        min_dV_dt_index=np.nanargmin(voltage_derivative_trace)
        time_max_abs_dV_dt=np.array(DT_Trans_table['Time_s'])[min_dV_dt_index]
        DT_trans_line=pd.DataFrame([time_max_abs_dV_dt,np.array(DT_Trans_table.loc[:,'Membrane_potential_mV'])[min_dV_dt_index],'DT_trans_min_dV_dt']).T
                                
        
    elif stim_amplitude > stimulus_baseline:

        max_dV_dt_index=np.nanargmax(voltage_derivative_trace)
        time_max_abs_dV_dt=np.array(DT_Trans_table['Time_s'])[max_dV_dt_index]
        DT_trans_line=pd.DataFrame([time_max_abs_dV_dt,np.array(DT_Trans_table.loc[:,'Membrane_potential_mV'])[max_dV_dt_index],'DT_trans_max_dV_dt']).T

    DT_trans_line.columns=["Time_s","Membrane_potential_mV","Feature"]
    point_table=pd.concat([point_table,DT_trans_line],ignore_index=True)
    
    
    
    
    ##get the time of the last positive(negative) zero crossing dV/dt for a positive (negative) Istep, respectively
    if stim_amplitude <= stimulus_baseline:

        for reversed_index,elt in enumerate(voltage_derivative_trace[::-1][1:]):
            previous_elt=voltage_derivative_trace[::-1][reversed_index]
            if previous_elt >=0 and elt<0:
                last_zero_crossing_index=len(voltage_derivative_trace)-1-reversed_index
                break
            else:
                last_zero_crossing_index=0
            
    elif stim_amplitude > stimulus_baseline:

        for reversed_index,elt in enumerate(voltage_derivative_trace[::-1][1:]):
            previous_elt=voltage_derivative_trace[::-1][reversed_index]
            if previous_elt <=0 and elt>0:
                last_zero_crossing_index=len(voltage_derivative_trace)-1-reversed_index
                break
            else:
                last_zero_crossing_index=0
    
    time_zero_crossing=np.array(DT_Trans_table['Time_s'])[last_zero_crossing_index]
    
    
    DT_trans_line=pd.DataFrame([time_zero_crossing,np.array(DT_Trans_table.loc[:,'Membrane_potential_mV'])[last_zero_crossing_index],'DT_trans_zero_crossing_dV_dt']).T
    DT_trans_line.columns=["Time_s","Membrane_potential_mV","Feature"]
    point_table=pd.concat([point_table,DT_trans_line],ignore_index=True)
    
    
    
    DT_trans=max(.002,abs(T_trans-time_max_abs_dV_dt),abs(T_trans-time_zero_crossing))

    
    #Get fit table : [T_seg_start; T_trans+T_seg_duration ]
    
    T_seg_start = T_trans+max(.002,2.5*DT_trans)
    T_seg_duration = max(.005,2.5*DT_trans)
    

    # Get a table (fit_table) of Time-Potential values between T_seg_start and T_seg_start+T_seg_duration
    fit_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_seg_start+T_seg_duration)]
    fit_table=fit_table[fit_table['Time_s']>=(T_seg_start)]
    test_spike_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_seg_start+T_seg_duration)]
    test_spike_table=stimulus_start_table[stimulus_start_table["Time_s"]>=T_trans-.01]
    
    
    
    # voltage_spike_table=identify_spike_shorten(np.array(test_spike_table.Membrane_potential_mV),
    #                                np.array(test_spike_table.Time_s),
    #                                np.array(test_spike_table.Input_current_pA),
    #                                np.array(test_spike_table.Time_s)[0],
    #                                np.array(test_spike_table.Time_s)[-1],do_plot=False)
    
    voltage_spike_table = sp_an.identify_spike(np.array(test_spike_table.Membrane_potential_mV),
                                   np.array(test_spike_table.Time_s),
                                   np.array(test_spike_table.Input_current_pA),
                                   np.array(test_spike_table.Time_s)[0],
                                   np.array(test_spike_table.Time_s)[-1],do_plot=False)
    if len(voltage_spike_table['Peak']) != 0:
        is_spike = True
    else:
        is_spike = False
    
    PreTrans_Est_table_to_fit=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_trans-.001)]
    PreTrans_Est_table_to_fit=PreTrans_Est_table_to_fit[PreTrans_Est_table_to_fit['Time_s']>=(T_trans-.01)]
    
    # #fit the voltage values to a 2nd order polynomial
    a_bis,b_bis,c_bis,RMSE_poly_bis = fit_second_order_poly(PreTrans_Est_table_to_fit,do_plot=False)
    
    # # Use the PreTrans_Est_table_to_fit fit to estimate the voltage value at T_Trans = Pre_Trans_Final
    Pre_Trans_Final=a_bis*((T_trans)**2)+b_bis*T_trans+c_bis
    
    
    PreTrans_Est_table_to_fit['Legend']='Pre_Trans_fit'
    
    if is_spike == True:

         Bridge_Error_stim_start=np.nan
         
    else:
        #Use fit_table to fit either a 2nd order poly or Exponential to the membrane voltage trace
        a,b,c,RMSE_poly = fit_second_order_poly(fit_table,do_plot=False)
       
        A,tau,RMSE_expo = fit_exponential_BE(fit_table,stim_start = False,do_plot=False)
        
        if np.isnan(RMSE_poly) and np.isnan(RMSE_expo):
            Bridge_Error_stim_start = np.nan
           
        else:
           
            if np.isnan(RMSE_poly):
                RMSE_poly=np.inf
            elif np.isnan(RMSE_expo):
                
                RMSE_expo=np.inf   
            
            # Get a table between [T_trans and T_trans+T_seg_duration ] to estimate backward to T_Trans from the fit
            Post_trans_Init_Value_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_seg_start+T_seg_duration)]
            Post_trans_Init_Value_table=Post_trans_Init_Value_table[Post_trans_Init_Value_table['Time_s']>=T_trans]
            
            #Use the best fit to extrapolate the membrane voltage fit backward to T_Trans
            if RMSE_poly<=RMSE_expo:
                backward_prediction=a*((Post_trans_Init_Value_table.loc[:,'Time_s'])**2)+b*Post_trans_Init_Value_table.loc[:,'Time_s']+c
                Post_trans_init_value=a*((T_trans)**2)+b*T_trans+c
                
                backward_table=pd.DataFrame(np.column_stack((Post_trans_Init_Value_table.loc[:,'Time_s'],backward_prediction)),columns=["Time_s","Membrane_potential_mV"])
                backward_table['Legend']='Post_trans_init_value'
                
            elif RMSE_poly>RMSE_expo:
                backward_prediction=A*(np.exp(-(Post_trans_Init_Value_table.loc[:,'Time_s'])/tau))
                Post_trans_init_value=A*(np.exp(-(T_trans)/tau))
                backward_table=pd.DataFrame(np.column_stack((Post_trans_Init_Value_table.loc[:,'Time_s'],backward_prediction)),columns=["Time_s","Membrane_potential_mV"])
                backward_table['Legend']='Post_trans_init_value' 
            
            
            
            post_trans_init_value_line=pd.DataFrame([T_trans,Post_trans_init_value,'Post_trans_init_value']).T
            post_trans_init_value_line.columns=["Time_s","Membrane_potential_mV","Feature"]
            point_table=pd.concat([point_table,post_trans_init_value_line],ignore_index=True)
    
            #PreTrans_Est_table_to_fit is a table of Time_voltage values between 1ms and 10ms before T_Trans,
            
          
            
            #Get the same table as PreTrans_Est_table_to_fit, but up to T_Trans, for plotting purpose
            PreTrans_Est_table_to_T_trans_table=stimulus_start_table[stimulus_start_table["Time_s"]<=(T_trans)]
            PreTrans_Est_table_to_T_trans_table=PreTrans_Est_table_to_T_trans_table[PreTrans_Est_table_to_T_trans_table['Time_s']>=(T_trans-.01)]
            Pre_trans_Est_pred=a_bis*((PreTrans_Est_table_to_T_trans_table.loc[:,"Time_s"])**2)+b_bis*PreTrans_Est_table_to_T_trans_table.loc[:,"Time_s"]+c_bis
            #PreTrans_Est_table=PreTrans_Est_table_to_fit.loc[:,["Time_s","Membrane_voltage_mV"]]
            PreTrans_Est_table=pd.DataFrame(np.column_stack((PreTrans_Est_table_to_T_trans_table.loc[:,"Time_s"],Pre_trans_Est_pred)),columns=["Time_s","Membrane_potential_mV"])
            PreTrans_Est_table['Legend']='PreTrans_Value'
            
            
            Pre_trans_final_value_line=pd.DataFrame([T_trans,Pre_Trans_Final,'PreTrans_Value']).T
            Pre_trans_final_value_line.columns=["Time_s","Membrane_potential_mV","Feature"]
            point_table=pd.concat([point_table,Pre_trans_final_value_line],ignore_index=True)
        
            Bridge_Error_stim_start = (Post_trans_init_value-Pre_Trans_Final)/I_step
            
            at_least_one_Bridge_Error=True
    ########################################################################################
    
    #Stimulus_end_window
    
    
    point_table_end=pd.DataFrame(columns=["Time_s","Membrane_potential_mV","Feature"])
    #Consider a 40msec windows centered on the specified stimulus start so +/- 20 msec and compute spline fit for current and voltage
    stimulus_end_table=TVC_table[TVC_table["Time_s"]<(stim_end_time+.020)]
    stimulus_end_table=stimulus_end_table[stimulus_end_table['Time_s']>(stim_end_time-.020)]
    time_end_array=np.array(stimulus_end_table.loc[:,'Time_s'])
    
    
    current_end_trace_spline=scipy.interpolate.UnivariateSpline(stimulus_end_table.loc[:,'Time_s'],np.array(stimulus_end_table.loc[:,'Input_current_pA']))
    current_end_trace_spline.set_smoothing_factor(.999)
    current_end_trace_filtered=current_end_trace_spline(time_end_array)
    current_end_trace_filtered_derivative = ordifunc.get_derivative( current_end_trace_filtered , time_end_array )
    current_end_trace_filtered_derivative=np.insert(current_end_trace_filtered_derivative,0,current_end_trace_filtered_derivative[0])
    

    
    stimulus_end_table['Input_current_derivative_pA/s']=current_end_trace_filtered_derivative
    
    
    #determine T_trans_end
    T_trans_table_end=stimulus_end_table[stimulus_end_table["Time_s"]<(stim_end_time+.002)]
    T_trans_table_end=T_trans_table_end[T_trans_table_end['Time_s']>(stim_end_time-.002)]
   
    T_Trans_end_current_derivative=np.array(T_trans_table_end['Input_current_derivative_pA/s'])
    
    if stim_amplitude <= stimulus_baseline: # at the end of the stimulus
        
        max_abs_dI_dt_index_end = np.nanargmax(T_Trans_end_current_derivative)
        
    elif stim_amplitude>stimulus_baseline:
        max_abs_dI_dt_index_end = np.nanargmin(T_Trans_end_current_derivative)
        
   
    T_trans_end=np.array(T_trans_table_end.loc[:,'Time_s'])[max_abs_dI_dt_index_end]
   
   
   
    ## determine Pre_Trans_Median and Post_Trans_median; 
    #Median current value in a window between T_trans-0.006 and T_trans-0.002 (and vice-versa)
    Pre_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']<=T_trans_end-.002]
    Pre_Trans_Median_table_end=Pre_Trans_Median_table_end[Pre_Trans_Median_table_end['Time_s']>=T_trans_end-.006]
    Pre_Trans_Median_end=np.median(Pre_Trans_Median_table_end['Input_current_pA'])
    
    Post_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']>=T_trans_end+.002]
    Post_Trans_Median_table_end=Post_Trans_Median_table_end[Post_Trans_Median_table_end['Time_s']<=T_trans_end+.006]
    Post_Trans_Median_end=np.median(Post_Trans_Median_table_end['Input_current_pA'])
    
    I_step_end=Post_Trans_Median_end-Pre_Trans_Median_end
   
    
   
   
    # T_trans_line_end=pd.Series([T_trans_end,np.array(T_trans_table_end.loc[:,'Membrane_potential_mV'])[max_abs_dI_dt_index_end],'T_trans_end'],index=["Time_s","Membrane_potential_mV","Feature"])
    # point_table_end=point_table_end.append(T_trans_line_end,ignore_index=True)
    
    
    ## determine Pre_Trans_Median and Post_Trans_median; 
    #Median current value in a window between T_trans_end-0.006 and T_trans_end-0.002 (and vice-versa)
    Pre_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']<=T_trans_end-.002]
    Pre_Trans_Median_table_end=Pre_Trans_Median_table_end[Pre_Trans_Median_table_end['Time_s']>=T_trans_end-.006]
    Pre_Trans_Median_end=np.median(Pre_Trans_Median_table_end['Input_current_pA'])
    
    Post_Trans_Median_table_end=stimulus_end_table[stimulus_end_table['Time_s']>=T_trans_end+.002]
    Post_Trans_Median_table_end=Post_Trans_Median_table_end[Post_Trans_Median_table_end['Time_s']<=T_trans_end+.006]
    Post_Trans_Median_end=np.median(Post_Trans_Median_table_end['Input_current_pA'])
    
    I_step_end=Post_Trans_Median_end-Pre_Trans_Median_end
    
    #determine DT_trans, the time scale of response transient to avoid contaminated period;
   
    ## get time of the max (min) dV/dt, for a positive (negative) Istep, respectively
    DT_Trans_table_end=stimulus_end_table[stimulus_end_table['Time_s']<=T_trans_end].reset_index(drop=True).copy()
   
   
    DT_Trans_table_end=DT_Trans_table_end[DT_Trans_table_end['Time_s']>=(T_trans_end-.003)]
    voltage_derivative_trace=np.array(DT_Trans_table_end.loc[:,'Potential_first_time_derivative_mV/s'])


    
    if stim_amplitude <= stimulus_baseline:
   
        max_dV_dt_index=np.nanargmax(voltage_derivative_trace)
        time_max_abs_dV_dt_end=np.array(DT_Trans_table_end['Time_s'])[max_dV_dt_index]
        
        DT_trans_line_end=pd.DataFrame([time_max_abs_dV_dt_end,np.array(DT_Trans_table_end.loc[:,'Membrane_potential_mV'])[max_dV_dt_index],'DT_trans_min_dV_dt']).T
        DT_trans_line_end.columns=["Time_s","Membrane_potential_mV","Feature"]
        
    elif stim_amplitude > stimulus_baseline:
   
        min_dV_dt_index=np.nanargmin(voltage_derivative_trace)
        time_max_abs_dV_dt_end=np.array(DT_Trans_table_end['Time_s'])[min_dV_dt_index]
        
        DT_trans_line_end=pd.DataFrame([time_max_abs_dV_dt_end,np.array(DT_Trans_table_end.loc[:,'Membrane_potential_mV'])[min_dV_dt_index],'DT_trans_max_dV_dt']).T
        DT_trans_line_end.columns=["Time_s","Membrane_potential_mV","Feature"]
        
    
    point_table_end=pd.concat([point_table_end,DT_trans_line_end],ignore_index=True)
    
    
    ##get the time of the last positive(negative) zero crossing dV/dt for a positive (negative) Istep, respectively
    if stim_amplitude <= stimulus_baseline:
   
        for index,elt in enumerate(voltage_derivative_trace[1:]):
            previous_elt=voltage_derivative_trace[index]
            if previous_elt <=0 and elt>0:
                last_zero_crossing_index=index
                break
            else:
                last_zero_crossing_index=len(voltage_derivative_trace)-1
            
    elif stim_amplitude > stimulus_baseline:
   
        for index,elt in enumerate(voltage_derivative_trace[1:]):
            previous_elt=voltage_derivative_trace[index]
            if previous_elt >=0 and elt<0:
                last_zero_crossing_index=index
                break
            else:
                last_zero_crossing_index=len(voltage_derivative_trace)-1
    
    time_zero_crossing_end=np.array(DT_Trans_table_end['Time_s'])[last_zero_crossing_index]
    
    
    DT_trans_line_end=pd.DataFrame([time_zero_crossing_end,np.array(DT_Trans_table_end.loc[:,'Membrane_potential_mV'])[last_zero_crossing_index],'DT_trans_zero_crossing_dV_dt']).T
    DT_trans_line_end.columns=["Time_s","Membrane_potential_mV","Feature"]
    point_table_end=pd.concat([point_table_end,DT_trans_line_end],ignore_index=True)
    
    DT_trans_end=max(.002,abs(T_trans_end-time_max_abs_dV_dt_end),abs(T_trans_end-time_zero_crossing_end))
    
    #Get fit table : [T_trans_end-T_seg_duration_end ; T_seg_start_end]
    
    T_seg_start_end = T_trans_end-max(.002,2.5*DT_trans_end)
    T_seg_duration_end = max(.005,2.5*DT_trans_end)
    
   
    # Get a table (fit_table_end) of Time-Potential values between T_seg_start_end and T_seg_start_end-T_seg_duration_end
    fit_table_end=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_seg_start_end-T_seg_duration_end)]
    
    fit_table_end=fit_table_end[fit_table_end['Time_s']<=(T_seg_start_end)]
    
    test_spike_table_end=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_seg_start_end-T_seg_duration_end)]
    test_spike_table_end=stimulus_end_table[stimulus_end_table["Time_s"]<=T_trans_end+.01]
    
    # voltage_spike_table=identify_spike_shorten(np.array(test_spike_table_end.Membrane_potential_mV),
    #                                 np.array(test_spike_table_end.Time_s),
    #                                 np.array(test_spike_table_end.Input_current_pA),
    #                                 np.array(test_spike_table_end.Time_s)[0],
    #                                 np.array(test_spike_table_end.Time_s)[-1],do_plot=False)
    
    voltage_spike_table = sp_an.identify_spike(np.array(test_spike_table_end.Membrane_potential_mV),
                                    np.array(test_spike_table_end.Time_s),
                                    np.array(test_spike_table_end.Input_current_pA),
                                    np.array(test_spike_table_end.Time_s)[0],
                                    np.array(test_spike_table_end.Time_s)[-1],do_plot=False)
    
    if len(voltage_spike_table['Peak']) != 0:
        is_spike = True
    else:
        is_spike = False
        
    if is_spike == True:
    #if np.nanmax(fit_table_end['Potential_first_time_derivative_mV/s']) >20. or np.nanmax(fit_table_end['Potential_first_time_derivative_mV/s']) <-20.:
        Bridge_Error_stim_end=np.nan


    else:
        
        #Use fit_table_end to fit either a 2nd order poly or Exponential to the membrane voltage trace
        a,b,c,RMSE_poly=fit_second_order_poly(fit_table_end,do_plot=False)
        #return fit_table_end
        
       #plt.plot(fit_table_end['Time_s'],fit_table_end['Membrane_potential_mV'])
        A,tau,RMSE_expo=fit_exponential_BE(fit_table_end,stim_start=False,do_plot=False)
        
        # Get a table between [T_trans_end and T_trans_end+T_seg_duration_end ] to estimate backward to T_trans_end from the fit
        Pre_trans_Init_Value_table_end=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_seg_start_end-T_seg_duration_end)]
        Pre_trans_Init_Value_table_end=Pre_trans_Init_Value_table_end[Pre_trans_Init_Value_table_end['Time_s']<=T_trans_end]
        
        if np.isnan(RMSE_poly) and np.isnan(RMSE_expo):
            Bridge_Error_stim_end = np.nan
           
        else:
           
            if np.isnan(RMSE_poly):
                RMSE_poly=np.inf
            elif np.isnan(RMSE_expo):
                
                RMSE_expo=np.inf
            
            #Use the best fit to extrapolate the membrane voltage fit backward to T_trans_end
            if RMSE_poly<=RMSE_expo:
                backward_prediction_end=a*((Pre_trans_Init_Value_table_end.loc[:,'Time_s'])**2)+b*Pre_trans_Init_Value_table_end.loc[:,'Time_s']+c
                Pre_trans_init_value_end=a*((T_trans_end)**2)+b*T_trans_end+c
                
                backward_table_end=pd.DataFrame(np.column_stack((Pre_trans_Init_Value_table_end.loc[:,'Time_s'],backward_prediction_end)),columns=["Time_s","Membrane_potential_mV"])
                backward_table_end['Legend']='Pre_trans_init_value'
                
            elif RMSE_poly>RMSE_expo:
                backward_prediction_end=A*(np.exp(-(Pre_trans_Init_Value_table_end.loc[:,'Time_s'])/tau))
                Pre_trans_init_value_end=A*(np.exp(-(T_trans_end)/tau))
                backward_table_end=pd.DataFrame(np.column_stack((Pre_trans_Init_Value_table_end.loc[:,'Time_s'],backward_prediction_end)),columns=["Time_s","Membrane_potential_mV"])
                backward_table_end['Legend']='Pre_trans_init_value' 
            
        
           
            pre_trans_init_value_line=pd.DataFrame([T_trans_end,Pre_trans_init_value_end,'Pre_trans_init_value']).T
            pre_trans_init_value_line.columns=["Time_s","Membrane_potential_mV","Feature"]
            point_table_end=pd.concat([point_table_end,pre_trans_init_value_line],ignore_index=True)
           
            #Corrected Version Post trans.Final
            #PostTrans_Est_table_to_fit is a table of Time_potential values between 1ms and 10ms before T_trans_end,
            PostTrans_Est_table_to_fit=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_trans_end+.001)]
            PostTrans_Est_table_to_fit=PostTrans_Est_table_to_fit[PostTrans_Est_table_to_fit['Time_s']<(T_trans_end+.01)]
            
            #fit the potential values to a 2nd order polynomial
            a_bis,b_bis,c_bis,RMSE_poly_bis=fit_second_order_poly(PostTrans_Est_table_to_fit,do_plot=False)
            
            # Use the PostTrans_Est_table_to_fit fit to estimate the potential value at T_trans_end = Post_Trans_Final
            Post_Trans_Final=a_bis*((T_trans_end)**2)+b_bis*T_trans_end+c_bis
            PostTrans_Est_table_to_fit['Legend']='PostTrans_fit'
            #Get the same table as PostTrans_Est_table_to_fit, but up to T_trans_end, for plotting purpose
            PostTrans_Est_table_to_T_trans_table=stimulus_end_table[stimulus_end_table["Time_s"]>=(T_trans_end)]
            PostTrans_Est_table_to_T_trans_table=PostTrans_Est_table_to_T_trans_table[PostTrans_Est_table_to_T_trans_table['Time_s']<(T_trans_end+.01)]
            PostTrans_Est_pred=a_bis*((PostTrans_Est_table_to_T_trans_table.loc[:,"Time_s"])**2)+b_bis*PostTrans_Est_table_to_T_trans_table.loc[:,"Time_s"]+c_bis
            
            PostTrans_Est_table=pd.DataFrame(np.column_stack((PostTrans_Est_table_to_T_trans_table.loc[:,"Time_s"],PostTrans_Est_pred)),columns=["Time_s","Membrane_potential_mV"])
            PostTrans_Est_table['Legend']='PostTrans_Est'
            
            
            Post_trans_final_value_line=pd.DataFrame([T_trans_end,Post_Trans_Final,'PostTrans_Est']).T
            Post_trans_final_value_line.columns=["Time_s","Membrane_potential_mV","Feature"]
            point_table_end=pd.concat([point_table_end,Post_trans_final_value_line],ignore_index=True)
            
            
            #fit_table_end=pd.concat([backward_table, PreTrans_Est_table], axis=0)
            
            #Bridge_Error= (Post_trans_init_value-Pre_Trans_Final_Value)/(stim_amplitude-stimulus_baseline)
            
       
         
     ########################################################################################
       # Pre_Trans_Final_Value=a_bis*((PreTrans_Est_table_to_fit.iloc[-1,0])**2)+b_bis*PreTrans_Est_table_to_fit.iloc[-1,0]+c_bis
        
        Bridge_Error_stim_end = (Post_Trans_Final-Pre_trans_init_value_end)/I_step_end
        at_least_one_Bridge_Error=True
    
    if at_least_one_Bridge_Error==True:
        # Bridge Error corresponds to the minimum absolute value of computed BE
        Bridge_Error_array = np.array([Bridge_Error_stim_start,Bridge_Error_stim_end])
        Bridge_Error = Bridge_Error_array[np.nanargmin(np.abs(Bridge_Error_array))]
        
        
    else: 
        Bridge_Error = np.nan

    ########################################################################################
   
    
    
    
    if do_plot:
        TVC_table=TVC_table.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float",
                                    "Input_current_pA":"float"})
        
        fit_table=fit_table.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float",
                                    "Input_current_pA":"float"})

        backward_table=backward_table.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        PreTrans_Est_table_to_fit=PreTrans_Est_table_to_fit.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        PreTrans_Est_table=PreTrans_Est_table.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        point_table=point_table.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        backward_table_end=backward_table_end.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        PostTrans_Est_table_to_fit=PostTrans_Est_table_to_fit.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        PostTrans_Est_table=PostTrans_Est_table.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        point_table_end=point_table_end.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        fit_table_end=fit_table_end.astype({"Time_s":"float",
                                    'Membrane_potential_mV':"float"})
        
        my_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Membrane_potential_mV'))+p9.geom_line(color='black')
        my_plot+= p9.geom_line(fit_table,p9.aes(x='Time_s',y='Membrane_potential_mV'),color='red',size=.7)
        my_plot_end=p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Membrane_potential_mV'))+p9.geom_line(color='black')
        my_plot_end+= p9.geom_line(fit_table_end,p9.aes(x='Time_s',y='Membrane_potential_mV'),color='red',size=.7)
        if np.isnan(Bridge_Error_stim_start) == False :
            
            my_plot+= p9.geom_line(backward_table,p9.aes(x='Time_s',y='Membrane_potential_mV',color="Legend"))
            my_plot+=p9.geom_line(PreTrans_Est_table_to_fit,p9.aes(x='Time_s',y='Membrane_potential_mV'),color='yellow',size=.7)
            my_plot+=p9.geom_line(PreTrans_Est_table,p9.aes(x='Time_s',y='Membrane_potential_mV',color='Legend'))
            my_plot+=p9.geom_point(point_table,p9.aes(x='Time_s',y='Membrane_potential_mV',color="Feature"))
            my_plot+=p9.xlim(min(np.array(PreTrans_Est_table['Time_s']))-.005,max(np.array(fit_table['Time_s']))+.005)
        else:
            my_plot+= p9.geom_line(test_spike_table,p9.aes(x='Time_s',y='Membrane_potential_mV'),color='red',linetype='dashed',size=.9)
            
            
        if np.isnan(Bridge_Error_stim_end) == False:
            
            
            my_plot_end+=p9.geom_line(backward_table_end,p9.aes(x='Time_s',y='Membrane_potential_mV',color='Legend'))
            my_plot_end+=p9.geom_line(PostTrans_Est_table_to_fit,p9.aes(x='Time_s',y='Membrane_potential_mV'),color='yellow',size=.7)
            my_plot_end+=p9.geom_line(PostTrans_Est_table,p9.aes(x='Time_s',y='Membrane_potential_mV',color='Legend'))
            my_plot_end+=p9.geom_point(point_table_end,p9.aes(x='Time_s',y='Membrane_potential_mV',color="Feature"))
            my_plot_end+=p9.xlim(min(np.array(fit_table_end['Time_s']))-.005,max(np.array(PostTrans_Est_table['Time_s']))+.005)
            
        else: 
            my_plot_end+= p9.geom_line(test_spike_table_end,p9.aes(x='Time_s',y='Membrane_potential_mV'),color='red',linetype='dashed',size=.9)
            #my_plot+=p9.xlim(stim_start_time-.03,stim_start_time+.03)
        my_plot+=p9.geom_vline(xintercept=T_trans,color='purple')
        #my_plot+=p9.ylim(-75,-40)
        my_plot_end+=p9.geom_vline(xintercept=T_trans_end,color='purple')
        my_plot_end+=p9.xlim(stim_end_time-.03,stim_end_time+.03)
        my_plot+=p9.ggtitle(str("Membrane_potential_trace_stim_start\n\nBridge_Error="+str(round(Bridge_Error_stim_start,3))+" GOhms"))
        my_plot_end+=p9.ggtitle(str("Membrane_potential_trace_stim_end\n\nBridge_Error="+str(round(Bridge_Error_stim_end,3))+" GOhms"))
        
            
            
       
        #print(my_plot)
        #print(my_plot_end)
        
        stimulus_start_table['Filtered_current_trace']=current_trace_filtered

        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Input_current_pA'))+p9.geom_line(color='black')
        current_plot+=p9.geom_line(stimulus_start_table,p9.aes(x='Time_s',y='Filtered_current_trace'),color='pink')
        current_plot+=p9.geom_line(stimulus_start_table,p9.aes(x='Time_s',y='Input_current_derivative_pA/s'),color='brown')
        current_plot+=p9.xlim(stim_start_time-.03,stim_start_time+.03)
        current_plot+=p9.ggtitle(str("Input_Current_trace_Stim_start\n\nBridge_Error="+str(round(Bridge_Error_stim_start,3))+" GOhms"))

        current_plot+=p9.geom_hline(yintercept=Pre_Trans_Median,color='blue')
        current_plot+=p9.geom_hline(yintercept=Post_Trans_Median,color='red')
        current_plot+=p9.geom_vline(xintercept=T_trans,color='purple')
        #print(current_plot)
        
        
        
        
        stimulus_start_table['Filtered_current_trace']=current_trace_filtered
        current_plot_end=p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Input_current_pA'))+p9.geom_line(color='black')
        current_plot_end+=p9.geom_line(stimulus_end_table,p9.aes(x='Time_s',y='Filtered_current_trace'),color='pink')
        current_plot_end+=p9.geom_line(stimulus_end_table,p9.aes(x='Time_s',y='Input_current_derivative_pA/s'),color='brown')
       
        current_plot_end+=p9.ggtitle(str("Input_Current_trace_Stim_end\n\nBridge_Error="+str(round(Bridge_Error_stim_end,3))+" GOhms"))
        
        stimulus_end_table['Filtered_current_trace']=current_end_trace_filtered
        current_plot_end+=p9.geom_hline(yintercept=Post_Trans_Median_end,color='blue')
        current_plot_end+=p9.geom_hline(yintercept=Pre_Trans_Median_end,color='red')
        current_plot_end+=p9.geom_line(stimulus_end_table,p9.aes(x='Time_s',y='Filtered_current_trace'),color='pink')
        current_plot_end+=p9.geom_line(stimulus_end_table,p9.aes(x='Time_s',y='Input_current_derivative_pA/s'),color='brown')
        current_plot_end+=p9.geom_vline(xintercept=T_trans_end,color='purple')
        current_plot_end+=p9.xlim(stim_end_time-.03,stim_end_time+.03)
        #print(current_plot_end)
        return my_plot,my_plot_end,current_plot,current_plot_end
        
    return Bridge_Error 


    
def fit_membrane_time_cst (original_TVC_table,start_time,end_time,do_plot=False):
    '''
    Fit decaying time constant model to membrane voltage trace

    Parameters
    ----------
    original_TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    start_time : Float
        Start time of the window to consider.
        
    end_time : Float
        End time of the window to consider.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_A, best_tau, best_C : Float
        Fitting results
        
    membrane_resting_voltage : Float
        
    NRMSE : Float
        Godness of fit.

    '''
    try:
        TVC_table=original_TVC_table.copy()
        sub_TVC_table=TVC_table[TVC_table['Time_s']<=(end_time+.5)]
        sub_TVC_table=sub_TVC_table[sub_TVC_table['Time_s']>=(start_time-.5)]
        
        x_data=np.array(sub_TVC_table.loc[:,'Time_s'])
        y_data=np.array(sub_TVC_table.loc[:,"Membrane_potential_mV"])
        
        start_idx = np.argmin(abs(x_data - start_time))
        end_idx = np.argmin(abs(x_data - end_time))
        
        
        #Estimate parameters initial values
        membrane_resting_voltage=np.median(y_data[:start_idx])

        mid_idx=int((end_idx+start_idx)/2)

        
        x_0=start_time
        y_0=y_data[np.argmin(abs(x_data - x_0))]
        
        x_1=start_time+.05
        y_1=y_data[np.argmin(abs(x_data - x_1))]
        
       
        initial_membrane_SS=np.median(y_data[mid_idx:end_idx])


        if (y_1-initial_membrane_SS)/(y_0-initial_membrane_SS)>0:
            initial_time_cst = (x_0-x_1)/np.log((y_1-initial_membrane_SS)/(y_0-initial_membrane_SS))
            initial_A=(y_1-initial_membrane_SS)*np.exp(x_1/initial_time_cst)


        else:

            membrane_delta=initial_membrane_SS-membrane_resting_voltage
        
            
            initial_voltage_time_cst=membrane_resting_voltage+(2/3)*membrane_delta

            initial_voltage_time_cst_idx=np.argmin(abs(y_data[start_idx:end_idx] - initial_voltage_time_cst))+start_idx
            initial_time_cst=x_data[initial_voltage_time_cst_idx]-start_time
            
        
            
            initial_A=(y_data[start_idx]-initial_membrane_SS)/np.exp(-x_data[start_idx]/initial_time_cst)
            
            
            
        double_step_model=Model(time_cst_model)
       
        double_step_model_pars=Parameters()
        
        double_step_model_pars.add('A',value=initial_A)
        double_step_model_pars.add('tau',value=initial_time_cst)
        double_step_model_pars.add('C',value=initial_membrane_SS)


        double_step_out=double_step_model.fit(y_data[start_idx:end_idx], double_step_model_pars, x=x_data[start_idx:end_idx])        

        best_A=double_step_out.best_values['A']
        best_tau=double_step_out.best_values['tau']
        best_C=double_step_out.best_values['C']
        
        
        simulation=time_cst_model(x_data[start_idx:end_idx],best_A,best_tau,best_C)
        sim_table=pd.DataFrame(np.column_stack((x_data[start_idx:end_idx],simulation)),columns=["Time_s","Membrane_potential_mV"])
        
        squared_error = np.square(y_data[start_idx:end_idx] - simulation)
        sum_squared_error = np.sum(squared_error)
        current_RMSE = np.sqrt(sum_squared_error / len(y_data[start_idx:end_idx]))
        
        NRMSE=current_RMSE/abs(membrane_resting_voltage-initial_membrane_SS)


        
        
        if do_plot==True:
            
            my_plot=p9.ggplot(TVC_table,p9.aes(x="Time_s",y="Membrane_potential_mV"))+p9.geom_line(color='blue')#+xlim((start_time-.1),(end_time+.1))
            
            
            my_plot=my_plot+p9.geom_line(sim_table,p9.aes(x='Time_s',y='Membrane_potential_mV'),color='red')
            
    
            #print(my_plot)
            return my_plot
            
        return best_A,best_tau,best_C,membrane_resting_voltage,NRMSE
    except (TypeError):
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        NRMSE=np.nan
        return best_A,best_tau,best_C,membrane_resting_voltage,NRMSE
    except(ValueError):
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        NRMSE=np.nan
        return best_A,best_tau,best_C,membrane_resting_voltage,NRMSE
    
def Double_Heaviside_function(x, stim_start, stim_end,baseline,stim_amplitude):
    """Heaviside step function."""
    
    if stim_end<=min(x):
        o=np.empty(x.size);o.fill(stim_amplitude)
        return o
    
    elif stim_start>=max(x):
        o=np.empty(x.size);o.fill(baseline)
        return o
    
    else:
        o=np.empty(x.size);o.fill(baseline)
        
        
        start_index = max(np.where( x < stim_start)[0])

        end_index=max(np.where( x < stim_end)[0])
        
        o[start_index:end_index] = stim_amplitude
    
        return o

def fit_second_order_poly(original_fit_table,do_plot=False):
    '''
    Fit 2nd order polynomial to time varying signal

    Parameters
    ----------
    original_fit_table : pd.DataFrame
        Contains the Time, voltage, Current traces arranged in columns.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    a, b, c : Float
        Fitting results.
    RMSE_poly : Float
        Godness of fit.

    '''
    try:
        fit_table=original_fit_table.copy()
        x_data=np.array(fit_table.loc[:,'Time_s'])
        y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
        
        poly_model=QuadraticModel()
        pars = poly_model.guess(y_data, x=x_data)
        out = poly_model.fit(y_data, pars, x=x_data)
        
        a=out.best_values["a"]
        b=out.best_values["b"]
        c=out.best_values["c"]
        
        pred=a*((x_data)**2)+b*x_data+c
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE_poly = np.sqrt(sum_squared_error / y_data.size)
        
        
        
        fit_table['Data']='Original_Data'
        simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
        simulation_table['Data']='Fit_Poly_Data'
       # fit_table=pd.concat([fit_table, simulation_table], axis=0)
        if do_plot:
            
            my_plot=p9.ggplot(fit_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+p9.geom_line()
            my_plot+=p9.geom_line(simulation_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))
            print(my_plot)
        return a,b,c,RMSE_poly
    
    except (ValueError):
        a = np.nan
        b = np.nan
        c = np.nan
        RMSE_poly = np.nan
        return  a,b,c,RMSE_poly
    
def fit_exponential_BE(original_fit_table,stim_start=True,do_plot=False):
    '''
    Fit an exponential curve to to membrane trace at either the stimulus start or the stimulus end, during the estimation of the Bridge Error

    Parameters
    ----------
    original_fit_table : pd.DataFrame
        Contains the Time, voltage, Current traces arranged in columns.
        
    stim_start : Boolean, optional
        Wether to fit at the stimulus start or end time. The default is True.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_A, best_tau : Float
        Fit result.
    RMSE_expo : Float
        Godness of fit.

    '''
    
    try:
        fit_table=original_fit_table.copy()
        x_data=np.array(fit_table.loc[:,'Time_s'])
        y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
        expo_model=ExponentialModel()
        
        
        membrane_start_voltage = y_data[0]
        membrane_end_voltage = y_data[-1]
        
        membrane_delta=membrane_end_voltage - membrane_start_voltage
    
        
        if stim_start == True:
            
            exp_offset = y_data[-1]
            initial_voltage_time_cst=membrane_end_voltage+(1-(1/np.exp(1)))*membrane_delta
            initial_voltage_time_cst_idx=np.argmin(abs(y_data - initial_voltage_time_cst))
            initial_time_cst=x_data[initial_voltage_time_cst_idx]-x_data[0]
            initial_A=(membrane_start_voltage-exp_offset)/np.exp(-x_data[0]/(initial_time_cst))
            

                
        elif stim_start == False:

            
            
            exp_offset = y_data[-1]
            initial_voltage_time_cst=membrane_end_voltage+(1-(1/np.exp(1)))*membrane_delta
            initial_voltage_time_cst_idx=np.argmin(abs(y_data - initial_voltage_time_cst))
            initial_time_cst=x_data[initial_voltage_time_cst_idx]-x_data[0]
            
            initial_time_cst= -initial_time_cst
            initial_A=(membrane_start_voltage-exp_offset)/np.exp(-x_data[0]/(initial_time_cst))
            
        
        
    
    
        expo_model = Model(time_cst_model)
        expo_model_pars=Parameters()
        expo_model_pars.add('A',value=initial_A)
        expo_model_pars.add('tau',value=initial_time_cst)
        expo_model_pars.add('C',value=exp_offset)
        
        expo_out=expo_model.fit(y_data, expo_model_pars, x=x_data)        
        
        
        best_A=expo_out.best_values['A']
        best_tau=expo_out.best_values['tau']
        best_C=expo_out.best_values['C']
        
        
        
        pred = time_cst_model(x_data,best_A,best_tau,best_C)
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE_expo = np.sqrt(sum_squared_error / y_data.size)
        fit_table['Data']='Original_Data'
        simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
        simulation_table['Data']='Fit_Expo_Data'
        fit_table=pd.concat([fit_table, simulation_table], axis=0)
        if do_plot:
            
            my_plot=p9.ggplot(fit_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+p9.geom_line()
            print(my_plot)
        
        return best_A,best_tau,RMSE_expo
    
    except (ValueError):
        best_A=np.nan
        best_tau=np.nan
        RMSE_expo=np.nan
        return best_A,best_tau,RMSE_expo
    
def time_cst_model(x,A,tau,C):

    y=A*np.exp(-(x)/tau)+C
    return y
    
    
def create_cell_sweep_QC_table_new_version(cell_sweep_info_table):
    '''
    Apply a series of Quality Criteria to each sweep, and indicates which sweep pass the QC

    Parameters
    ----------
    cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).

    Returns
    -------
    sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit

    '''


    sweep_QC_table = pd.DataFrame(columns=["Sweep", "Passed_QC"])
    QC_list=['Passed_QC']
    sweep_id_list = cell_sweep_info_table.loc[:, 'Sweep']
    

    
    
    for sweep in sweep_id_list:

        
        sweep_QC = True


        sweep_line = pd.DataFrame([str(sweep),sweep_QC]).T
        
        sweep_line.columns=["Sweep","Passed_QC"]
        sweep_QC_table=pd.concat([sweep_QC_table,sweep_line],ignore_index=True)


    for line in sweep_QC_table.index:
        sweep_QC_table.loc[line,'Passed_QC']=sweep_QC_table.loc[line,QC_list].product()
    
    sweep_QC_table.index = sweep_QC_table['Sweep']
    sweep_QC_table.index = sweep_QC_table.index.astype(str)
    sweep_QC_table.index.name = 'Index'

    return sweep_QC_table

def get_TVC_table(arg_list):
    '''
    Function to be used in Parallel to extract TVC tables, each iteration having a different args list

    Parameters
    ----------
    arg_list : List
        List of parameters required by the function.

    Returns
    -------
    sweep_TVC_line : pd.DataFrame
        2 columns DataFrame first columns 'Sweep' corresponds to Sweep_id, second 'TVC' contains table Contains the Time, Potential, Current
        
    stim_time_line : pd.DataFrame
        3 columns DataFrame first columns 'Sweep' corresponds to Sweep_id, second 'Stim_start_s' contains stimulus start time, Third 'Stim_end_s' contains stimulus end time, 

    '''
    
    
    module,full_path_to_python_script,db_function_name,db_original_file_directory,cell_id,current_sweep,db_cell_sweep_file,stimulus_time_provided, stimulus_duration  = arg_list
    
    spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)
    DB_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DB_module)
    DB_get_trace_function = getattr(DB_module,db_function_name)
    if stimulus_time_provided == True:
        time_trace, voltage_trace, current_trace,stimulus_start, stimulus_end = DB_get_trace_function(db_original_file_directory,
                                                                                                        cell_id,
                                                                                                        current_sweep,
                                                                                                        db_cell_sweep_file)
        
        sweep_TVC = ordifunc.create_TVC(time_trace=time_trace,
                                              voltage_trace=voltage_trace,
                                              current_trace=current_trace)
        
        
    else : 
        time_trace, voltage_trace, current_trace = DB_get_trace_function(db_original_file_directory,cell_id,current_sweep,db_cell_sweep_file)
        sweep_TVC = ordifunc.create_TVC(time_trace=time_trace,
                                              voltage_trace=voltage_trace,
                                              current_trace=current_trace)
        
        stimulus_start,stimulus_end = ordifunc.estimate_trace_stim_limits(sweep_TVC, 
                                                                            stimulus_duration,
                                                                            do_plot=False)
        
    sweep_TVC_line = pd.DataFrame([str(current_sweep), sweep_TVC]).T
    sweep_TVC_line.columns = ['Sweep', "TVC"]
    stim_time_line = pd.DataFrame([str(current_sweep), stimulus_start, stimulus_end]).T
    stim_time_line.columns = ['Sweep','Stim_start_s', 'Stim_end_s']
    return sweep_TVC_line, stim_time_line
    
    

