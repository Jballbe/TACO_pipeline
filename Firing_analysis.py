#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:03:38 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
from lmfit.models import PolynomialModel, Model, ConstantModel
from lmfit import Parameters
import plotnine as p9
import plotly.graph_objects as go
import math

def compute_cell_features(Full_SF_table,cell_sweep_info_table,response_duration_dictionnary,sweep_QC_table):
    '''
    Compute cell I/O features, for a dictionnnary of response_type:output_duration.
    Compute Adaptation for 500ms response.
     and adapation (cell_fit_table) as well as the different features computed (cell_feature_table)

    Parameters
    ----------
    
    Full_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    response_duration_dictionnary : dict
        Dictionnary containing as key the response type (Time_based, Index_based, Interval_based),
        and for each key a list of output_duration.
        
    sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit

    Returns
    -------
    cell_feature_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the Adaptation and I/O features.
    cell_fit_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the fit parameters to reconstruct I/O curve and adaptation curve.

    '''
    
    fit_columns = ['Response_type','Output_Duration', 'I_O_obs', 'I_O_QNRMSE', 'Hill_amplitude',
                   'Hill_coef', 'Hill_Half_cst','Hill_x0','Sigmoid_x0','Sigmoid_sigma']
    cell_fit_table = pd.DataFrame(columns=fit_columns)

    feature_columns = ['Response_type','Output_Duration', 'Gain','Threshold', 'Saturation_Frequency',"Saturation_Stimulus"]
    cell_feature_table = pd.DataFrame(columns=feature_columns)

    for response_type in response_duration_dictionnary.keys():
        output_duration_list=response_duration_dictionnary[response_type]

        
        for output_duration in output_duration_list:

    
            stim_freq_table = get_stim_freq_table(
                Full_SF_table.copy(), cell_sweep_info_table.copy(),sweep_QC_table.copy(), output_duration,response_type)
    
            pruning_obs, do_fit = data_pruning_I_O(stim_freq_table,cell_sweep_info_table)

            if do_fit == True:
                #I_O_obs, Hill_Amplitude, Hill_coef, Hill_Half_cst,Hill_x0, sigmoid_x0,sigmoid_sigma,I_O_QNRMSE, Gain, Threshold, Saturation_freq,Saturation_stim = fit_IO_relationship(stim_freq_table,response_type,output_duration, do_plot=False)
                I_O_obs, Hill_Amplitude, Hill_coef, Hill_Half_cst,Hill_x0, sigmoid_x0,sigmoid_sigma,I_O_QNRMSE, Gain, Threshold, Saturation_freq,Saturation_stim, parameters_table = extract_IO_features_Test(stim_freq_table, response_type, output_duration, False)
            else:

                I_O_obs = pruning_obs
                empty_array = np.empty(11)
                empty_array[:] = np.nan
                Hill_Amplitude, Hill_coef, Hill_Half_cst,Hill_x0, sigmoid_x0,sigmoid_sigma,I_O_QNRMSE, Gain, Threshold, Saturation_freq,Saturation_stim = empty_array
    
    
    
            new_fit_table_line = pd.DataFrame([response_type,
                                               output_duration,
                                            I_O_obs,
                                            I_O_QNRMSE,
                                            Hill_Amplitude,
                                            Hill_coef,
                                            Hill_Half_cst,
                                            Hill_x0,
                                            sigmoid_x0,
                                            sigmoid_sigma]).T
            new_fit_table_line.columns=fit_columns
            cell_fit_table=pd.concat([cell_fit_table,new_fit_table_line],ignore_index=True)
    
    
            new_feature_table_line = pd.DataFrame([response_type,
                                               output_duration,
                                                Gain,
                                                Threshold,
                                                Saturation_freq,
                                                Saturation_stim]).T
            new_feature_table_line.columns=feature_columns
            cell_feature_table=pd.concat([cell_feature_table,new_feature_table_line],ignore_index=True)

        

    cell_fit_table=cell_fit_table.astype({'Response_type': 'str',
                                          'Output_Duration':'float',
                                          'I_O_QNRMSE':'float',
                                          'Hill_amplitude':'float',
                                          'Hill_coef':'float',
                                          'Hill_Half_cst':'float',
                                          'Hill_x0':'float',
                                          'Sigmoid_x0':'float',
                                          'Sigmoid_sigma':'float'})
    
    cell_feature_table_convert_dict={'Response_type': str,
                                   'Output_Duration':float,
                    'Gain':float,
                    'Threshold':float,
                    'Saturation_Frequency':float,
                    'Saturation_Stimulus':float}
    cell_feature_table=cell_feature_table.astype(cell_feature_table_convert_dict)

    if cell_fit_table.shape[0] == 0:
        cell_fit_table.loc[0, :] = np.nan
        cell_fit_table = cell_fit_table.apply(pd.to_numeric)
    if cell_feature_table.shape[0] == 0:
        cell_feature_table.loc[0, :] = np.nan
        cell_feature_table = cell_feature_table.apply(pd.to_numeric)
    
    return cell_feature_table,cell_fit_table


def get_stim_freq_table(original_SF_table, original_cell_sweep_info_table,sweep_QC_table,response_duration, response_based="Time_based"):
    '''
    Given a table containing the spike features for the different sweep, compute the frequency for a given response

    Parameters
    ----------
    original_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit.
        
    response_duration : float
        Duration of teh response to consider.
        
    response_based : str, optional
        Type of response to consider (can be 'Index_based' or 'Interval_based'. The default is "Time_based".

    Returns
    -------
    stim_freq_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.

    '''
    

    SF_table=original_SF_table.copy()


    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_list=np.array(original_SF_table.loc[:,"Sweep"])

    stim_freq_table=pd.DataFrame(columns=["Sweep","Stim_amp_pA","Frequency_Hz"])
    # stim_freq_table=cell_sweep_info_table.copy()
    # stim_freq_table['Frequency_Hz']=0

    for current_sweep in sweep_list:
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'].copy())
        df=df[df['Feature']=='Upstroke']
        stim_amp = cell_sweep_info_table.loc[current_sweep,"Stim_amp_pA"]

        if response_based == 'Time_based':
            #stim_freq_table.loc[current_sweep,'Frequency_Hz']=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_duration)].shape[0])/response_duration
            frequency = (df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_duration)].shape[0])/response_duration
        elif response_based == 'Index_based':
            df=df.sort_values(by=["Time_s"])
            if df.shape[0] < response_duration: # if there are less spikes than response duration required, then set frequency to NaN
                #e.g.: If we want spike 3, we need 3 spikes at least

                continue
                #stim_freq_table.loc[current_sweep,'Frequency_Hz']=np.nan
                
            
            else:
                spike_time=np.array(df['Time_s'])[int(response_duration-1)]
                
                #stim_freq_table.loc[current_sweep,'Frequency_Hz'] = df.shape[0]/(spike_time-cell_sweep_info_table.loc[current_sweep,'Stim_start_s'])
                frequency = df.shape[0]/(spike_time-cell_sweep_info_table.loc[current_sweep,'Stim_start_s'])
        elif response_based == 'Interval_based':
            df=df.sort_values(by=["Time_s"])
            
            
            if df.shape[0]<=int(response_duration): # if there are less interval than response duration required, then set frequency to NaN
                #e.g.: If we want interval 3, we need 4 spikes at least
                continue
                #stim_freq_table.loc[current_sweep,'Frequency_Hz']=np.nan
            else:
                spike_time=np.array(df['Time_s'])
                response_duration = int(response_duration)
                #As python indexing starts at 0,, and interval number starts at 1 then interval_n = [spike_(n)- spike_(n-1)]
                # So that interval 1=spike_1-spike_0 (the interval between the second spike and the first spike)
                #stim_freq_table.loc[current_sweep,'Frequency_Hz']=1/(spike_time[response_duration]-spike_time[response_duration-1])
                frequency = 1/(spike_time[response_duration]-spike_time[response_duration-1])
                
        
        new_line=pd.DataFrame([current_sweep,stim_amp,frequency]).T
        new_line.columns = stim_freq_table.columns

        stim_freq_table = pd.concat([stim_freq_table,new_line],ignore_index=True)
        
        
            
            
  
    stim_freq_table=pd.merge(stim_freq_table,sweep_QC_table,on='Sweep')
    stim_freq_table=stim_freq_table.astype({"Stim_amp_pA":float,
                                            "Frequency_Hz":float})
    
    return stim_freq_table

def data_pruning_I_O(original_stim_freq_table,cell_sweep_info_table):
    '''
    For a given set of couple (Input_current - Stim_Freq), test wether there is enough information before proceeding to IO fit.
    Test the number of non-zero response, the number of different non-zero response
    Estimate the original frequency step, to avoid fitting continuous IO curve to Type II neurons

    Parameters
    ----------
    original_stim_freq_table : pd.DataFrame
        DataFrame containing one row per sweep, with the corresponding input current and firing frequency.
        
    cell_sweep_info_table : pd.DataFraem
        DESCRIPTION.

    Returns
    -------
    obs : str
        If do_fit == False, then obs contains the reason why no to proceed to IO fit .
    do_fit : bool
        Wether or not proceeding to IO fit.

    '''
    stim_freq_table=original_stim_freq_table.copy()
    stim_freq_table=stim_freq_table[stim_freq_table["Passed_QC"]==True]
    stim_freq_table = stim_freq_table.sort_values(
        by=['Stim_amp_pA', 'Frequency_Hz'])
    
    frequency_array = np.array(stim_freq_table.loc[:, 'Frequency_Hz'])
    obs = '-'
    do_fit = True
    
    
    
    
    non_zero_freq = frequency_array[np.where(frequency_array > 0)]
   
    if np.count_nonzero(frequency_array) < 4:
        obs = 'Less_than_4_response'
        do_fit = False
        return obs, do_fit  # , max_freq,max_freq_step,(max_freq_step/max_freq)

    
    
    if len(np.unique(non_zero_freq)) < 3:
        obs = 'Less_than_3_different_frequencies'
        do_fit = False
        return obs, do_fit  # , max_freq,max_freq_step,(max_freq_step/max_freq)

   
    minimum_frequency_step = get_min_freq_step(stim_freq_table,do_plot=False)[0]

    if minimum_frequency_step >30:
        obs = 'Minimum_frequency_step_higher_than_30Hz'
        do_fit = False
        return obs, do_fit
        
    
    obs = '--'
    do_fit = True
    return obs, do_fit



def fit_IO_relationship(original_stimulus_frequency_table, response_type,response_duration,do_plot=False,print_plot=False):
    '''
    Fit to the Input Output (stimulus-frequency) relationship  a continuous curve, and compute I/O features (Gain, Threshold, Saturation...)
    If a saturation is detected, then the IO relationship is fit to a Hill-Sigmoid function
    Otherwise the IO relationship is fit to a Hill function

    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.
        
    do_plot : Boolean, optional
        Do plot. The default is False.
    print_plot : Boolean, optional
        Print plot. The default is False.

    Returns
    -------
    feature_obs, Amp, Hill_coef, Hill_Half_cst, Hill_x0, sigmoid_x0, sigmoid_sigma : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_QNRMSE : float
        Godness of fit
        
    Gain, Threshold, Saturation_frequency, Saturation_stimulation : float
        Results of neuronal firing features computation.

    '''
    
    
    scale_dict = {'3rd_order_poly' : 'black',
                  '2nd_order_poly' : 'black',
                  'Trimmed_asc_seg' : 'pink',
                  "Trimmed_desc_seg" : "red",
                  'Ascending_Sigmoid' : 'blue',
                  "Descending_Sigmoid"  : 'orange',
                  'Amplitude' : 'yellow',
                  "Asc_Hill_Desc_Sigmoid": "red",
                  "First_Hill_Fit" : 'green',
                  'Hill_Sigmoid_Fit' : 'green',
                  'Sigmoid_fit' : "blue",
                  'Hill_Fit' : 'green',
                  True : 'o',
                  False : 's'
                  
                  }
    
    
    
    Gain=np.nan
    Threshold=np.nan
    Saturation_frequency=np.nan
    Saturation_stimulation=np.nan
    
    original_data_table =  original_stimulus_frequency_table.copy()
    original_data_table = original_data_table.dropna()
    
    
    if do_plot==True:
        plot_list=dict()
        original_data_table=original_data_table.astype({'Passed_QC':bool})

    else:
        plot_list=None
    
    original_data_subset_QC = original_data_table.copy()
    if 'Passed_QC' in original_data_subset_QC.columns:
        original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
    
    
   

    original_data_subset_QC = original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    
    trimmed_stimulus_frequency_table=original_data_subset_QC.copy()
    trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.reset_index(drop=True)
    trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    
    
    ### 0 - Trim Data
    
    response_threshold = (np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
    
    for elt in range(trimmed_stimulus_frequency_table.shape[0]):
        if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
            trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
        else:
            break
    
    trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
    for elt in range(trimmed_stimulus_frequency_table.shape[0]):
        if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
            trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
        else:
            break
    
    ### 0 - end
    
    ### 1 - Try to fit a 3rd order polynomial
    trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    trimmed_x_data=trimmed_stimulus_frequency_table.loc[:,'Stim_amp_pA']
    trimmed_y_data=trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]
    extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))

    if len(trimmed_y_data)>=4:
        third_order_poly_model = PolynomialModel(degree = 3)

        pars = third_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
        third_order_poly_model_results = third_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)

        best_c0 = third_order_poly_model_results.best_values['c0']
        best_c1 = third_order_poly_model_results.best_values['c1']
        best_c2 = third_order_poly_model_results.best_values['c2']
        best_c3 = third_order_poly_model_results.best_values['c3']
        
        extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
        
        trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                               "Frequency_Hz" : extended_trimmed_3rd_poly_model})
        trimmed_3rd_poly_table['Legend']='3rd_order_poly'
        
        if do_plot:
            
            polynomial_plot = p9.ggplot()
            polynomial_plot += p9.geom_point(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
            polynomial_plot += p9.geom_point(trimmed_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
            polynomial_plot += p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += p9.ggtitle('3rd order polynomial fit to trimmed_data')
            polynomial_plot += p9.scale_color_manual(values=scale_dict)
            polynomial_plot += p9.scale_shape_manual(values=scale_dict)
            if print_plot==True:
                print(polynomial_plot)
            
        extended_trimmed_3rd_poly_model_freq_diff=np.diff(extended_trimmed_3rd_poly_model)
        zero_crossings_3rd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_3rd_poly_model_freq_diff)))[0] # detect last index before change of sign
        if len(zero_crossings_3rd_poly_model_freq_diff)==0: # if the derivative of the 3rd order polynomial does not change sign --> Assume there is no Descending Segment
            Descending_segment=False
            ascending_segment = trimmed_3rd_poly_table.copy()
            
        elif len(zero_crossings_3rd_poly_model_freq_diff)==1:# if the derivative of the 3rd order polynomial changes sign 1 time
            first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
            if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                ## if before the change of sign the derivative of the 3rd order poly is positive, 
                ## then we know there is a Descending Segment after the change of sign
                Descending_segment=True
                
                ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                
            elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                ## if before the change of sign the derivative of the 3rd order poly is negative, 
                ## then we know the "Descending Segment" is fitted to the beginning of the data = artifact --> there is no Descending segment after the Acsending Segment
                Descending_segment=False
                ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] >= first_stim_root ]
                
        elif len(zero_crossings_3rd_poly_model_freq_diff)==2:# if the derivative of the 3rd order polynomial changes sign 2 times
            first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
            second_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[1]]
            
            
            if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                ## if before the first change of sign the derivative of the 3rd order poly is positive
                ## then we consider the Ascending Segment before the first root and the Descending Segment after the First root
                ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
                descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
                
                
                trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
                
                descending_linear_slope_init,descending_linear_intercept_init=linear_fit(trimmed_descending_segment["Stim_amp_pA"],
                                                      trimmed_descending_segment['Frequency_Hz'])
                
                if descending_linear_slope_init<=0:
                    Descending_segment = True
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                else:
                    Descending_segment=False
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                
                
            elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                Descending_segment=True
                ascending_segment = trimmed_3rd_poly_table[(trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root) & (trimmed_3rd_poly_table['Stim_amp_pA'] <= second_stim_root) ]
                descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > second_stim_root]
        
        ### 1 - end
    
    else:
        Descending_segment=False
    

    
    if Descending_segment:

        obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,sigmoid_x0, sigmoid_sigma,Legend,plot_list = fit_Hill_Sigmoid(original_stimulus_frequency_table, third_order_poly_model_results, ascending_segment, descending_segment, scale_dict,do_plot=do_plot,plot_list=plot_list,print_plot=print_plot)
     
        if obs == 'Hill-Sigmoid':

            feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table, response_type,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,response_duration,do_plot,plot_list=plot_list,print_plot=print_plot)

            if feature_obs == 'Hill-Sigmoid':
                if do_plot==True:

                    return plot_list
                return feature_obs,Amp,Hill_coef,Hill_Half_cst,Hill_x0,sigmoid_x0,sigmoid_sigma,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation
            
            else:
                Descending_segment=False
                obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,Legend,plot_list = fit_Single_Hill(original_stimulus_frequency_table, scale_dict,do_plot=do_plot,plot_list=plot_list,print_plot=print_plot)
                sigmoid_x0=np.nan
                sigmoid_sigma = np.nan

                if obs =='Hill':
                    feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table,response_type,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,response_duration,do_plot,plot_list=plot_list,print_plot=print_plot)
                    
                else:
                    empty_array = np.empty(5)
                    empty_array[:] = np.nan
                    feature_obs=obs
                    best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
                
        
        else:
            Descending_segment=False
            obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,Legend,plot_list = fit_Single_Hill(original_stimulus_frequency_table, scale_dict,do_plot=do_plot,plot_list=plot_list,print_plot=print_plot)
            sigmoid_x0=np.nan
            sigmoid_sigma = np.nan
            
            if obs =='Hill':
                feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table,response_type,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,response_duration,do_plot,plot_list=plot_list,print_plot=print_plot)
                
            else:
                empty_array = np.empty(5)
                empty_array[:] = np.nan
                feature_obs=obs
                best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
            
    
            
    else:

        obs,fit_model_table,Amp,Hill_x0, Hill_Half_cst,Hill_coef,Legend,plot_list = fit_Single_Hill(original_stimulus_frequency_table, scale_dict,do_plot=do_plot,plot_list=plot_list,print_plot=print_plot)
        sigmoid_x0=np.nan
        sigmoid_sigma = np.nan
        if obs =='Hill':
            feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table,response_type,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,response_duration,do_plot,plot_list=plot_list,print_plot=print_plot)
        else:
            empty_array = np.empty(5)
            empty_array[:] = np.nan
           
            feature_obs=obs
            best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
    
    if do_plot==True:
        return plot_list
    return feature_obs,Amp,Hill_coef,Hill_Half_cst,Hill_x0,sigmoid_x0,sigmoid_sigma,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation

    
def fit_IO_relationship_Test(original_stimulus_frequency_table,Force_single_Hill = False,do_plot=False,print_plot=False):
    '''
    Fit to the Input Output (stimulus-frequency) relationship  a continuous curve, and compute I/O features (Gain, Threshold, Saturation...)
    If a saturation is detected, then the IO relationship is fit to a Hill-Sigmoid function
    Otherwise the IO relationship is fit to a Hill function

    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.
        
    do_plot : Boolean, optional
        Do plot. The default is False.
    print_plot : Boolean, optional
        Print plot. The default is False.

    Returns
    -------
    feature_obs, Amp, Hill_coef, Hill_Half_cst, Hill_x0, sigmoid_x0, sigmoid_sigma : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_QNRMSE : float
        Godness of fit
        
    Gain, Threshold, Saturation_frequency, Saturation_stimulation : float
        Results of neuronal firing features computation.

    '''
    
    
    scale_dict = {'3rd_order_poly' : 'black',
                  '2nd_order_poly' : 'black',
                  'Trimmed_asc_seg' : 'pink',
                  "Trimmed_desc_seg" : "red",
                  'Ascending_Sigmoid' : 'blue',
                  "Descending_Sigmoid"  : 'orange',
                  'Amplitude' : 'yellow',
                  "Asc_Hill_Desc_Sigmoid": "red",
                  "First_Hill_Fit" : 'green',
                  'Hill_Sigmoid_Fit' : 'green',
                  'Sigmoid_fit' : "blue",
                  'Hill_Fit' : 'green',
                  True : 'o',
                  False : 's'
                  
                  }
    
    
    
    Gain=np.nan
    Threshold=np.nan
    Saturation_frequency=np.nan
    Saturation_stimulation=np.nan
    try:
        original_data_table =  original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        
        
        if do_plot==True:
            plot_list=dict()
            i=1
            original_data_table=original_data_table.astype({'Passed_QC':bool})
    
        else:
            plot_list=None
        
        original_data_subset_QC = original_data_table.copy()
        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        
       
    
        original_data_subset_QC = original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        trimmed_stimulus_frequency_table=original_data_subset_QC.copy()
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.reset_index(drop=True)
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        
        ### 0 - Trim Data
        
        response_threshold = (np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
        
        for elt in range(trimmed_stimulus_frequency_table.shape[0]):
            if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
                trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
            else:
                break
        
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
        for elt in range(trimmed_stimulus_frequency_table.shape[0]):
            if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
                trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
            else:
                break
        
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_x_data=trimmed_stimulus_frequency_table.loc[:,'Stim_amp_pA']
        trimmed_y_data=trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]
        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
    
        if len(trimmed_y_data)>=4:
            third_order_poly_model = PolynomialModel(degree = 3)
    
            pars = third_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
            third_order_poly_model_results = third_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
    
            best_c0 = third_order_poly_model_results.best_values['c0']
            best_c1 = third_order_poly_model_results.best_values['c1']
            best_c2 = third_order_poly_model_results.best_values['c2']
            best_c3 = third_order_poly_model_results.best_values['c3']
            
            extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
            
            trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                                   "Frequency_Hz" : extended_trimmed_3rd_poly_model})
            trimmed_3rd_poly_table['Legend']='3rd_order_poly'
            
            third_order_fit_params_table = get_parameters_table(pars, third_order_poly_model_results)
            
            third_order_fit_params_table.loc[:,'Fit'] = "3rd_order_polynomial_fit"
            
            if do_plot:
                
                polynomial_plot = p9.ggplot()
                polynomial_plot += p9.geom_point(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
                polynomial_plot += p9.geom_point(trimmed_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
                polynomial_plot += p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
                polynomial_plot += p9.ggtitle('3rd order polynomial fit to trimmed_data')
                polynomial_plot += p9.scale_color_manual(values=scale_dict)
                polynomial_plot += p9.scale_shape_manual(values=scale_dict)
                polynomial_fit_dict_scale_dict = {'3rd_order_poly' : 'black',
                              '2nd_order_poly' : 'black',
                              True : 'o',
                              False : 's'
                              
                              }
                polynomial_fit_dict = {"original_data_table":original_data_table,
                                       'trimmed_stimulus_frequency_table' : trimmed_stimulus_frequency_table,
                                       'trimmed_3rd_poly_table': trimmed_3rd_poly_table,
                                       'color_shape_dict' : polynomial_fit_dict_scale_dict}
                
                plot_list[f'{i}-Polynomial_fit']=polynomial_fit_dict
                i+=1
                if print_plot==True:
                    polynomial_plot.show()
                
            extended_trimmed_3rd_poly_model_freq_diff=np.diff(extended_trimmed_3rd_poly_model)
            zero_crossings_3rd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_3rd_poly_model_freq_diff)))[0] # detect last index before change of sign
            if len(zero_crossings_3rd_poly_model_freq_diff)==0: # if the derivative of the 3rd order polynomial does not change sign --> Assume there is no Descending Segment
                Descending_segment=False
                ascending_segment = trimmed_3rd_poly_table.copy()
                
            elif len(zero_crossings_3rd_poly_model_freq_diff)==1:# if the derivative of the 3rd order polynomial changes sign 1 time
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    ## if before the change of sign the derivative of the 3rd order poly is positive, 
                    ## then we know there is a Descending Segment after the change of sign
                    Descending_segment=True
                    
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    ## if before the change of sign the derivative of the 3rd order poly is negative, 
                    ## then we know the "Descending Segment" is fitted to the beginning of the data = artifact --> there is no Descending segment after the Acsending Segment
                    Descending_segment=False
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] >= first_stim_root ]
                    
            elif len(zero_crossings_3rd_poly_model_freq_diff)==2:# if the derivative of the 3rd order polynomial changes sign 2 times
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                second_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[1]]
                
                
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    ## if before the first change of sign the derivative of the 3rd order poly is positive
                    ## then we consider the Ascending Segment before the first root and the Descending Segment after the First root
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
                    descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
                    
                    
                    trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
                    
                    descending_linear_slope_init,descending_linear_intercept_init=linear_fit(trimmed_descending_segment["Stim_amp_pA"],
                                                          trimmed_descending_segment['Frequency_Hz'])
                    
                    if descending_linear_slope_init<=0:
                        Descending_segment = True
                        ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                        descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    else:
                        Descending_segment=False
                        ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=True
                    ascending_segment = trimmed_3rd_poly_table[(trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root) & (trimmed_3rd_poly_table['Stim_amp_pA'] <= second_stim_root) ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > second_stim_root]
            
            ### 1 - end
        
        else:
            Descending_segment=False
        
        if Force_single_Hill == True:
            Descending_segment=False
        original_data_table =  original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        if do_plot == True:
            original_data_table=original_data_table.astype({'Passed_QC':bool})
            
        
        original_data_subset_QC = original_data_table.copy()
        
    
        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        original_data_subset_QC=original_data_subset_QC.reset_index(drop=True)
        original_data_subset_QC=original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        original_x_data = np.array(original_data_subset_QC['Stim_amp_pA'])
        original_y_data = np.array(original_data_subset_QC['Frequency_Hz'])
        extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
        
        
        ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
        ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
        trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
        ascending_linear_slope_init,ascending_linear_intercept_init=linear_fit(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        
        trimmed_ascending_segment.loc[:,'Legend'] = "Trimmed ascending segment"
        #polynomial_plot += p9.geom_line(trimmed_ascending_segment, p9.aes(x="Stim_amp_pA", y="Frequency_Hz"), color='red')
        Trimmed_polynomial_plot = p9.ggplot()
        Trimmed_polynomial_plot += p9.geom_point(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
        Trimmed_polynomial_plot += p9.geom_point(trimmed_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
        Trimmed_polynomial_plot += p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"), color='black')
        Trimmed_polynomial_plot += p9.geom_abline(slope=ascending_linear_slope_init,
                                     intercept=ascending_linear_intercept_init,
                                     colour="red",
                                     linetype='dashed')
        
        Trimmed_polynomial_plot_color_dict = {"Trimmed ascending segment" : "red"}
        if Descending_segment:
            descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
            descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
            trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
            descending_linear_slope_init,descending_linear_intercept_init=linear_fit(trimmed_descending_segment["Stim_amp_pA"],
                                                  trimmed_descending_segment['Frequency_Hz'])
            
            trimmed_descending_segment.loc[:,'Legend'] = "Trimmed descending segment"
            trimmed_data_table = pd.concat([trimmed_ascending_segment, trimmed_descending_segment], ignore_index = True)
            
            
            
            max_freq_poly_fit = np.nanmax([np.nanmax(ascending_segment['Frequency_Hz']),np.nanmax(descending_segment['Frequency_Hz'])])
            #polynomial_plot += p9.geom_line(trimmed_descending_segment, p9.aes(x="Stim_amp_pA", y="Frequency_Hz"), color='pink')
            Trimmed_polynomial_plot += p9.geom_abline(slope=descending_linear_slope_init,
                                         intercept=descending_linear_intercept_init,
                                         colour="pink",
                                         linetype='dashed')
            
            Trimmed_polynomial_plot_color_dict["Trimmed descending segment"] = 'pink'
            
            
            
        else:
            
            trimmed_data_table = trimmed_ascending_segment
            max_freq_poly_fit = np.nanmax(extended_trimmed_3rd_poly_model)
        
        Trimmed_polynomial_plot += p9.geom_line(trimmed_data_table, p9.aes(x="Stim_amp_pA", y="Frequency_Hz", color= 'Legend'))
        Trimmed_polynomial_plot+=p9.scale_color_manual(values=Trimmed_polynomial_plot_color_dict)
        Trimmed_polynomial_plot+=p9.ggtitle("Trimmed segments")
        if do_plot: 
            Trimmed_polynomial_dict_scale_dict = {'3rd_order_poly' : 'black',
                          '2nd_order_poly' : 'black',
                          'Trimmed ascending segment' : "red",
                          'Trimmed descending segment' : "pink",
                          True : 'o',
                          False : 's'
                          
                          }
            trimmed_polynomial_fit_dict = {"original_data_table":original_data_table,
                                   'trimmed_stimulus_frequency_table' : trimmed_stimulus_frequency_table,
                                   'trimmed_3rd_poly_table': trimmed_3rd_poly_table,
                                   'ascending_linear_slope_init' : ascending_linear_slope_init,
                                   'ascending_linear_intercept_init' : ascending_linear_intercept_init,
                                   'color_shape_dict' : Trimmed_polynomial_dict_scale_dict,
                                   'trimmed_data_table' : trimmed_data_table}
            if Descending_segment:
                trimmed_polynomial_fit_dict['descending_linear_slope_init']=descending_linear_slope_init
                trimmed_polynomial_fit_dict['descending_linear_intercept_init']=descending_linear_intercept_init
            
            plot_list[f'{i}-Trimmed_polynomial_fit']=trimmed_polynomial_fit_dict
            i+=1
            
            if print_plot:
            
            
                Trimmed_polynomial_plot.show() 

        ### 2 - Fit Double Sigmoid : Amp*Sigmoid_asc*Sigmoid_desc
        
        # Ascending Sigmoid
        ascending_sigmoid_slope = max_freq_poly_fit/(4*ascending_linear_slope_init)
        ascending_sigmoid_fit= Model(sigmoid_function,prefix='asc_')
        ascending_segment_fit_params = ascending_sigmoid_fit.make_params()
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        # Amplitude
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*np.nanmax(extended_trimmed_3rd_poly_model))
        
        # Amp*Ascending segment
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        if Descending_segment:
            # Descending segment
            descending_sigmoid_slope = max_freq_poly_fit/(4*descending_linear_slope_init)
            descending_sigmoid_fit = Model(sigmoid_function,prefix='desc_')
            descending_sigmoid_fit_pars = descending_sigmoid_fit.make_params()
            descending_sigmoid_fit_pars.add("desc_x0",value=np.nanmean(trimmed_descending_segment['Stim_amp_pA']))
            descending_sigmoid_fit_pars.add("desc_sigma",value=descending_sigmoid_slope,min=3*descending_sigmoid_slope,max=-1e-9)
                
            # Amp*Ascending_sigmoid*Descending_sigmoid
            ascending_sigmoid_fit *=descending_sigmoid_fit
            ascending_segment_fit_params+=descending_sigmoid_fit_pars
            
        # Fit model to original data
        
        

        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        # Get fit best parameters
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        sigmoid_fit_parameters = get_parameters_table(ascending_segment_fit_params, ascending_sigmoid_fit_results)
        
        sigmoid_fit_parameters.loc[:,'Fit'] = "Double_sigmoid_fit"

        if Descending_segment:
            best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
            best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']
            full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data, best_value_x0_asc, best_value_sigma_asc) * sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
            full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                        "Frequency_Hz" : full_sigmoid_fit})
            full_sigmoid_fit_table['Legend'] = 'Fit result'

            initial_sigmoid_fit = ascending_segment_fit_params['Amp_c'] * sigmoid_function(extended_x_data, 
                                                                       ascending_segment_fit_params['asc_x0'], 
                                                                       ascending_segment_fit_params['asc_sigma']) * sigmoid_function(extended_x_data , ascending_sigmoid_fit_results.best_values["desc_x0"], ascending_sigmoid_fit_results.best_values['desc_sigma'])
            
            
            
            
        else:
            # Create simulation table
            full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data, best_value_x0_asc, best_value_sigma_asc)
            full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                        "Frequency_Hz" : full_sigmoid_fit})
            
            full_sigmoid_fit_table['Legend'] = 'Fit result'
            
            initial_sigmoid_fit = ascending_segment_fit_params['Amp_c'] * sigmoid_function(extended_x_data, 
                                                                       ascending_segment_fit_params['asc_x0'], 
                                                                       ascending_segment_fit_params['asc_sigma'])
        initial_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : initial_sigmoid_fit,
                                    "Legend" : "Initial conditions"})
        
        Double_sigmoid_fit_color_dict = {"Initial conditions":"red",
                                         "Fit result" : "green"}

        Double_sigmoid_fit_plot = p9.ggplot()
        Double_sigmoid_fit_plot += p9.geom_point(original_data_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
        Double_sigmoid_fit_plot += p9.geom_line(full_sigmoid_fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz", color = "Legend"))
        Double_sigmoid_fit_plot += p9.geom_line(initial_sigmoid_fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz", color = "Legend"))
        Double_sigmoid_fit_plot += p9.scale_color_manual(Double_sigmoid_fit_color_dict)
        Double_sigmoid_fit_plot += p9.ggtitle("Double_sigmoid_fit")
        if do_plot:
            
            
            
            Double_sigmoid_fit_dict = {"original_data_table":original_data_table,
                                       'full_sigmoid_fit_table':full_sigmoid_fit_table, 
                                       'initial_sigmoid_fit_table' : initial_sigmoid_fit_table,
                                   
                                   'color_shape_dict' : Double_sigmoid_fit_color_dict,
                                   }
            
            
            plot_list[f'{i}-Sigmoid_fit']=Double_sigmoid_fit_dict
            i+=1
            if print_plot:
                Double_sigmoid_fit_plot.show()
        
        
        
        # Fit a Hill to the ascending Sigmoid
        #Determine Hill half coef for first hill fit --> First stim_amp to elicit half max(frequency)
        half_response_index=np.argmax(original_y_data>=(0.5*max(original_y_data)))
       
        half_response_stim = original_x_data[half_response_index]
        
        # Set limit to Hill coef to prevent the parameter space exporation to go too high which breaks the fit
        max_Hill_coef=math.log(1e308)/math.log(max(original_x_data)+100)
    
       
    
        # Amplitude for hill fit
        amplitude_fit_Hill = ConstantModel(prefix='Amp_')
        amplitude_fit_Hill_pars = amplitude_fit_Hill.make_params()
        max_freq_index = original_data_table['Frequency_Hz'].idxmax()
        max_freq_stim = original_data_table.loc[max_freq_index, 'Stim_amp_pA']
        Sigmoid_derivative = best_value_Amp * sigmoid_function(max_freq_stim, best_value_x0_asc, best_value_sigma_asc) * (1-sigmoid_function(max_freq_stim, best_value_x0_asc, best_value_sigma_asc))
        if Sigmoid_derivative>0:
        
            amplitude_fit_Hill_pars['Amp_c'].set(value=best_value_Amp,min=0,max=3*best_value_Amp)
        else:
            amplitude_fit_Hill_pars['Amp_c'].set(value=best_value_Amp,min=0,max=1.2*best_value_Amp)
        
        
        # Create Hill model to fit on ascending sigmoid
        first_Hill_model = Model(hill_function)
        first_Hill_pars = first_Hill_model.make_params()
        first_Hill_pars.add("x0",value=np.nanmin(original_x_data),min=np.nanmin(original_x_data)-100,max=2*np.nanmin(original_x_data))
        first_Hill_pars.add("Half_cst",value=half_response_stim,min=1e-9, max = 5*half_response_stim)
        first_Hill_pars.add('Hill_coef',value=1.2,max=np.nanmin([2*1.2, max_Hill_coef]), min=1)
       
        #Create Amp*Hill_fit
        first_Hill_model *= amplitude_fit_Hill
        first_Hill_pars+=amplitude_fit_Hill_pars
        
        asc_sigmoid_to_fit = best_value_Amp * sigmoid_function(original_x_data , best_value_x0_asc, best_value_sigma_asc)
        
        
    
        first_Hill_result = first_Hill_model.fit(asc_sigmoid_to_fit, first_Hill_pars, x=original_x_data)
        
        best_H_amp = first_Hill_result.best_values['Amp_c']
        best_H_Half_cst=first_Hill_result.best_values['Half_cst']
        best_H_Hill_coef=first_Hill_result.best_values['Hill_coef']
        best_H_x0 = first_Hill_result.best_values['x0']
        
        
        
        Hill_fit_parameters = get_parameters_table(first_Hill_pars, first_Hill_result)
      
        Hill_fit_parameters.loc[:,'Fit'] = "Hill fit to sigmoid"

        Hill_fit_color_dict = {"Sigmoid to fit":"black",
                               "Initial conditions":"red",
                               "Fit results" : "green"}
        
        first_Hill_extended_fit = best_H_amp*hill_function(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        first_Hill_extended_fit_table=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : first_Hill_extended_fit})
        first_Hill_extended_fit_table.loc[:,'Legend'] = "Fit results"
        
        Asc_sig_to_fit = pd.DataFrame({'Stim_amp_pA':original_x_data,
                                                    'Frequency_Hz' : asc_sigmoid_to_fit})
        Asc_sig_to_fit.loc[:,'Legend'] = "Sigmoid to fit"
        initial_Hill_fit = best_value_Amp*hill_function(extended_x_data, first_Hill_pars['x0'], first_Hill_pars['Hill_coef'], first_Hill_pars['Half_cst'])
        initial_Hill_extended_fit_table=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : initial_Hill_fit})
        
        initial_Hill_extended_fit_table.loc[:,'Legend'] = "Initial conditions"
        Hill_table_plot = pd.concat([Asc_sig_to_fit,initial_Hill_extended_fit_table], ignore_index=True)
        Hill_table_plot = pd.concat([Hill_table_plot,first_Hill_extended_fit_table], ignore_index = True)

        
        Hill_fit_to_sigmoid_plot = p9.ggplot()
        #Hill_fit_to_sigmoid_plot+=p9.geom_line(Hill_table_plot, p9.aes(x="Stim_amp_pA", y="Frequency_Hz", group='Legend', color='Legend'))
        
        Hill_fit_to_sigmoid_plot+=p9.geom_line(Asc_sig_to_fit, p9.aes(x="Stim_amp_pA", y="Frequency_Hz", group='Legend', color='Legend'))
        Hill_fit_to_sigmoid_plot+=p9.geom_line(initial_Hill_extended_fit_table, p9.aes(x="Stim_amp_pA", y="Frequency_Hz", group='Legend', color='Legend'))
        Hill_fit_to_sigmoid_plot+=p9.geom_line(first_Hill_extended_fit_table, p9.aes(x="Stim_amp_pA", y="Frequency_Hz", group='Legend', color='Legend'))
        Hill_fit_to_sigmoid_plot+=p9.scale_color_manual(Hill_fit_color_dict)
        Hill_fit_to_sigmoid_plot+=p9.ggtitle("Hill_fit_to_Ascending_sigmoid")
        
        if do_plot:
            
            Hill_fit_to_Sigmoid_dict = {"Asc_sig_to_fit":Asc_sig_to_fit,
                                       'initial_Hill_extended_fit_table':initial_Hill_extended_fit_table, 
                                       'first_Hill_extended_fit_table' : first_Hill_extended_fit_table,
                                   
                                   'color_shape_dict' : Hill_fit_color_dict,
                                   }
            
            
            plot_list[f'{i}-Hill_fit_to_Sigmoid']=Hill_fit_to_Sigmoid_dict
            i+=1
            if print_plot:
            
                Hill_fit_to_sigmoid_plot.show()
        
        
        # Final fit
        #Fit Amplitude*Hill*Descending Sigmoid to original data points
        
        Final_amplitude_fit = ConstantModel(prefix='Final_Amp_')
        Final_amplitude_fit_pars = Final_amplitude_fit.make_params()
    
        Final_amplitude_fit_pars['Final_Amp_c'].set(value=best_H_amp)
        max_Hill_coef=math.log(1e308)/math.log(max(original_x_data)+100)
        Hill_model = Model(hill_function, prefix='Final_Hill_')
        Hill_pars = Parameters()
        Hill_pars.add("Final_Hill_x0",value=best_H_x0, min = min(original_x_data)-100)
        Hill_pars.add("Final_Hill_Half_cst",value=best_H_Half_cst)
        Hill_pars.add('Final_Hill_Hill_coef',value=best_H_Hill_coef, max= max_Hill_coef)
        Final_fit_model =  Final_amplitude_fit*Hill_model
        Final_fit_model_pars = Final_amplitude_fit_pars+Hill_pars

        if Descending_segment:
            Sigmoid_model = Model(sigmoid_function, prefix="Final_Sigmoid_")
            Sigmoid_pars = Parameters()
            
            Sigmoid_pars.add("Final_Sigmoid_x0",value=best_value_x0_desc)
            Sigmoid_pars.add("Final_Sigmoid_sigma",value=best_value_sigma_desc, min = best_value_sigma_desc, max= -1e-9)
            
            Final_fit_model *= Sigmoid_model
            Final_fit_model_pars += Sigmoid_pars
        
        Final_Fit_results = Final_fit_model.fit(original_y_data, Final_fit_model_pars, x=original_x_data)
        
        Final_fit_parameters = get_parameters_table(Final_fit_model_pars, Final_Fit_results)


        Final_fit_parameters.loc[:,'Fit'] = "Final fit to data"
        
        Final_Amp = Final_Fit_results.best_values['Final_Amp_c']
        
        Final_H_Half_cst=Final_Fit_results.best_values['Final_Hill_Half_cst']
        Final_H_Hill_coef=Final_Fit_results.best_values['Final_Hill_Hill_coef']
        Final_H_x0 = Final_Fit_results.best_values['Final_Hill_x0']
        
        if Descending_segment:
            Legend = "Final_Hill_Sigmoid_fit"
            obs = 'Hill-Sigmoid'
            Final_S_x0 = Final_Fit_results.best_values['Final_Sigmoid_x0']
            Final_S_sigma = Final_Fit_results.best_values['Final_Sigmoid_sigma']
            Final_fit_extended = Final_Amp * hill_function(extended_x_data, Final_H_x0, Final_H_Hill_coef, Final_H_Half_cst) * sigmoid_function(extended_x_data, Final_S_x0, Final_S_sigma)
            Final_fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Final_fit_extended})
            Final_fit_table['Legend'] = Legend
            
            Initial_Hill_sigmoid_fit = best_H_amp * hill_function(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst) * sigmoid_function(extended_x_data, best_value_x0_desc, best_value_sigma_desc)
            Initial_Fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Initial_Hill_sigmoid_fit})
            Initial_Fit_table['Legend'] = "Initial conditions"
            
        else:
            Legend = "Final_Hill_fit"
            obs = 'Hill'
            Final_fit_extended = Final_Amp * hill_function(extended_x_data, Final_H_x0, Final_H_Hill_coef, Final_H_Half_cst)
            Final_fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Final_fit_extended})
            Final_fit_table['Legend'] = Legend
            
            Initial_Hill_fit = best_H_amp * hill_function(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst) 
            Initial_Fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Initial_Hill_fit})
            Initial_Fit_table['Legend'] = "Initial conditions"
            Final_S_x0 = np.nan
            Final_S_sigma = np.nan
        
        Hill_sigmoid_color_dict = {"Final_Hill_Fit":"green",
                                   "Final_Hill_Sigmoid_fit" : "green",
                                   "Initial conditions" : "red"}
        
        parameters_table = pd.concat([third_order_fit_params_table, sigmoid_fit_parameters, Hill_fit_parameters, Final_fit_parameters], ignore_index= True)
        
        
        Hill_Sigmoid_plot =  p9.ggplot()
        Hill_Sigmoid_plot += p9.geom_point(original_data_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
        Hill_Sigmoid_plot += p9.geom_line(Initial_Fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz", group='Legend', color='Legend'))
        Hill_Sigmoid_plot += p9.geom_line(Final_fit_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz", group='Legend', color='Legend'))
        Hill_Sigmoid_plot += p9.scale_color_manual(Hill_sigmoid_color_dict)
        Hill_Sigmoid_plot += p9.ggtitle("Final_Hill_Sigmoid_Fit")
        if do_plot:
            
            Hill_Sigmoid_fit_dict = {"original_data_table":original_data_table,
                                       'Initial_Fit_table':Initial_Fit_table, 
                                       'Final_fit_table' : Final_fit_table,
                                       
                                   
                                   'color_shape_dict' : Hill_sigmoid_color_dict,
                                   }
            
            
            plot_list[f'{i}-Final_Hill_Sigmoid_Fit']=Hill_Sigmoid_fit_dict
            i+=1
            
            if print_plot:
                Hill_Sigmoid_plot.show()
        return obs,Final_fit_table,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table
    
    except(StopIteration):
          obs='Error_Iteration'

          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_sigma=np.nan
          Legend=np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table
             
    except (ValueError):
          obs='Error_Value'
          
          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_sigma=np.nan
          Legend=np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table
              
              
    except (RuntimeError):
          obs='Error_Runtime'

          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_sigma=np.nan
          Legend=np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table
             
    except (TypeError) as e:
          obs=str(e)

          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_sigma=np.nan
          Legend=np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table
             
    
def get_parameters_table(fit_model_params, result):
    
    initial_params = {param: fit_model_params[param].value for param in fit_model_params}
    result_params = {param: result.params[param].value for param in result.params} 
    min_params = {param: fit_model_params[param].min for param in fit_model_params}
    max_params = {param: fit_model_params[param].max for param in fit_model_params}
    parameters_table = pd.DataFrame({
        'Parameter': initial_params.keys(),
        'Initial Value': initial_params.values(),
        'Min Value': min_params.values(),
        'Max Value': max_params.values(),
        'Resulting Value': result_params.values()

    })
    
    return parameters_table
    
    
    

def extract_IO_features_Test(original_stimulus_frequency_table,response_type, response_duration, do_plot = False, print_plot = False):
    
    
    obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table = fit_IO_relationship_Test(original_stimulus_frequency_table,do_plot=do_plot,print_plot=print_plot)
   
    if obs == 'Hill-Sigmoid':

        Descending_segment = True
        feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table,response_type,Final_fit_extended,Final_Amp,Final_H_x0,Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0,Final_S_sigma,Descending_segment,Legend,response_duration, do_plot,plot_list=plot_list,print_plot=print_plot)
        

        if feature_obs == 'Hill-Sigmoid':
            if do_plot==True:

                return plot_list
            return feature_obs,Final_Amp,Final_H_Hill_coef,Final_H_Half_cst,Final_H_x0,Final_S_x0,Final_S_sigma,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation, parameters_table
        
        else:
            Descending_segment=False

            obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table = fit_IO_relationship_Test(original_stimulus_frequency_table,Force_single_Hill=True,do_plot=do_plot,print_plot=print_plot)
            
            if obs =='Hill':
                feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table,response_type,Final_fit_extended,Final_Amp,Final_H_x0,Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0,Final_S_sigma,Descending_segment,Legend,response_duration, do_plot,plot_list=plot_list,print_plot=print_plot)
                
            else:
                empty_array = np.empty(5)
                empty_array[:] = np.nan
                feature_obs=obs
                best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
            
    
    elif obs == "Hill":
        Descending_segment=False

        feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table,response_type,Final_fit_extended,Final_Amp,Final_H_x0,Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0,Final_S_sigma,Descending_segment,Legend,response_duration,do_plot,plot_list=plot_list,print_plot=print_plot)
        
        
    else:
        Descending_segment=False
        obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_sigma,Legend,plot_list, parameters_table = fit_IO_relationship_Test(original_stimulus_frequency_table,Force_single_Hill=True,do_plot=do_plot,print_plot=print_plot)
        
        if obs =='Hill':
            feature_obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list=extract_IO_features(original_stimulus_frequency_table,response_type,Final_fit_extended,Final_Amp,Final_H_x0,Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0,Final_S_sigma,Descending_segment,Legend,response_duration, do_plot,plot_list=plot_list,print_plot=print_plot)
            
        else:
            empty_array = np.empty(5)
            empty_array[:] = np.nan
            feature_obs=obs
            best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation = empty_array
            
    if do_plot:
        return plot_list
    return feature_obs,Final_Amp,Final_H_Hill_coef,Final_H_Half_cst,Final_H_x0,Final_S_x0,Final_S_sigma,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation, parameters_table
 
    
def fit_Hill_Sigmoid(original_stimulus_frequency_table, third_order_poly_model_results,ascending_segment,descending_segment,scale_dict,do_plot=False,plot_list=None,print_plot=False):
    '''
    Fit Hill-Sigmoid relationship to the Stimulus-Frequency values

    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.
        
    third_order_poly_model_results : dict
        Result from 3rd order polynomial fit.
        
    ascending_segment : pd.DataFrame
        Subset of table corresponding to the ascending segment of the 3rd order polynomial fit.
        
    descending_segment : pd.DataFrame
        Subset of table corresponding to the descending segment of the 3rd order polynomial fit.
        
    scale_dict : Dict
        Dictionnary of color and line type for plotting.
        
    do_plot : Boolean, optional
        Do plot. The default is False.
        
    plot_list : List, optional
        List of plot during the fitting procedures (required do_plot == True). The default is None.
        
    print_plot : Boolean, optional
        Print plot. The default is False.

    Returns
    -------
    obs : str
        Observation of the result of the fit.
        
    asc_Hill_desc_sigmoid_table_with_amp : pd.DataFrame
        Table of fitting results (Stimulus-Frequency table).
        
    best_value_Amp : float
        Results of the fitting procedure to reproduce the I/O curve fit
            
    best_H_x0 : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_H_Half_cst : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_H_Hill_coef : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_value_x0_desc : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_value_sigma_desc : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    Legend : str
        
    plot_list : List
        List of plot during the fitting procedures (required do_plot == True).

    '''
    try:   

        original_data_table =  original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        
        if do_plot == True:
            original_data_table=original_data_table.astype({'Passed_QC':bool
                                                            })

        
        original_data_subset_QC = original_data_table.copy()
        

        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        
       
                
        
        original_data_subset_QC=original_data_subset_QC.reset_index(drop=True)
        original_data_subset_QC=original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        original_x_data = np.array(original_data_subset_QC['Stim_amp_pA'])
        original_y_data = np.array(original_data_subset_QC['Frequency_Hz'])
        extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
        

        
    
        ### 0 - Trim Data
        trimmed_stimulus_frequency_table = original_data_subset_QC.copy()
        response_threshold = (np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
        
        for elt in range(trimmed_stimulus_frequency_table.shape[0]):
            if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
                trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
            else:
                break
        
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
        for elt in range(trimmed_stimulus_frequency_table.shape[0]):
            if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
                trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
            else:
                break
        
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_x_data=trimmed_stimulus_frequency_table.loc[:,'Stim_amp_pA']

        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
        best_c0 = third_order_poly_model_results.best_values['c0']
        best_c1 = third_order_poly_model_results.best_values['c1']
        best_c2 = third_order_poly_model_results.best_values['c2']
        best_c3 = third_order_poly_model_results.best_values['c3']
        
        extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
        trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                               "Frequency_Hz" : extended_trimmed_3rd_poly_model})
        trimmed_3rd_poly_table['Legend']='3rd_order_poly'
        
        
        
        ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
        ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
        
        descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
        descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
        
        trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
        trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
        
        
        max_freq_poly_fit = np.nanmax([np.nanmax(ascending_segment['Frequency_Hz']),np.nanmax(descending_segment['Frequency_Hz'])])
        #max_freq_poly_fit=np.nanmax(trimmed_3rd_poly_table['Frequency_Hz'])
        
        trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"

        trimmed_descending_segment.loc[trimmed_descending_segment.index,'Legend']="Trimmed_desc_seg"
        if do_plot:
            
            polynomial_plot = p9.ggplot()
            polynomial_plot += p9.geom_point(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
            polynomial_plot += p9.geom_point(trimmed_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
            polynomial_plot += p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += p9.ggtitle('3rd order polynomial fit to trimmed_data')
            polynomial_plot += p9.scale_color_manual(values=scale_dict)
            polynomial_plot += p9.scale_shape_manual(values=scale_dict)
            if print_plot==True:
                print(polynomial_plot)
            trimmed_segments_table = pd.concat([trimmed_ascending_segment,trimmed_descending_segment],ignore_index=True)
            
            polynomial_plot += p9.geom_line(trimmed_segments_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",colour='Legend',group="Legend"))
            polynomial_plot += p9.scale_color_manual(values=scale_dict)
            polynomial_plot += p9.scale_shape_manual(values=scale_dict)
            #polynomial_plot+=geom_line(trimmed_descending_segment,aes(x='Stim_amp_pA',y="Frequency_Hz"),colour='red')
            if print_plot==True:
                print(polynomial_plot)
        
        
        ascending_linear_slope_init,ascending_linear_intercept_init=linear_fit(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        ascending_sigmoid_slope=max_freq_poly_fit/(4*ascending_linear_slope_init)
        
        descending_linear_slope_init,descending_linear_intercept_init=linear_fit(trimmed_descending_segment["Stim_amp_pA"],
                                              trimmed_descending_segment['Frequency_Hz'])
        
        if do_plot:   
            polynomial_plot+=p9.geom_abline(slope=ascending_linear_slope_init,
                                         intercept=ascending_linear_intercept_init,
                                         colour="pink",
                                         linetype='dashed')
            
            polynomial_plot+=p9.geom_abline(slope=descending_linear_slope_init,
                                         intercept=descending_linear_intercept_init,
                                         colour="red",
                                         linetype='dashed')
            plot_list[str(str(len(plot_list.keys()))+"-3rd_order_polynomial_plot")]=polynomial_plot
            if print_plot==True:
                print(polynomial_plot)
        
        ### 2 - Fit Double Sigmoid : Amp*Sigmoid_asc*Sigmoid_desc
        
        # Ascending Sigmoid
        ascending_sigmoid_fit= Model(sigmoid_function,prefix='asc_')
        ascending_segment_fit_params  = ascending_sigmoid_fit.make_params()
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        # Amplitude
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*max_freq_poly_fit)
        
        # Amp*Ascending segment
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        # Descending segment
        descending_sigmoid_slope = max_freq_poly_fit/(4*descending_linear_slope_init)
        descending_sigmoid_fit = Model(sigmoid_function,prefix='desc_')
        descending_sigmoid_fit_pars = descending_sigmoid_fit.make_params()
        descending_sigmoid_fit_pars.add("desc_x0",value=np.nanmean(trimmed_descending_segment['Stim_amp_pA']))
        descending_sigmoid_fit_pars.add("desc_sigma",value=descending_sigmoid_slope,min=3*descending_sigmoid_slope,max=-1e-9)
        
        # Amp*Ascending_sigmoid*Descending_sigmoid
        ascending_sigmoid_fit *=descending_sigmoid_fit
        ascending_segment_fit_params+=descending_sigmoid_fit_pars
        
        # Fit model to original data
        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        # Get fit best parameters
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
        best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']
        
        # Create simulation table
        full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data, best_value_x0_asc, best_value_sigma_asc) * sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
        full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : full_sigmoid_fit})
        full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
        
        
        if do_plot:
            double_sigmoid_comps = ascending_sigmoid_fit_results.eval_components(x=original_x_data)
            asc_sig_comp = double_sigmoid_comps['asc_']
            asc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : asc_sig_comp,
                                        'Legend' : 'Ascending_Sigmoid'})
            
            asc_sig_comp_table['Frequency_Hz'] = asc_sig_comp_table['Frequency_Hz']*max(original_data_table['Frequency_Hz'])/max(original_data_table['Frequency_Hz'])
            amp_comp = double_sigmoid_comps['Amp_']
            amp_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : amp_comp,
                                        'Legend' : "Amplitude"})
            component_table = pd.concat([asc_sig_comp_table,amp_comp_table],ignore_index=True)
            component_plot = p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point(p9.aes(shape='Passed_QC'))
           
            
            sigmoid_fit_plot =  p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point(p9.aes(shape='Passed_QC'))
            sigmoid_fit_plot += p9.geom_line(full_sigmoid_fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
            sigmoid_fit_plot += p9.geom_line(trimmed_ascending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            #sigmoid_fit_plot += geom_line(component_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            
            sigmoid_fit_plot += p9.ggtitle("Sigmoid_fit_to_original_data")
            
            desc_sig_comp = double_sigmoid_comps['desc_']
            desc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : desc_sig_comp,
                                        'Legend' : "Descending_Sigmoid"})
            desc_sig_comp_table['Frequency_Hz'] = desc_sig_comp_table['Frequency_Hz']*max(original_data_table['Frequency_Hz'])/max(desc_sig_comp_table['Frequency_Hz'])
            component_table = pd.concat([component_table,desc_sig_comp_table],ignore_index=True)
            
            sigmoid_fit_plot += p9.geom_line(trimmed_descending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            component_plot += p9.geom_line(trimmed_descending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        
            component_plot += p9.geom_line(trimmed_ascending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            component_plot += p9.geom_line(component_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            component_plot += p9.ggtitle('Sigmoid_fit_components')
            component_plot += p9.scale_color_manual(values=scale_dict)
            component_plot += p9.scale_shape_manual(values=scale_dict)
            plot_list[str(str(len(plot_list.keys()))+"-Sigmoid_fit_components")]=component_plot
            if print_plot==True:   
                print(component_plot)
            
            
            sigmoid_fit_plot += p9.scale_color_manual(values=scale_dict)
            sigmoid_fit_plot += p9.scale_shape_manual(values=scale_dict)
            plot_list[str(str(len(plot_list.keys()))+"-Sigmoid_fit_to_original_data")]=sigmoid_fit_plot
            if print_plot==True:
                print(sigmoid_fit_plot)
            
        #Determine Hill half coef for first hill fit --> First stim_amp to elicit half max(frequency)
        half_response_index=np.argmax(original_y_data>=(0.5*max(original_y_data)))
       
        half_response_stim = original_x_data[half_response_index]
        
        # Set limit to Hill coef to prevent the parameter space exporation to go too high which breaks the fit
        max_Hill_coef=math.log(1e308)/math.log(max(original_x_data)+100)
    
       
        # Create Hill model to fit on ascending sigmoid
        first_Hill_model = Model(hill_function)
        first_Hill_pars = first_Hill_model.make_params()
        first_Hill_pars.add("x0",value=np.nanmin(original_x_data),min=np.nanmin(original_x_data)-100,max=np.nanmax(original_x_data)+100)
        first_Hill_pars.add("Half_cst",value=half_response_stim,min=1e-9)
        first_Hill_pars.add('Hill_coef',value=1.2,max=max_Hill_coef, min=1e-9)
       
        
        asc_sigmoid_to_fit = sigmoid_function(original_x_data , best_value_x0_asc, best_value_sigma_asc)
        
        
    
        first_Hill_result = first_Hill_model.fit(asc_sigmoid_to_fit, first_Hill_pars, x=original_x_data)
        
        
        best_H_Half_cst=first_Hill_result.best_values['Half_cst']
        best_H_Hill_coef=first_Hill_result.best_values['Hill_coef']
        best_H_x0 = first_Hill_result.best_values['x0']
        
        first_Hill_extended_fit = hill_function(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        first_Hill_extended_fit_table=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : first_Hill_extended_fit})
        first_Hill_extended_fit_table['Legend'] = "First_Hill_Fit"
        
        if do_plot:
            asc_sigmoid_fit_table = pd.DataFrame({'Stim_amp_pA':original_x_data,
                                                        'Frequency_Hz' : asc_sigmoid_to_fit})
            asc_sigmoid_fit_table['Legend'] = 'Ascending_Sigmoid'
            
            first_hill_fit_plot = p9.ggplot()
            first_hill_fit_plot += p9.geom_point(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'))
            Hill_asc_table = pd.concat([first_Hill_extended_fit_table,asc_sigmoid_fit_table],ignore_index=True)
            first_hill_fit_plot += p9.geom_line(Hill_asc_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            
            
            first_hill_fit_plot += p9.ggtitle("Ascending_Hill_Fit_to_asc_sigmoid")
            first_hill_fit_plot += p9.scale_color_manual(values=scale_dict)
            plot_list[str(str(len(plot_list.keys()))+"-Ascending_Hill_Fit_to_asc_sigmoid")]=first_hill_fit_plot
            if print_plot==True:
                print(first_hill_fit_plot)
            
        best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
        best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']       
        
    
        Hill_model = Model(hill_function, prefix='Hill_')
        Hill_pars = Parameters()
        Hill_pars.add("Hill_x0",value=best_H_x0,min=best_H_x0-0.5*best_H_Half_cst, max=best_H_x0+0.5*best_H_Half_cst)
        Hill_pars.add("Hill_Half_cst",value=best_H_Half_cst, min=0, max= 5*best_H_Half_cst)
        Hill_pars.add('Hill_Hill_coef',value=best_H_Hill_coef,vary = False)
        
        
        
        
        
        Sigmoid_model = Model(sigmoid_function, prefix="Sigmoid_")
        Sigmoid_pars = Parameters()
        
        Sigmoid_pars.add("Sigmoid_x0",value=best_value_x0_desc)
        Sigmoid_pars.add("Sigmoid_sigma",value=best_value_sigma_desc)
        
        Hill_Sigmoid_model = Hill_model*Sigmoid_model
        Hill_Sigmoid_pars = Hill_pars+Sigmoid_pars
        
        asc_Hill_desc_sigmoid_to_be_fit = hill_function(original_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        asc_Hill_desc_sigmoid_to_be_fit *= sigmoid_function(original_x_data , best_value_x0_desc, best_value_sigma_desc)
        
        # fit a Hill*sigmoid to First_Hill*Desc_Sigmoid
        
        Hill_Sigmoid_result = Hill_Sigmoid_model.fit(asc_Hill_desc_sigmoid_to_be_fit, Hill_Sigmoid_pars, x=original_x_data)
        
        H_Half_cst=Hill_Sigmoid_result.best_values['Hill_Half_cst']
        H_Hill_coef=Hill_Sigmoid_result.best_values['Hill_Hill_coef']
        H_x0 = Hill_Sigmoid_result.best_values['Hill_x0']
        
    
        S_x0 = Hill_Sigmoid_result.best_values['Sigmoid_x0']
        S_sigma = Hill_Sigmoid_result.best_values['Sigmoid_sigma']
        
        Hill_Sigmoid_model_results = hill_function(extended_x_data, H_x0, H_Hill_coef, H_Half_cst)
        Hill_Sigmoid_model_results *= sigmoid_function(extended_x_data , S_x0, S_sigma)
        
        
        
        asc_Hill_desc_sigmoid_extended = hill_function(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        asc_Hill_desc_sigmoid_extended *= sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
        asc_Hill_desc_sigmoid_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                  "Frequency_Hz" : asc_Hill_desc_sigmoid_extended})
        asc_Hill_desc_sigmoid_table['Legend'] = "Asc_Hill_Desc_Sigmoid"
        
        Hill_Sigmoid_model_results_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                  "Frequency_Hz" : Hill_Sigmoid_model_results})
        Hill_Sigmoid_model_results_table['Legend'] = 'Hill_Sigmoid_Fit'
        
        
        Hill_Sigmoid_plot =  p9.ggplot()
        Hill_Sigmoid_plot += p9.geom_line(asc_Hill_desc_sigmoid_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Hill_Sigmoid_plot += p9.geom_line(Hill_Sigmoid_model_results_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Hill_Sigmoid_plot += p9.ggtitle("Hill_Sigmoid_Fit_to_Asc_Hill_Desc_Sigm")
        Hill_Sigmoid_plot += p9.scale_color_manual(values=scale_dict)
        if do_plot:
            plot_list[str(str(len(plot_list.keys()))+"-Hill_Sigmoid_Fit_to_Asc_Hill_Desc_Sigm")]=Hill_Sigmoid_plot
            if print_plot==True:
                print(Hill_Sigmoid_plot)
        
        asc_Hill_desc_sigmoid_table_with_amp = asc_Hill_desc_sigmoid_table.copy()
        asc_Hill_desc_sigmoid_table_with_amp['Frequency_Hz'] *= best_value_Amp
        
        Hill_Sigmoid_model_results_table_with_amplitude = Hill_Sigmoid_model_results_table.copy()
        Hill_Sigmoid_model_results_table_with_amplitude['Frequency_Hz']*=best_value_Amp
        
        Hill_Sigmoid_plot =  p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point(p9.aes(shape='Passed_QC'))
        Hill_Sigmoid_plot += p9.geom_line(Hill_Sigmoid_model_results_table_with_amplitude,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Hill_Sigmoid_plot += p9.ggtitle("Final_Hill_Sigmoid_Fit_mytest")
        Hill_Sigmoid_plot += p9.scale_color_manual(values=scale_dict)
        Hill_Sigmoid_plot += p9.scale_shape_manual(values=scale_dict)
        
        if do_plot:
            plot_list[str(str(len(plot_list.keys()))+"-Final_Hill_Sigmoid_Fit_mytest")]=Hill_Sigmoid_plot
            if print_plot==True:   
                print(Hill_Sigmoid_plot)
            
        
        

        Legend="Hill_Sigmoid_Fit"
        
        
        obs = 'Hill-Sigmoid'
        
        return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend,plot_list
        
    except(StopIteration):
         obs='Error_Iteration'

         asc_Hill_desc_sigmoid_table_with_amp=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         best_value_x0_desc=np.nan
         best_value_sigma_desc=np.nan
         Legend=np.nan
         return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend,plot_list
             
    except (ValueError):
          obs='Error_Value'
          
          asc_Hill_desc_sigmoid_table_with_amp=np.nan
          best_value_Amp=np.nan
          best_H_Hill_coef=np.nan
          best_H_Half_cst=np.nan
          best_H_x0=np.nan
          best_value_x0_desc=np.nan
          best_value_sigma_desc=np.nan
          Legend=np.nan
          
          return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend,plot_list
              
              
    except (RuntimeError):
         obs='Error_Runtime'

         asc_Hill_desc_sigmoid_table_with_amp=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         best_value_x0_desc=np.nan
         best_value_sigma_desc=np.nan
         Legend=np.nan
         return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend,plot_list
             
    except (TypeError) as e:
         obs=str(e)

         asc_Hill_desc_sigmoid_table_with_amp=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         best_value_x0_desc=np.nan
         best_value_sigma_desc=np.nan
         Legend=np.nan
         return obs,asc_Hill_desc_sigmoid_table_with_amp,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,best_value_x0_desc, best_value_sigma_desc,Legend,plot_list
    
             
def fit_Single_Hill(original_stimulus_frequency_table, scale_dict,do_plot=False,plot_list=None,print_plot=False):
    '''
    Fit Hill function to the Stimulus-Frequency values

    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.
        
    scale_dict : Dict
        Dictionnary of color and line type for plotting.
        
    do_plot : Boolean, optional
        Do plot. The default is False.
        
    plot_list : List, optional
        List of plot during the fitting procedures (required do_plot == True). The default is None.
        
    print_plot : Boolean, optional
        Print plot. The default is False.
        
    Returns
    -------
    obs : str
        Observation of the result of the fit.
        
    Final_Hill_fit : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_value_Amp : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_H_x0 : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_H_Half_cst : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_H_Hill_coef : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    Legend : str
        
    plot_list : List
        List of plot during the fitting procedures (required do_plot == True).

    '''
    try:


        original_data_table =  original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        if do_plot == True:
            original_data_table=original_data_table.astype({'Passed_QC':bool
                                                            })
        original_data_subset_QC = original_data_table.copy()

        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        
        original_data_subset_QC=original_data_subset_QC.reset_index(drop=True)
        original_data_subset_QC=original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        original_x_data = np.array(original_data_table['Stim_amp_pA'])
        original_y_data = np.array(original_data_table['Frequency_Hz'])
        extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))

        trimmed_stimulus_frequency_table = original_data_subset_QC.copy()
        ### 0 - Trim Data
        response_threshold = (np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
        
        for elt in range(trimmed_stimulus_frequency_table.shape[0]):
            if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
                trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
            else:
                break
        
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
        for elt in range(trimmed_stimulus_frequency_table.shape[0]):
            if trimmed_stimulus_frequency_table.iloc[0,2] < response_threshold :
                trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.drop(trimmed_stimulus_frequency_table.index[0])
            else:
                break
        
        ### 0 - end
        
        ### 1 - Try to fit a 2nd order polynomial
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_x_data=trimmed_stimulus_frequency_table.loc[:,'Stim_amp_pA']
        trimmed_y_data=trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]
        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
        
        second_order_poly_model = PolynomialModel(degree = 2)
        pars = second_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
        second_order_poly_model_results = second_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
    
        best_c0 = second_order_poly_model_results.best_values['c0']
        best_c1 = second_order_poly_model_results.best_values['c1']
        best_c2 = second_order_poly_model_results.best_values['c2']
        
        extended_trimmed_2nd_poly_model = best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
        extended_trimmed_2nd_poly_model_freq_diff=np.diff(extended_trimmed_2nd_poly_model)
        trimmed_2nd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                               "Frequency_Hz" : extended_trimmed_2nd_poly_model})
        
       
        zero_crossings_2nd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_2nd_poly_model_freq_diff)))[0] # detect last index before change of sign
        
        if len(zero_crossings_2nd_poly_model_freq_diff) == 1:
    
            if extended_trimmed_2nd_poly_model_freq_diff[0]<0:
                trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] >= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
            else:
                trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] <= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
            ascending_segment = trimmed_2nd_poly_table.copy()
        else:    
            ascending_segment = trimmed_2nd_poly_table.copy()
        ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
        ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
        
        trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
        max_freq_poly_fit=np.nanmax(trimmed_2nd_poly_table['Frequency_Hz'])
        
        trimmed_2nd_poly_table.loc[trimmed_2nd_poly_table.index,'Legend']='2nd_order_poly'
        trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"
        if do_plot:
            
            polynomial_plot = p9.ggplot(trimmed_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point(p9.aes(shape='Passed_QC'))
            polynomial_plot += p9.geom_line(trimmed_2nd_poly_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += p9.geom_line(trimmed_ascending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            polynomial_plot += p9.ggtitle('2nd order polynomial fit to trimmed_data')
            polynomial_plot += p9.scale_shape_manual(values=scale_dict)
            plot_list[str(str(len(plot_list.keys()))+"-2nd_order_polynomial_fit_to_trimmed_data")]=polynomial_plot
            if print_plot==True:
                print(polynomial_plot)
        ### 2 - end 
        
        ### 3 - Linear fit on polynomial trimmed data
        ascending_linear_slope_init,ascending_linear_intercept_init=linear_fit(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        ascending_sigmoid_slope=max_freq_poly_fit/(4*ascending_linear_slope_init)
        
        if do_plot:   
            polynomial_plot+=p9.geom_abline(slope=ascending_linear_slope_init,
                                         intercept=ascending_linear_intercept_init,
                                         colour="pink",
                                         linetype='dashed')
            
        ### 3 - end
        
        ### 4 - Fit single Sigmoid
        
        ascending_sigmoid_fit= Model(sigmoid_function,prefix='asc_')
        ascending_segment_fit_params  = ascending_sigmoid_fit.make_params()
        
        
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*max_freq_poly_fit)
        
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        
        full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data , best_value_x0_asc, best_value_sigma_asc)
        full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : full_sigmoid_fit})
        
        full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
        
        if do_plot:
            
            
            double_sigmoid_comps = ascending_sigmoid_fit_results.eval_components(x=original_x_data)
            asc_sig_comp = double_sigmoid_comps['asc_']
            asc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : asc_sig_comp,
                                        'Legend' : 'Ascending_Sigmoid'})
            
            asc_sig_comp_table['Frequency_Hz'] = asc_sig_comp_table['Frequency_Hz']*max(original_data_table['Frequency_Hz'])/max(asc_sig_comp_table['Frequency_Hz'])
            amp_comp = double_sigmoid_comps['Amp_']
            amp_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : amp_comp,
                                        'Legend' : "Amplitude"})
            
            component_table = pd.concat([asc_sig_comp_table,amp_comp_table],ignore_index = True)
            component_plot = p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point(p9.aes(shape='Passed_QC'))
           
            #trimmed_ascending_segment["Stim_amp_pA"]-=x_shift
            sigmoid_fit_plot =  p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point(p9.aes(shape='Passed_QC'))
            sigmoid_fit_plot += p9.geom_line(full_sigmoid_fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
            sigmoid_fit_plot += p9.geom_line(trimmed_ascending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            #sigmoid_fit_plot += geom_line(component_table,aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            
            sigmoid_fit_plot += p9.ggtitle("Sigmoid_fit_to_original_data")

            component_plot += p9.geom_line(trimmed_ascending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            component_plot += p9.geom_line(component_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            component_plot += p9.ggtitle('Sigmoid_fit_components')
            component_plot += p9.scale_color_manual(values=scale_dict)
            component_plot += p9.scale_shape_manual(values=scale_dict)
            plot_list[str(str(len(plot_list.keys()))+"-Sigmoid_fit_components")]=component_plot
            if print_plot==True:
                print(component_plot)
            
            
            sigmoid_fit_plot += p9.scale_color_manual(values=scale_dict)
            sigmoid_fit_plot += p9.scale_shape_manual(values=scale_dict)
            plot_list[str(str(len(plot_list.keys()))+"-Sigmoid_fit_to_original_data")]=sigmoid_fit_plot
            if print_plot==True:
                print(sigmoid_fit_plot)
        
        ### 4 - end
        
        ### 5 - Fit Hill function to ascending Sigmoid (without considering amplitude)
        
        #Determine Hill half coef for first hill fit --> First stim_amp to elicit half max(frequency)
        half_response_index=np.argmax(original_y_data>=(0.5*max(original_y_data)))

        half_response_stim = original_x_data[half_response_index]

        max_Hill_coef=math.log(1e308)/math.log(max(original_x_data)+100)
    
       

        first_Hill_model = Model(hill_function)
        first_Hill_pars = first_Hill_model.make_params()
        first_Hill_pars.add("x0",value=np.nanmin(original_x_data),min=np.nanmin(original_x_data)-100,max=np.nanmax(original_x_data)+100)
        first_Hill_pars.add("Half_cst",value=half_response_stim,min=1e-9)
        first_Hill_pars.add('Hill_coef',value=1.2,min=1e-9,max=max_Hill_coef)

        

        asc_sigmoid_to_fit = sigmoid_function(original_x_data , best_value_x0_asc, best_value_sigma_asc)
       
        first_Hill_result = first_Hill_model.fit(asc_sigmoid_to_fit, first_Hill_pars, x=original_x_data)

        

        best_H_Half_cst=first_Hill_result.best_values['Half_cst']
        best_H_Hill_coef=first_Hill_result.best_values['Hill_coef']
        best_H_x0 = first_Hill_result.best_values['x0']
        
        first_Hill_extended_fit = hill_function(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        first_Hill_extended_fit_table=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : first_Hill_extended_fit})
        first_Hill_extended_fit_table['Legend'] = "First_Hill_Fit"
        
        if do_plot:
            asc_sigmoid_fit_table = pd.DataFrame({'Stim_amp_pA':original_x_data,
                                                        'Frequency_Hz' : asc_sigmoid_to_fit})
            asc_sigmoid_fit_table['Legend'] = 'Ascending_Sigmoid'
            
            first_hill_fit_plot = p9.ggplot()
            Hill_asc_table = pd.concat([first_Hill_extended_fit_table,asc_sigmoid_fit_table],ignore_index=True)
            
            first_hill_fit_plot += p9.geom_line(Hill_asc_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            
            
            first_hill_fit_plot += p9.ggtitle("Hill_Fit_to_ascending_Sigmoid")
            first_hill_fit_plot += p9.scale_color_manual(values=scale_dict)
            plot_list[str(str(len(plot_list.keys()))+"-Hill_Fit_to_ascending_Sigmoid")]=first_hill_fit_plot
            if print_plot==True:
                print(first_hill_fit_plot)
            
        Final_Hill_fit = hill_function(extended_x_data, best_H_x0, best_H_Hill_coef, best_H_Half_cst)
        Final_Hill_fit=pd.DataFrame({'Stim_amp_pA':extended_x_data,
                                                    'Frequency_Hz' : first_Hill_extended_fit})
        Final_Hill_fit['Legend'] = "Hill_Fit"
        Final_Hill_fit['Frequency_Hz'] *= best_value_Amp
        
        Final_Hill_plot =  p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point(p9.aes(shape='Passed_QC'))
        
        Final_Hill_plot += p9.geom_line(Final_Hill_fit,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
        Final_Hill_plot += p9.ggtitle("Final_Hill_Fit")
        Final_Hill_plot += p9.scale_color_manual(values=scale_dict)
        Final_Hill_plot += p9.scale_shape_manual(values=scale_dict)
        if do_plot:
            plot_list[str(str(len(plot_list.keys()))+"-Final_Hill_Fit")]=Final_Hill_plot
            if print_plot==True:
                print(Final_Hill_plot)
            
        
        
        Legend='Hill_Fit'

        obs = 'Hill'
        
        
        return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend,plot_list
        
    except(StopIteration):
         obs='Error_Iteration'
         Final_Hill_fit=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         Legend='--'
         
         
         return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend,plot_list
             
    except (ValueError):
          obs='Error_Value'
          Final_Hill_fit=np.nan
          best_value_Amp=np.nan
          best_H_Hill_coef=np.nan
          best_H_Half_cst=np.nan
          best_H_x0=np.nan
          Legend='--'
          
          
          return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend,plot_list
              
    except (RuntimeError):
         obs='Error_Runtime'
         Final_Hill_fit=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         Legend='--'
         
         
         return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend,plot_list
             
    except (TypeError):
         obs='Type_Error'
         Final_Hill_fit=np.nan
         best_value_Amp=np.nan
         best_H_Hill_coef=np.nan
         best_H_Half_cst=np.nan
         best_H_x0=np.nan
         Legend='--'

         
         return obs,Final_Hill_fit,best_value_Amp,best_H_x0, best_H_Half_cst,best_H_Hill_coef,Legend,plot_list
    
def extract_IO_features(original_stimulus_frequency_table,response_type,fit_model_table,Amp,Hill_x0,Hill_Half_cst,Hill_coef,sigmoid_x0,sigmoid_sigma,Descending_segment,Legend,response_duration, do_plot=False,plot_list=None,print_plot=False):
    '''
    From the fitted IO relationship, computed neuronal firing features 

    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.
            
    fit_model_table : pd.DataFrame
        Table of resulting fit (Stimulus-Frequency) .
        
    Amp, Hill_x0 , Hill_Half_cst, Hill_coef, sigmoid_x0, sigmoid_sigma : Float
        Results of the fitting procedure to reproduce the I/O curve fit.
        
    Descending_segment : Boolean
        Wether a Descending segment has been detected.
        
    Legend : str

    do_plot : Boolean, optional
        Do plot. The default is False.
   	        
    plot_list : List, optional
        List of plot during the fitting procedures (required do_plot == True). The default is None.
   	        
    print_plot : Boolean, optional
        Print plot. The default is False.

    Returns
    -------
    obs : str
        Observation of the result of the fit.
        
    best_QNRMSE : Float
        Godness of fit.
        
    Gain : Float
        Neuronal Gain, slope of the linear portion of the ascending segment .
        
    Threshold : Float
        Firing threshold, x-axis intercept of the fit to the linear portion of the ascending segment .
        
    Saturation_frequency : Float
        Firing Saturation (if any, otherwise nan), maximum frequency of the IO fit.
        
    Saturation_stimulation : Float
        Stimulus eliciting Saturation (if any, otherwise nan), Stimulus amplitude eleiciting maximum frequency of the IO fit.
        
    plot_list : List
        List of plot during the fitting procedures (required do_plot == True).

    '''
    try:
        scale_dict={True:"o",
                    False:'s'}
        if Descending_segment:
            obs='Hill-Sigmoid'
        else:
            obs='Hill'


        original_data_table=original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        
        if do_plot == True:
            original_data_table=original_data_table.astype({'Passed_QC':bool})
        original_data_subset_QC = original_data_table.copy()
        
        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        

        original_x_data = np.array(original_data_subset_QC['Stim_amp_pA'])
        original_y_data = np.array(original_data_subset_QC['Frequency_Hz'])

        extended_x_data= fit_model_table['Stim_amp_pA']
        predicted_y_data = fit_model_table['Frequency_Hz']

        
        my_derivative = np.array(predicted_y_data)[1:]-np.array(predicted_y_data)[:-1]

        twentyfive_index=np.argmax(predicted_y_data>(0.25*(max(predicted_y_data)-min(predicted_y_data))+min(predicted_y_data)))
        seventyfive_index=np.argmax(predicted_y_data>(0.75*(max(predicted_y_data)-min(predicted_y_data))+min(predicted_y_data)))
        
        Gain,Intercept=linear_fit(extended_x_data.iloc[twentyfive_index:seventyfive_index],predicted_y_data.iloc[twentyfive_index:seventyfive_index])

        #Threshold=(0-Intercept)/Gain
        if response_type == 'Time_based':
            
            minimum_frequency = 1/response_duration
        else:
            minimum_frequency = 1
        filtered_df = fit_model_table.loc[fit_model_table['Frequency_Hz'] > minimum_frequency,:]
        Threshold = filtered_df['Stim_amp_pA'].min()


        model_table=pd.DataFrame(np.column_stack((extended_x_data,predicted_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])
        model_table['Legend']=Legend
        
        
        if np.nanmean(my_derivative[-10:]) <= np.nanmax(my_derivative[twentyfive_index:seventyfive_index])*.01:

            Saturation_frequency = np.nanmax(predicted_y_data)

            sat_model_table=fit_model_table[fit_model_table["Frequency_Hz"] == Saturation_frequency]

            Saturation_frequency=sat_model_table['Frequency_Hz'].values[0]
            Saturation_stimulation=sat_model_table['Stim_amp_pA'].values[0]

            
        else:
            
            Saturation_frequency=np.nan
            Saturation_stimulation=np.nan
            
            
        ### Compute QNRMSE
        
        if len(np.flatnonzero(original_y_data))>0:
            without_zero_index=np.flatnonzero(original_y_data)[0]

        else:
            without_zero_index=original_y_data.iloc[0]

        sub_x_data=original_x_data[without_zero_index:]
        sub_y_data=original_y_data[without_zero_index:]
        new_x_data_without_zero=pd.Series(np.arange(min(sub_x_data),max(sub_x_data),1))
        
        pred = hill_function(sub_x_data, Hill_x0, Hill_coef, Hill_Half_cst)
        

        pred *= Amp
        
        pred_without_zero = hill_function(new_x_data_without_zero, Hill_x0, Hill_coef, Hill_Half_cst)
        
        pred_without_zero *= Amp
        
        if Descending_segment:
            pred *= sigmoid_function(sub_x_data , sigmoid_x0, sigmoid_sigma)
            pred_without_zero *= sigmoid_function(new_x_data_without_zero ,sigmoid_x0, sigmoid_sigma)

        best_QNRMSE=normalized_root_mean_squared_error(sub_y_data,pred,pred_without_zero)


        if do_plot:

            feature_plot=p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))+p9.geom_point(p9.aes(shape='Passed_QC'))

            feature_plot+=p9.geom_line(model_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz',color='Legend',group='Legend'))
    
            feature_plot+=p9.geom_abline(p9.aes(intercept=Intercept,slope=Gain))
            
            Threshold_table=pd.DataFrame({'Stim_amp_pA':[Threshold],'Frequency_Hz':[0]})
            feature_plot+=p9.geom_point(Threshold_table,p9.aes(x=Threshold_table["Stim_amp_pA"],y=Threshold_table["Frequency_Hz"]),color='green')
            feature_plot_dict={'original_data_table':original_data_table,
                               "model_table" : model_table,
                               'intercept':Intercept,
                               'Gain':Gain,
                               "Threshold" : Threshold_table}
            
            if not np.isnan(Saturation_frequency):
                feature_plot += p9.geom_point(sat_model_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'),color="green")
                feature_plot_dict['Saturation'] = sat_model_table
            feature_plot += p9.scale_shape_manual(values=scale_dict)
            
            
            plot_list[f"{len(plot_list.keys())+1}-IO_fit"]=feature_plot_dict

            #plot_list[str(str(len(plot_list.keys()))+"-IO_fit")]=feature_plot
            if print_plot==True:
                print(feature_plot)
    
        
            
        return obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list
            
    except (TypeError) as e:
        
        obs=str(e)
        
        best_QNRMSE=np.nan
        Gain=np.nan
        Threshold=np.nan
        Saturation_frequency=np.nan
        Saturation_stimulation=np.nan
        
        return obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list
    except (ValueError) as e:
        
        obs=str(e)
        
        best_QNRMSE=np.nan
        Gain=np.nan
        Threshold=np.nan
        Saturation_frequency=np.nan
        Saturation_stimulation=np.nan
        return obs,best_QNRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,plot_list


def plot_IO_fit(plot_list, plot_to_do, return_plot=False):
    

    symbol_map = {True: 'circle', False: 'circle-x-open'}
    if "-Polynomial_fit" in plot_to_do:
        polynomial_fit_dict = plot_list["1-Polynomial_fit"]
        # Extract data from dictionary
        original_data_table = polynomial_fit_dict["original_data_table"]
        trimmed_stimulus_frequency_table = polynomial_fit_dict['trimmed_stimulus_frequency_table']
        trimmed_3rd_poly_table = polynomial_fit_dict['trimmed_3rd_poly_table']
        color_shape_dict = polynomial_fit_dict['color_shape_dict']

        
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
        
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle'),
            name='Original',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open'),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        # Add trimmed stimulus frequency points
        fig.add_trace(go.Scatter(
            x=trimmed_stimulus_frequency_table['Stim_amp_pA'],
            y=trimmed_stimulus_frequency_table['Frequency_Hz'],
            mode='markers',
            marker=dict(color='black'),
            name='Trimmed Stimulus Frequency',
            text=trimmed_stimulus_frequency_table['Passed_QC']
        ))
        
        # Add polynomial fit lines
        for legend in trimmed_3rd_poly_table['Legend'].unique():
            legend_data = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Legend'] == legend]
            fig.add_trace(go.Scatter(
                x=legend_data['Stim_amp_pA'],
                y=legend_data['Frequency_Hz'],
                mode='lines',
                name=legend,
                line=dict(color=color_shape_dict[legend])
            ))
        
        # Update layout
        fig.update_layout(
            title='3rd order polynomial fit to trimmed_data',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz',
            legend_title='Legend'
        )
    
    elif "-Trimmed_polynomial_fit" in plot_to_do:
        trimmed_polynomial_fit_dict = plot_list['2-Trimmed_polynomial_fit']
        original_data_table = trimmed_polynomial_fit_dict["original_data_table"]
        trimmed_stimulus_frequency_table = trimmed_polynomial_fit_dict['trimmed_stimulus_frequency_table']
        trimmed_3rd_poly_table = trimmed_polynomial_fit_dict['trimmed_3rd_poly_table']
        ascending_linear_slope_init = trimmed_polynomial_fit_dict['ascending_linear_slope_init']
        ascending_linear_intercept_init = trimmed_polynomial_fit_dict['ascending_linear_intercept_init']
        color_shape_dict = trimmed_polynomial_fit_dict['color_shape_dict']
        trimmed_data_table = trimmed_polynomial_fit_dict['trimmed_data_table']
        
        is_Descending_Segment = False
        for key in trimmed_polynomial_fit_dict.keys():
            if "descending_segment" in key:
                is_Descending_Segment = True
                descending_linear_slope_init = trimmed_polynomial_fit_dict['descending_linear_slope_init']
                descending_linear_intercept_init = trimmed_polynomial_fit_dict['descending_linear_intercept_init']
                break
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
        
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle',color='grey'),
            name='Original Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open', color='grey'),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        # Add trimmed stimulus frequency points
        fig.add_trace(go.Scatter(
            x=trimmed_stimulus_frequency_table['Stim_amp_pA'],
            y=trimmed_stimulus_frequency_table['Frequency_Hz'],
            mode='markers',
            marker=dict(color='black',symbol=[symbol_map[val] for val in original_data_table['Passed_QC']]),
            name='Trimmed Stimulus Frequency',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        # Add polynomial fit lines
        fig.add_trace(go.Scatter(
            x=trimmed_3rd_poly_table['Stim_amp_pA'],
            y=trimmed_3rd_poly_table['Frequency_Hz'],
            mode='lines',
            line=dict(color='black'),
            name='3rd Poly Fit'
        ))
        
        
        
        # Add descending linear fit if present
        ascending_segment_trimmed_data_table = trimmed_data_table.loc[trimmed_data_table['Legend'] == "Trimmed ascending segment",:]
        fig.add_trace(go.Scatter(
            x=ascending_segment_trimmed_data_table['Stim_amp_pA'],
            y=ascending_segment_trimmed_data_table['Frequency_Hz'],
            mode='lines',
            line=dict(color='red'),
            name='Trimmed ascending segment'
        ))
        
        if is_Descending_Segment == True:
            descending_segment_trimmed_data_table = trimmed_data_table.loc[trimmed_data_table['Legend'] == "Trimmed descending segment",:]
            fig.add_trace(go.Scatter(
                x=descending_segment_trimmed_data_table['Stim_amp_pA'],
                y=descending_segment_trimmed_data_table['Frequency_Hz'],
                mode='lines',
                line=dict(color='pink'),
                name='Trimmed descending segment'
            ))
        
        # Add ascending linear fit
        fig.add_trace(go.Scatter(
            x=np.array([min(trimmed_stimulus_frequency_table['Stim_amp_pA']), max(trimmed_stimulus_frequency_table['Stim_amp_pA'])]),
            y=ascending_linear_slope_init * np.array([min(trimmed_stimulus_frequency_table['Stim_amp_pA']), max(trimmed_stimulus_frequency_table['Stim_amp_pA'])]) + ascending_linear_intercept_init,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Ascending Linear Fit'
        ))
        if is_Descending_Segment == True:
            fig.add_trace(go.Scatter(
                x=np.array([min(trimmed_stimulus_frequency_table['Stim_amp_pA']), max(trimmed_stimulus_frequency_table['Stim_amp_pA'])]),
                y=descending_linear_slope_init * np.array([min(trimmed_stimulus_frequency_table['Stim_amp_pA']), max(trimmed_stimulus_frequency_table['Stim_amp_pA'])]) + descending_linear_intercept_init,
                mode='lines',
                line=dict(color='pink', dash='dash'),
                name='Descending Linear Fit'
            ))
            
        # Update layout
        fig.update_layout(
            title='Trimmed segments',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz',
            legend_title='Legend'
        )
        
    elif "-Sigmoid_fit" in plot_to_do:
        
        Double_sigmoid_fit_dict = plot_list['3-Sigmoid_fit']
        
        # Extract data from dictionary
        original_data_table = Double_sigmoid_fit_dict["original_data_table"]
        full_sigmoid_fit_table = Double_sigmoid_fit_dict['full_sigmoid_fit_table']
        initial_sigmoid_fit_table = Double_sigmoid_fit_dict['initial_sigmoid_fit_table']
        color_shape_dict = Double_sigmoid_fit_dict['color_shape_dict']
        
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
        
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle',color='black'),
            name='Original Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open', color='grey'),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        fig.add_trace(go.Scatter(
            x=full_sigmoid_fit_table['Stim_amp_pA'],
            y=full_sigmoid_fit_table['Frequency_Hz'],
            mode='lines',
            line=dict(color='green'),
            name='Sigmoid fit'
        ))
        
        fig.add_trace(go.Scatter(
            x=initial_sigmoid_fit_table['Stim_amp_pA'],
            y=initial_sigmoid_fit_table['Frequency_Hz'],
            mode='lines',
            line=dict(color='red'),
            name='Initial conditions'
        ))
        
        
        # Update layout
        fig.update_layout(
            title='Double Sigmoid Fit',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz',
            legend_title='Legend'
        )

    elif '-Hill_fit_to_Sigmoid' in plot_to_do:
        Hill_fit_to_Sigmoid_dict = plot_list['4-Hill_fit_to_Sigmoid']
        
        # Extract data from the dictionary
        Asc_sig_to_fit = Hill_fit_to_Sigmoid_dict['Asc_sig_to_fit']
        initial_Hill_extended_fit_table = Hill_fit_to_Sigmoid_dict['initial_Hill_extended_fit_table']
        first_Hill_extended_fit_table = Hill_fit_to_Sigmoid_dict['first_Hill_extended_fit_table']
        color_shape_dict = Hill_fit_to_Sigmoid_dict['color_shape_dict']
        
        # Create the plotly figure
        fig = go.Figure()
        
        # Add the Asc_sig_to_fit line
        for legend in Asc_sig_to_fit['Legend'].unique():
            data = Asc_sig_to_fit[Asc_sig_to_fit['Legend'] == legend]
            fig.add_trace(go.Scatter(x=data['Stim_amp_pA'], y=data['Frequency_Hz'], mode='lines', name=legend, line=dict(color=color_shape_dict[legend])))
        
        # Add the initial_Hill_extended_fit_table line
        for legend in initial_Hill_extended_fit_table['Legend'].unique():
            data = initial_Hill_extended_fit_table[initial_Hill_extended_fit_table['Legend'] == legend]
            fig.add_trace(go.Scatter(x=data['Stim_amp_pA'], y=data['Frequency_Hz'], mode='lines', name=legend, line=dict(color=color_shape_dict[legend])))
        
        # Add the first_Hill_extended_fit_table line
        for legend in first_Hill_extended_fit_table['Legend'].unique():
            data = first_Hill_extended_fit_table[first_Hill_extended_fit_table['Legend'] == legend]
            fig.add_trace(go.Scatter(x=data['Stim_amp_pA'], y=data['Frequency_Hz'], mode='lines', name=legend, line=dict(color=color_shape_dict[legend])))
        
        # Update the layout
        fig.update_layout(
            title='Hill_fit_to_Ascending_sigmoid',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz'
        )

    elif '-Final_Hill_Sigmoid_Fit' in plot_to_do:
        
        Hill_Sigmoid_fit_dict = plot_list['5-Final_Hill_Sigmoid_Fit']
        # Extract data from the dictionary
        original_data_table = Hill_Sigmoid_fit_dict['original_data_table']
        Initial_Fit_table = Hill_Sigmoid_fit_dict['Initial_Fit_table']
        Final_fit_table = Hill_Sigmoid_fit_dict['Final_fit_table']
        color_shape_dict = Hill_Sigmoid_fit_dict['color_shape_dict']
        
        # Create the plotly figure
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
        
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle',color='black'),
            name='Original Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open', color='grey'),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        fig.add_trace(go.Scatter(x=Initial_Fit_table['Stim_amp_pA'], 
                                 y=Initial_Fit_table['Frequency_Hz'], 
                                 mode='lines', 
                                 name="Initial conditions", 
                                 line=dict(color=color_shape_dict["Initial conditions"])))
        
        fig.add_trace(go.Scatter(x=Final_fit_table['Stim_amp_pA'], 
                                 y=Final_fit_table['Frequency_Hz'], 
                                 mode='lines', 
                                 name="Final Fit", 
                                 line=dict(color=color_shape_dict["Final_Hill_Sigmoid_fit"])))
        
        
        # Update the layout
        fig.update_layout(
            title='Final_Hill_Sigmoid_Fit',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz',
            legend_title='Legend'
        )
        
    elif "-IO_fit" in plot_to_do:
        feature_plot_dict = plot_list['6-IO_fit']
        
        # Extract data from the dictionary
        original_data_table = feature_plot_dict['original_data_table']
        model_table = feature_plot_dict['model_table']
        Intercept = feature_plot_dict['intercept']
        Gain = feature_plot_dict['Gain']
        Threshold_table = feature_plot_dict['Threshold']
        
        # Create the plotly figure
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
        
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle',color='black'),
            name='Original Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open', color='grey'),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        # Add model line
        fig.add_trace(go.Scatter(x=model_table['Stim_amp_pA'], 
                                 y=model_table['Frequency_Hz'], 
                                 mode='lines', 
                                 name='IO Fit',
                                 line=dict(color='blue')))
        
        # Add the abline (slope and intercept)
        x_range = np.arange(original_data_table['Stim_amp_pA'].min(), original_data_table['Stim_amp_pA'].max(), 1)

        fig.add_trace(go.Scatter(x=x_range, 
                                 y=Intercept + Gain * x_range, 
                                 mode='lines', 
                                 name='Linear Fit', 
                                 line=dict(color='red', dash='dash')))
        
        # Add the Threshold points
        fig.add_trace(go.Scatter(x=Threshold_table["Stim_amp_pA"], 
                                 y=Threshold_table["Frequency_Hz"], 
                                 mode='markers', 
                                 name='Threshold', 
                                 marker=dict(color='green', symbol='x')))
        
        # Add Saturation points if not NaN
        if "Saturation" in feature_plot_dict.keys():
            Saturation_table = feature_plot_dict['Saturation']
            fig.add_trace(go.Scatter(x=Saturation_table['Stim_amp_pA'], 
                                     y=Saturation_table['Frequency_Hz'], 
                                     mode='markers', 
                                     name='Saturation', 
                                     marker=dict(color='green', symbol='triangle-up')))
        
        # Update the layout
        fig.update_layout(
            title='Feature Plot',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz',
            legend_title='Legend'
        )
                       
        
        # Show plot
    if return_plot == False:
        fig.show()
    else:
        return fig

        



def extract_inst_freq_table(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time):
    '''
    Compute the instananous frequency in each interspike interval per sweep for a cell

    Parameters
    ----------
    original_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    sweep_QC_table_inst : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        the sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit.
        
    response_time : float
        Response duration in s to consider
    

    Returns
    -------
    interval_freq_table: DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.

    '''
    
    maximum_nb_interval =0
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_QC_table=sweep_QC_table_inst.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])

        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes

            
            
    new_columns=["Interval_"+str(i) for i in range(1,(maximum_nb_interval))]

    

    SF_table = SF_table.reindex(SF_table.columns.tolist() + new_columns ,axis=1)

    for current_sweep in sweep_list:


        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        df=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)])
        spike_time_list=np.array(df.loc[:,'Time_s'])
        
        # Put a minimum number of spikes to compute adaptation
        if len(spike_time_list) >2:
            for current_spike_time_index in range(1,len(spike_time_list)):
                current_inst_frequency=1/(spike_time_list[current_spike_time_index]-spike_time_list[current_spike_time_index-1])

                SF_table.loc[current_sweep,str('Interval_'+str(current_spike_time_index))]=current_inst_frequency

                
            SF_table.loc[current_sweep,'Interval_1':]/=SF_table.loc[current_sweep,'Interval_1']

    interval_freq_table=pd.DataFrame(columns=['Spike_Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep'])
    isnull_table=SF_table.isnull()
    isnull_table.columns=SF_table.columns
    isnull_table.index=SF_table.index
    
    for interval,col in enumerate(new_columns):
        for line in sweep_list:
            if isnull_table.loc[line,col] == False:

                new_line=pd.DataFrame([int(interval)+1, # Interval#
                                    SF_table.loc[line,col], # Instantaneous frequency
                                    np.float64(cell_sweep_info_table.loc[line,'Stim_amp_pA']), # Stimulus amplitude
                                    line]).T# Sweep id
                                   
                new_line.columns=['Spike_Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep']
                interval_freq_table=pd.concat([interval_freq_table,new_line],ignore_index=True)
                
    
    interval_freq_table = pd.merge(interval_freq_table,sweep_QC_table,on='Sweep')
    return interval_freq_table

def get_min_freq_step(original_stimulus_frequency_table,do_plot=False):
    '''
    Estimate the initial frequency step, in order to decipher between Type I and TypeII neurons
    Fit the non-zero response, and use the fit value at the first stimulus amplitude eliciting a response.


    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        DataFrame containing one row per sweep, with the corresponding input current and firing frequency.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    minimum_freq_step : float
        Estimation of the initial frequency step
    
    minimum_freq_step_stim,
        Stimulus amplitude corresponding to the first frequency step
    
    noisy_spike_freq_threshold,
        Frequency value considered as 'noisy' at the beginning of the non-zero responses values 
    
    np.nanmax(original_data_table['Frequency_Hz'])
        Maximum frequency observed

    '''
    
    original_data_table =  original_stimulus_frequency_table.copy()
   
    original_data = original_data_table.copy()
    
    original_data = original_data.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    original_x_data = np.array(original_data['Stim_amp_pA'])
    original_y_data = np.array(original_data['Frequency_Hz'])
    extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
    
    first_stimulus_frequency_table=original_data_table.copy()
    first_stimulus_frequency_table=first_stimulus_frequency_table.reset_index(drop=True)
    first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    
    try:
        ### 0 - Trim Data
        response_threshold = (np.nanmax(first_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(first_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
    
        
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        
            
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_x_data=first_stimulus_frequency_table.loc[:,'Stim_amp_pA']
        trimmed_y_data=first_stimulus_frequency_table.loc[:,"Frequency_Hz"]
        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
       
    
        third_order_poly_model = PolynomialModel(degree = 3)
        pars = third_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
        
        if len(trimmed_y_data)>=4:
            
            third_order_poly_model_results = third_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
        
            best_c0 = third_order_poly_model_results.best_values['c0']
            best_c1 = third_order_poly_model_results.best_values['c1']
            best_c2 = third_order_poly_model_results.best_values['c2']
            best_c3 = third_order_poly_model_results.best_values['c3']
            
            
            
            
        
            extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
            
            extended_trimmed_3rd_poly_model_freq_diff=np.diff(extended_trimmed_3rd_poly_model)
            trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                                   "Frequency_Hz" : extended_trimmed_3rd_poly_model})
            trimmed_3rd_poly_table['Legend']='3rd_order_poly'
            my_plot=p9.ggplot(first_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))+p9.geom_point()
            my_plot+=p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))
            if do_plot:
                
                print(my_plot)
            
            zero_crossings_3rd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_3rd_poly_model_freq_diff)))[0] # detect last index before change of sign
           
          
            
            if len(zero_crossings_3rd_poly_model_freq_diff)==0:
                Descending_segment=False
                ascending_segment = trimmed_3rd_poly_table.copy()
                
            elif len(zero_crossings_3rd_poly_model_freq_diff)==1:
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    Descending_segment=True
                    
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=False
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] >= first_stim_root ]
                    
            elif len(zero_crossings_3rd_poly_model_freq_diff)==2:
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                second_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[1]]
                
                
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    Descending_segment=True
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=True
                    ascending_segment = trimmed_3rd_poly_table[(trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root) & (trimmed_3rd_poly_table['Stim_amp_pA'] <= second_stim_root) ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > second_stim_root]
        
        else:
            Descending_segment=False
                
                
                
                
                
            
       
            ### 1 - end
            

        
        ### 2 - Trim polynomial fit; keep [mean-0.5*poly_amplitude ; [mean+0.5*poly_amplitude]
        if Descending_segment:
            
            ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
            ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
            
            descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
            descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
            
            trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
            trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
            
            max_freq_poly_fit=np.nanmax(trimmed_3rd_poly_table['Frequency_Hz'])
            trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"
            trimmed_descending_segment.loc[trimmed_descending_segment.index,'Legend']="Trimmed_desc_seg"
           
        
        else:
            #end_slope_positive --> go for 2nd order polynomial
            second_order_poly_model = PolynomialModel(degree = 2)
            pars = second_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
            second_order_poly_model_results = second_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
    
            best_c0 = second_order_poly_model_results.best_values['c0']
            best_c1 = second_order_poly_model_results.best_values['c1']
            best_c2 = second_order_poly_model_results.best_values['c2']
            
            extended_trimmed_2nd_poly_model = best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
            extended_trimmed_2nd_poly_model_freq_diff=np.diff(extended_trimmed_2nd_poly_model)
            trimmed_2nd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                                   "Frequency_Hz" : extended_trimmed_2nd_poly_model})
            
           
            
            zero_crossings_2nd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_2nd_poly_model_freq_diff)))[0] # detect last index before change of sign
            
            if len(zero_crossings_2nd_poly_model_freq_diff) == 1:
    
                if extended_trimmed_2nd_poly_model_freq_diff[0]<0:
                    trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] >= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
                else:
                    trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] <= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
             
                
            ascending_segment = trimmed_2nd_poly_table.copy()
            ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
            ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
            
            trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
            max_freq_poly_fit=np.nanmax(trimmed_2nd_poly_table['Frequency_Hz'])
            
            
            trimmed_2nd_poly_table.loc[trimmed_2nd_poly_table.index,'Legend']='2nd_order_poly'
            trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"
            
        ### 2 - end 
        
        ### 3 - Linear fit on polynomial trimmed data
        ascending_linear_slope_init,ascending_linear_intercept_init=linear_fit(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        ascending_sigmoid_slope=max_freq_poly_fit/(4*ascending_linear_slope_init)
        
        if Descending_segment:

            descending_linear_slope_init,descending_linear_intercept_init=linear_fit(trimmed_descending_segment["Stim_amp_pA"],
                                                  trimmed_descending_segment['Frequency_Hz'])
        
        
        
        
        ### 3 - end
        
        ### 4 - Fit single or double Sigmoid
        
        ascending_sigmoid_fit= Model(sigmoid_function,prefix='asc_')
        ascending_segment_fit_params  = ascending_sigmoid_fit.make_params()
        
        
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*max_freq_poly_fit)
        
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        if Descending_segment:

            
            
           
            descending_sigmoid_slope = max_freq_poly_fit/(4*descending_linear_slope_init)
            
            descending_sigmoid_fit = Model(sigmoid_function,prefix='desc_')
            descending_sigmoid_fit_pars = descending_sigmoid_fit.make_params()
            descending_sigmoid_fit_pars.add("desc_x0",value=np.nanmean(trimmed_descending_segment['Stim_amp_pA']))
            descending_sigmoid_fit_pars.add("desc_sigma",value=descending_sigmoid_slope,min=3*descending_sigmoid_slope,max=-1e-9)
            
            ascending_sigmoid_fit *=descending_sigmoid_fit
            ascending_segment_fit_params+=descending_sigmoid_fit_pars

        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        

        full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data , best_value_x0_asc, best_value_sigma_asc)
        full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : full_sigmoid_fit})
        
       
        full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
        
        
        if Descending_segment:
            
            best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
            best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']
            
            full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data, best_value_x0_asc, best_value_sigma_asc) * sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
            full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                        "Frequency_Hz" : full_sigmoid_fit})
            full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
            
            
        
        if do_plot:
            
            
            double_sigmoid_comps = ascending_sigmoid_fit_results.eval_components(x=original_x_data)
            asc_sig_comp = double_sigmoid_comps['asc_']
            asc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : asc_sig_comp,
                                        'Legend' : 'Ascending_Sigmoid'})
            
            asc_sig_comp_table['Frequency_Hz'] = asc_sig_comp_table['Frequency_Hz']*max(original_data['Frequency_Hz'])/max(asc_sig_comp_table['Frequency_Hz'])

           
            
           
           
            
            sigmoid_fit_plot =  p9.ggplot(original_data,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point()
            sigmoid_fit_plot += p9.geom_line(full_sigmoid_fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
            #sigmoid_fit_plot += p9.geom_line(trimmed_ascending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
            #sigmoid_fit_plot += p9.geom_line(component_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'),linetype='dashed')
            
            sigmoid_fit_plot += p9.ggtitle("Sigmoid_fit_to original_data")
            
            # if Descending_segment:
                
                
            #     sigmoid_fit_plot += p9.geom_line(trimmed_descending_segment,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
                
            
           
            
            
    
            print(sigmoid_fit_plot)
         ### 4 - end

        maximum_fit_frequency=np.nanmax(full_sigmoid_fit)
        maximum_fit_frequency_index = np.nanargmax(full_sigmoid_fit)
        maximum_fit_stimulus = extended_x_data[maximum_fit_frequency_index]
        original_data_table = original_data_table.sort_values(by=['Stim_amp_pA'])
        
        noisy_spike_freq_threshold = np.nanmax([.04*maximum_fit_frequency,2.])
        until_maximum_data = original_data_table[original_data_table['Stim_amp_pA']<=maximum_fit_stimulus].copy()
       
        until_maximum_data = until_maximum_data[until_maximum_data['Frequency_Hz']>noisy_spike_freq_threshold]
        

        minimum_freq_step_index = np.nanargmin(until_maximum_data['Frequency_Hz'])
        minimum_freq_step = until_maximum_data.iloc[minimum_freq_step_index,2]
        minimum_freq_step_stim = until_maximum_data.iloc[minimum_freq_step_index,1]


    except Exception :

        x_data=np.array(original_data.loc[:,'Stim_amp_pA'])
        y_data=np.array(original_data.loc[:,'Frequency_Hz'])
        
        maximum_fit_frequency_index = np.nanargmax(y_data)
        maximum_fit_stimulus = x_data[maximum_fit_frequency_index]
        noisy_spike_freq_threshold = np.nanmax([.04*np.nanmax(y_data),2.])
        #noisy_spike_freq_threshold = 2.
        until_maximum_data = original_data_table[original_data_table['Stim_amp_pA']<=maximum_fit_stimulus].copy()
       
        until_maximum_data = until_maximum_data[until_maximum_data['Frequency_Hz']>noisy_spike_freq_threshold]
        
        if until_maximum_data.shape[0] == 0:
            without_zero_table = original_data_table[original_data_table['Frequency_Hz']!=0]

            minimum_freq_step_index = np.nanargmin(without_zero_table['Frequency_Hz'])
            minimum_freq_step = without_zero_table.iloc[minimum_freq_step_index,2]
            minimum_freq_step_stim = without_zero_table.iloc[minimum_freq_step_index,1]
        else:
            minimum_freq_step_index = np.nanargmin(until_maximum_data['Frequency_Hz'])
            minimum_freq_step = until_maximum_data.iloc[minimum_freq_step_index,2]
            minimum_freq_step_stim = until_maximum_data.iloc[minimum_freq_step_index,1]



        
    
    finally:

        if do_plot:
            minimum_freq_step_plot = p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))+p9.geom_point()
            minimum_freq_step_plot += p9.geom_point(until_maximum_data,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'),color='red')
            #minimum_freq_step_plot += p9.geom_line(full_sigmoid_fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
            minimum_freq_step_plot += p9.geom_vline(xintercept = minimum_freq_step_stim )
            minimum_freq_step_plot += p9.geom_hline(yintercept = minimum_freq_step)
            minimum_freq_step_plot += p9.geom_hline(yintercept = noisy_spike_freq_threshold,linetype='dashed',alpha=.4)
            
            print(minimum_freq_step_plot)
            
        return minimum_freq_step,minimum_freq_step_stim,noisy_spike_freq_threshold,np.nanmax(original_data_table['Frequency_Hz'])
    
def linear_fit(x, y):
    """
    Fit x-y to a line and return the slope of the fit.

    Parameters
    ----------
    x: array of values
    y: array of values
    Returns
    -------
    m: f-I curve slope for the specimen
    c:f-I curve intercept for the specimen

    """

   
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c


def sigmoid_function(x,x0,sigma):
    return (1-(1/(1+np.exp((x-x0)/sigma))))


def hill_function (x, x0, Hill_coef, Half_cst):
    
    y=np.empty(x.size)
    
    
    
    if len(max(np.where( x <= x0))) !=0:


        x0_index = max(np.where( x <= x0)[0])+1
        
    
        y[:x0_index]=0.

        y[x0_index:] =(((x[x0_index:]-x0)**(Hill_coef))/((Half_cst**Hill_coef)+((x[x0_index:]-x0)**(Hill_coef))))

    else:

        y = (((x-x0)**(Hill_coef))/((Half_cst**Hill_coef)+((x-x0)**(Hill_coef))))

    return y

def fit_adaptation_curve(interval_frequency_table_init,do_plot=False):
    '''
    Fit exponential curve to Spike-Interval - Instantaneous firing frequency
    

    Parameters
    ----------
    interval_frequency_table_init : pd.DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    obs : str
        Observation of the fitting procedure.
        
    best_A , best_B , best_C : Float 
        Fitting results.
        
    RMSE : Float
        Godness of fit.

    '''

    try:
        interval_frequency_table = interval_frequency_table_init.copy()
        interval_frequency_table=interval_frequency_table[interval_frequency_table['Passed_QC']==True]
        if interval_frequency_table.shape[0]==0:
            obs='Not_enough_spike'

            best_A=np.nan
            best_B=np.nan
            best_C=np.nan
            RMSE=np.nan
            return obs,best_A,best_B,best_C,RMSE
        interval_frequency_table.loc[:,'Spike_Interval']=interval_frequency_table.loc[:,'Spike_Interval'].astype(float)
        interval_frequency_table=interval_frequency_table.astype({"Spike_Interval":"float",
                                                                  "Normalized_Inst_frequency":"float",
                                                                  "Stimulus_amp_pA":'float'})
        x_data=interval_frequency_table.loc[:,'Spike_Interval']
        x_data=x_data.astype(float)
        y_data=interval_frequency_table.loc[:,'Normalized_Inst_frequency']
        y_data=y_data.astype(float)

        
        median_table=interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).median(numeric_only=True)
        median_table["Count_weights"]=pd.DataFrame(interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).count()).loc[:,"Sweep"] #count number of sweep containing a response in interval#
        median_table["Spike_Interval"]=median_table.index
        median_table["Spike_Interval"]=np.float64(median_table["Spike_Interval"])  
        
        y_delta=y_data.iloc[-1]-y_data.iloc[0]
       
        y_delta_two_third=y_data.iloc[0]-.66*y_delta
        
        initial_time_cst_guess_idx=np.argmin(abs(y_data - y_delta_two_third))
        
        initial_time_cst_guess=np.array(x_data)[initial_time_cst_guess_idx]
        
        
        
        
        
        initial_A=(y_data.iloc[0]-y_data.iloc[-1])/np.exp(-x_data.iloc[0]/initial_time_cst_guess)

        decayModel=Model(exponential_decay_function)

        decay_parameters=Parameters()

        decay_parameters.add("A",value=initial_A)
        decay_parameters.add('B',value=initial_time_cst_guess,min=0)
        decay_parameters.add('C',value=median_table["Normalized_Inst_frequency"][max(median_table["Spike_Interval"])])

        result = decayModel.fit(y_data, decay_parameters, x=x_data)

        best_A=result.best_values['A']
        best_B=result.best_values['B']
        best_C=result.best_values['C']


        pred=exponential_decay_function(np.array(x_data),best_A,best_B,best_C)
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE = np.sqrt(sum_squared_error / y_data.size)

        A_norm=best_A/(best_A+best_C)
        C_norm=best_C/(best_A+best_C)
        interval_range=np.arange(1,max(median_table["Spike_Interval"])+1,.1)

        simulation=exponential_decay_function(interval_range,best_A,best_B,best_C)
        norm_simulation=exponential_decay_function(interval_range,A_norm,best_B,C_norm)
        sim_table=pd.DataFrame(np.column_stack((interval_range,simulation)),columns=["Spike_Interval","Normalized_Inst_frequency"])
        norm_sim_table=pd.DataFrame(np.column_stack((interval_range,norm_simulation)),columns=["Spike_Interval","Normalized_Inst_frequency"])



        my_plot=np.nan
        if do_plot==True:

            my_plot=p9.ggplot(interval_frequency_table,p9.aes(x=interval_frequency_table["Spike_Interval"],y=interval_frequency_table["Normalized_Inst_frequency"]))+p9.geom_point(p9.aes(color=interval_frequency_table["Stimulus_amp_pA"]))

            my_plot=my_plot+p9.geom_point(median_table,p9.aes(x='Spike_Interval',y='Normalized_Inst_frequency',size=median_table["Count_weights"]),shape='s',color='red')
            my_plot=my_plot+p9.geom_line(sim_table,p9.aes(x='Spike_Interval',y='Normalized_Inst_frequency'),color='black')
            my_plot=my_plot+p9.geom_line(norm_sim_table,p9.aes(x='Spike_Interval',y='Normalized_Inst_frequency'),color="green")

            print(my_plot)


        obs='--'
        return obs,best_A,best_B,best_C,RMSE

    except (StopIteration):
        obs='Error_Iteration'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    except (ValueError):
        obs='Error_Value'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    except(RuntimeError):
        obs='Error_RunTime'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    
def normalized_root_mean_squared_error(true, pred,pred_extended):
    '''
    Compute the Root Mean Squared Error, normalized to the interquartile range

    Parameters
    ----------
    true : np.array
        Observed values.
    pred : np.array
        Values predicted by the fit.
    pred_extended : np.array
        Values predicted by the fit with higer number of points .

    Returns
    -------
    nrmse_loss : float
        Fit Error.

    '''
    #Normalization by the interquartile range
    squared_error = np.square((true - pred))
    sum_squared_error = np.sum(squared_error)
    rmse = np.sqrt(sum_squared_error / true.size)
    Q1=np.percentile(pred_extended,25)
    Q3=np.percentile(pred_extended,75)

    nrmse_loss = rmse/(Q3-Q1)
    return nrmse_loss

def exponential_decay_function(x,A,B,C):
    '''
    Parameters
    ----------
    x : Array
        interspike interval index array.
    A: flt
        initial instantanous frequency .
    B : flt
        Adaptation index constant.
    C : flt
        intantaneous frequency limit.

    Returns
    -------
    y : array
        Modelled instantanous frequency.

    '''
   
    
    return  A*np.exp(-(x)/B)+C


def fit_adaptation_test(interval_frequency_table_init,do_plot=False):
    '''
    Fit exponential curve to Spike-Interval - Instantaneous firing frequency
    

    Parameters
    ----------
    interval_frequency_table_init : pd.DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    obs : str
        Observation of the fitting procedure.
        
    best_A , best_B , best_C : Float 
        Fitting results.
        
    RMSE : Float
        Godness of fit.

    '''

    try:
        interval_frequency_table = interval_frequency_table_init.copy()
        interval_frequency_table=interval_frequency_table[interval_frequency_table['Passed_QC']==True]
        
        if interval_frequency_table.shape[0]==0:
            obs='Not_enough_spike'
            Adaptation_index = np.nan
            M = np.nan
            C = np.nan
            median_table = np.nan
            
            best_alpha=np.nan
            best_beta=np.nan
            best_gamma=np.nan
            RMSE=np.nan

            return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
        
        interval_frequency_table.loc[:,'Spike_Interval']=interval_frequency_table.loc[:,'Spike_Interval'].astype(float)
        
        interval_frequency_table=interval_frequency_table.astype({"Spike_Interval":"float",
                                                                  "Normalized_feature":"float",
                                                                  "Stimulus_amp_pA":'float'})
       
        median_table = get_median_feature_table_test(interval_frequency_table)

        x_data=median_table.loc[:,'Spike_Interval']
        x_data=x_data.astype(float)
        y_data=median_table.loc[:,'Normalized_feature']
        y_data=y_data.astype(float)
        weight_array = median_table.loc[:,'Count_weigths']
        weight_array = weight_array.astype(float)
    
        
        #Get initial condition 
        y_delta=y_data.iloc[-1]-y_data.iloc[0]
        y_delta_two_third=y_data.iloc[0]+.66*y_delta
        
        initial_time_cst_guess_idx=np.argmin(abs(y_data - y_delta_two_third))
        
        initial_time_cst_guess=np.array(x_data)[initial_time_cst_guess_idx]
        
        
        initial_alpha=(y_data.iloc[0]-y_data.iloc[-1])/np.exp(-x_data.iloc[0]/initial_time_cst_guess)
        
        decayModel=Model(exponential_decay_function)
        
        decay_parameters=Parameters()
    
        decay_parameters.add("A",value=initial_alpha)
        decay_parameters.add('B',value=initial_time_cst_guess,min=0, max=np.nanmax(x_data))
        decay_parameters.add('C',value=median_table["Normalized_feature"][max(median_table["Spike_Interval"])])
    
        result = decayModel.fit(y_data, decay_parameters, x=x_data)
    
        best_alpha=result.best_values['A']
        best_beta=result.best_values['B']
        best_gamma=result.best_values['C']

        
    
        y_data_pred=exponential_decay_function(np.array(x_data),best_alpha,best_beta,best_gamma)
        
        squared_error = np.square((y_data - y_data_pred))
        sum_squared_error = np.sum(squared_error)
        RMSE = np.sqrt(sum_squared_error / y_data.size)
        
        if np.abs(best_alpha/best_gamma) < 0.001: # consider the response as constant 
            
            y_data_pred_mean = np.nanmean(y_data_pred)
            average_list = [y_data_pred_mean]*len(y_data_pred)
            y_data_pred = np.array(average_list)
            
            best_alpha = 0
            best_gamma = 0
        
        #retain positive part of the fit, replace negative values by 0.
        positive_y_data_pred = []
        for i in y_data_pred:
            positive_y_data_pred.append(max(0.0,i))
        
        Na = int(np.nanmin([30,np.nanmax(x_data)]))


    
        C_ref = np.nanmin(positive_y_data_pred[:int(Na-1)])
        
        C = (Na-1)*C_ref

        modulated_y_data_pred = []
        for i in positive_y_data_pred:
            new_value = i - positive_y_data_pred[(Na-1)]
            modulated_y_data_pred.append(new_value)
        
        M = np.abs(np.nansum(modulated_y_data_pred))
        
        #we describe the adaptation index as the relative amount a certain characteritic changes during a spike train versus a constant component
        Adaptation_index = M/(C+M)
        obs='--'

        if do_plot:
            
            
            original_sim_table = pd.DataFrame(np.column_stack((x_data,y_data_pred)),columns=["Spike_Interval","Normalized_feature"])
            original_sim_table=original_sim_table.iloc[:int(Na-1),:]
            interval_range=np.arange(0,max(median_table["Spike_Interval"]),.1)
    
            simulation_extended=exponential_decay_function(interval_range,best_alpha,best_beta,best_gamma)
            
            sim_table=pd.DataFrame(np.column_stack((interval_range,simulation_extended)),columns=["Spike_Interval","Normalized_feature"])
            
            plot_dict = {"sim_table":sim_table, 
                         "original_sim_table" : original_sim_table,
                         "interval_frequency_table" : interval_frequency_table,
                         "median_table" : median_table,
                         "Na" : Na,
                         "C_ref" : C_ref,
                         'M' : M,
                         "C" : C}
            return plot_dict
        return obs, Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    
    except (StopIteration):
        obs='Error_Iteration'
        Adaptation_index = np.nan
        M = np.nan
        C = np.nan
        median_table = np.nan
        
        best_alpha=np.nan
        best_beta=np.nan
        best_gamma=np.nan
        RMSE=np.nan

        return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    except (ValueError):
        obs='Error_Value'
        Adaptation_index = np.nan
        M = np.nan
        C = np.nan
        median_table = np.nan
        
        best_alpha=np.nan
        best_beta=np.nan
        best_gamma=np.nan
        RMSE=np.nan
        return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    except(RuntimeError):
        obs='Error_RunTime'
        Adaptation_index = np.nan
        M = np.nan
        C = np.nan
        median_table = np.nan
        
        best_alpha=np.nan
        best_beta=np.nan
        best_gamma=np.nan
        RMSE=np.nan

        return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    
def plot_adaptation(plot_dict):
    import plotly.graph_objects as go
    original_sim_table = plot_dict["original_sim_table"]
    sim_table = plot_dict["sim_table"]
    Na = plot_dict["Na"]
    C_ref = plot_dict['C_ref']
    interval_frequency_table = plot_dict['interval_frequency_table']
    median_table = plot_dict['median_table']
    M = plot_dict['M']
    C = plot_dict['C']
    
    fig = go.Figure()

    # Line plot
    fig.add_trace(go.Scatter(x=original_sim_table['Spike_Interval'], y=original_sim_table['Normalized_feature'], mode='lines', name='Original Sim Table'))
    
    # Area plot
    fig.add_trace(go.Scatter(x=original_sim_table['Spike_Interval'], y=original_sim_table['Normalized_feature'], fill='tozeroy', fillcolor='#e5c8d6', line=dict(color='rgba(0,0,0,0)')))
    
    # Rect plot (as shape)
    fig.add_shape(type='rect', x0=np.nanmin(sim_table['Spike_Interval']), x1=(Na-1)-1, y0=0, y1=C_ref, fillcolor='gray', opacity=0.7, line=dict(width=0))
    
    # Line plot for sim_table
    fig.add_trace(go.Scatter(x=sim_table['Spike_Interval'], y=sim_table['Normalized_feature'], mode='lines', name='Sim Table'))
    
    # Points for interval_frequency_table with hovertext
    hover_text = [
        f'Sweep: {s}<br>Stim_amp_pA: {p}' 
        for s, p in zip(interval_frequency_table['Sweep'], interval_frequency_table['Stimulus_amp_pA'])
    ]
    
    fig.add_trace(go.Scatter(
        x=interval_frequency_table['Spike_Interval'], 
        y=interval_frequency_table['Normalized_feature'], 
        mode='markers', 
        marker=dict(color=interval_frequency_table['Stimulus_amp_pA'], colorscale='Viridis'),
        name='Interval Frequency Table',
        hovertext=hover_text,
        hoverinfo='text'
    ))
        # Normalize sizes
    max_size = 25
    sizes = (median_table['Count_weigths'] / median_table['Count_weigths'].max()) * max_size
    
   # Points for median_table with normalized sizes and hovertext for Count_weigths
    hover_text_median = [
        f'Count_weight: {w}' 
        for w in median_table['Count_weigths']
    ]
    
    fig.add_trace(go.Scatter(
        x=median_table['Spike_Interval'], 
        y=median_table['Normalized_feature'], 
        mode='markers', 
        marker=dict(size=sizes, color='red', symbol='square'), 
        name='Median Table',
        hovertext=hover_text_median,
        hoverinfo='text'
    ))
    
        # Invisible traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='#e5c8d6'),
        name=f'M: {M}'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='gray'),
        name=f'C: {C}'
    ))

    
    fig.update_layout(xaxis_title='Spike Interval', yaxis_title='Normalized Feature')
    
    fig.show()
                
def get_maximum_number_of_spikes_test(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time):
    maximum_nb_interval =0
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_QC_table=sweep_QC_table_inst.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])

        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes
            
    return maximum_nb_interval

def compute_cell_adaptation_behavior(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst):
    
    adaptation_dict = {'Instantaneous_Frequency':'Membrane_potential_mV',
                       "Spike_width_at_half_heigth" : "Time_s",
                       "Spike_heigth" : "Membrane_potential_mV", 
                       'Threshold' : "Membrane_potential_mV",
                       'Upstroke':"Potential_first_time_derivative_mV/s",
                       "Peak" : "Membrane_potential_mV",
                       "Downstroke" : "Potential_first_time_derivative_mV/s",
                       "Fast_Trough" : "Membrane_potential_mV",
                       "fAHP" : "Membrane_potential_mV",
                       "Trough" : "Membrane_potential_mV"}
    
    Adaptation_table = pd.DataFrame(columns = ["Obs", 'Feature', "Measure", "Adaptation_Index", 'M', 'C', 'alpha', 'beta', 'gamma', 'RMSE'])
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_QC_table=sweep_QC_table_inst.copy()

    
    for feature, measure in adaptation_dict.items():
        
        interval_based_feature = collect_interval_based_features_test(SF_table, cell_sweep_info_table, sweep_QC_table, 0.5, feature, measure)
        
        current_obs, current_Adaptation_index, current_M, current_C, current_median_table, current_best_alpha, current_best_beta, current_best_gamma, current_RMSE = fit_adaptation_test(interval_based_feature, False)
        
        new_line = pd.DataFrame([current_obs, feature, measure, current_Adaptation_index, current_M, current_C, current_best_alpha, current_best_beta, current_best_gamma, current_RMSE]).T
        new_line.columns = Adaptation_table.columns
        Adaptation_table = pd.concat([Adaptation_table, new_line], ignore_index = True)
    
    return Adaptation_table
        

    
    
def collect_interval_based_features_test(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time, feature, measure):
    '''
    Compute the instananous frequency in each interspike interval per sweep for a cell

    Parameters
    ----------
    original_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    sweep_QC_table_inst : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        the sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit.
        
    response_time : float
        Response duration in s to consider
    

    Returns
    -------
    interval_freq_table: DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.

    '''

    
   
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_QC_table=sweep_QC_table_inst.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
   

    maximum_number_of_spikes = get_maximum_number_of_spikes_test(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time)
    
    if feature == "Instantaneous_Frequency":
        #start interval indexing at 0 --> first interval between spike 0 and spike 1
        new_columns=["Interval_"+str(i) for i in range((maximum_number_of_spikes))]
        
    else:
        # start spike indexing at 0
        new_columns=["Spike_"+str(i) for i in range((maximum_number_of_spikes))]
    
    

    SF_table = SF_table.reindex(SF_table.columns.tolist() + new_columns ,axis=1)

    for current_sweep in sweep_list:


        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        df=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)])

        spike_time_list=np.array(df.loc[:,'Time_s'])
        spike_index_in_time_range = np.array(df.loc[:,'Spike_index'].unique())
        
        # Put a minimum number of spikes to compute adaptation
        if feature == "Instantaneous_Frequency":
            if len(spike_time_list) >2: # strictly greater than 2 so that we have at least 2 intervals
                for current_spike_time_index in range(1,len(spike_time_list)): # start at 0 so the first substraction is index 1 - index 0...
                    current_inst_frequency=1/(spike_time_list[current_spike_time_index]-spike_time_list[current_spike_time_index-1])
    
                    SF_table.loc[current_sweep,str('Interval_'+str(int(current_spike_time_index-1)))]=current_inst_frequency
    
                    
                SF_table.loc[current_sweep,'Interval_0':]/=SF_table.loc[current_sweep,'Interval_0']
        else:
            if len(spike_time_list) >2:
                sub_SF = pd.DataFrame(SF_table.loc[current_sweep,'SF'])

                sub_SF = sub_SF.loc[sub_SF['Spike_index'].isin(spike_index_in_time_range),:]

                feature_df = sub_SF.loc[sub_SF['Feature']==feature,:]
                feature_df = feature_df.sort_values(by=['Time_s'])
                feature_list = np.array(feature_df.loc[:,measure])
                
                
                for current_spike_index in range(len(feature_list)): # start at 0 so the first substraction is index 1 - index 0...
                    if measure == "Membrane_potential_mV":
                        current_feature_normalized = feature_list[0]-feature_list[current_spike_index]
                    else:
                        current_feature_normalized = feature_list[current_spike_index]/feature_list[0]
                        
                    
                    SF_table.loc[current_sweep,str('Spike_'+str(int(current_spike_index)))]=current_feature_normalized
    
                if measure == "Membrane_potential_mV":
                    for col in new_columns:
                        SF_table.loc[current_sweep,col]=SF_table.loc[current_sweep,'Spike_0']-SF_table.loc[current_sweep,col]
                else:
                    
                    SF_table.loc[current_sweep,'Spike_0':]/=SF_table.loc[current_sweep,'Spike_0']
                

    interval_freq_table=pd.DataFrame(columns=['Spike_Interval','Normalized_feature','Stimulus_amp_pA','Sweep'])
    isnull_table=SF_table.isnull()
    isnull_table.columns=SF_table.columns
    isnull_table.index=SF_table.index
    
    for interval,col in enumerate(new_columns):
        for line in sweep_list:
            if isnull_table.loc[line,col] == False:

                new_line=pd.DataFrame([int(interval), # Interval#
                                    SF_table.loc[line,col], # Instantaneous frequency
                                    np.float64(cell_sweep_info_table.loc[line,'Stim_amp_pA']), # Stimulus amplitude
                                    line]).T# Sweep id
                                   
                new_line.columns=['Spike_Interval','Normalized_feature','Stimulus_amp_pA','Sweep']
                interval_freq_table=pd.concat([interval_freq_table,new_line],ignore_index=True)
                
    
    interval_freq_table = pd.merge(interval_freq_table,sweep_QC_table,on='Sweep')
    return interval_freq_table

def get_median_feature_table_test(interval_freq_table_init):
    interval_frequency_table = interval_freq_table_init.copy()
    interval_frequency_table=interval_frequency_table[interval_frequency_table['Passed_QC']==True]
    
    interval_frequency_table=interval_frequency_table.astype({"Spike_Interval":"float",
                                                              "Normalized_feature":"float",
                                                              "Stimulus_amp_pA":'float'})
    
    median_table=interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).median(numeric_only=True)
    median_table["Count_weigths"]=pd.DataFrame(interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).count()).loc[:,"Sweep"] #count number of sweep containing a response in interval#
    median_table["Spike_Interval"]=median_table.index
    median_table["Spike_Interval"]=np.float64(median_table["Spike_Interval"])  
    
    return median_table
    


     

