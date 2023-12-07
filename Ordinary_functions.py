#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:36:59 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
import plotnine as p9
import scipy
import h5py
import importlib

import Sweep_analysis as sw_an
import Spike_analysis as sp_an

def get_filtered_TVC_table(original_cell_full_TVC_table,sweep,do_filter=True,filter=5.,do_plot=False):
    '''
    From the cell Full TVC table, get the sweep related TVC table, and if required, with filtered Potential and Current values
    

    Parameters
    ----------
    original_cell_full_TVC_table : pd.DataFrame
        2 columns DataFrame, containing in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time, Current and Potential Traces
    sweep : str
        Sweep id.
    do_filter : Bool, optional
        Wether or not to Filter Membrane voltage and Current traces. The default is True.
    do_plot : Bool, optional
       The default is False.

    Returns
    -------
    TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.

    '''
    
    cell_full_TVC_table = original_cell_full_TVC_table.copy()
    TVC_table=cell_full_TVC_table.loc[str(sweep),'TVC'].copy()
    
    if do_filter:
        
        TVC_table['Membrane_potential_mV']=np.array(filter_trace(TVC_table['Membrane_potential_mV'],
                                                                            TVC_table['Time_s'],
                                                                            filter=filter,
                                                                            do_plot=do_plot))
        
        TVC_table['Input_current_pA']=np.array(filter_trace(TVC_table['Input_current_pA'],
                                                                            TVC_table['Time_s'],
                                                                            filter=filter,
                                                                            do_plot=do_plot))
    
    
    
    first_derivative=get_derivative(np.array(TVC_table['Membrane_potential_mV']),np.array(TVC_table['Time_s']))
    first_derivative=np.insert(first_derivative,0,np.nan)

    second_derivative=get_derivative(first_derivative,np.array(TVC_table['Time_s']))
    second_derivative=np.insert(second_derivative,0,[np.nan,np.nan])
    
    TVC_table['Potential_first_time_derivative_mV/s'] = first_derivative
    TVC_table["Potential_second_time_derivative_mV/s/s"] = second_derivative
    TVC_table=TVC_table.astype({'Time_s':float,
                               'Membrane_potential_mV':float,
                               'Input_current_pA':float,
                               'Potential_first_time_derivative_mV/s':float,
                               'Potential_second_time_derivative_mV/s/s':float}) 
    
    
    if do_plot:
        voltage_plot = p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Membrane_potential_mV'))+p9.geom_line()
        voltage_plot+=p9.ggtitle(str('Sweep:'+str(sweep)+'Membrane_Potential_mV'))
        print(voltage_plot)
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Input_current_pA'))+p9.geom_line()
        current_plot+=p9.ggtitle(str('Sweep:'+str(sweep)+'Input_current_pA'))
        print(current_plot)


    return TVC_table

def filter_trace(value_trace, time_trace, filter=5., filter_order = 2, do_plot=False): 
    '''
    Apply a Butterworth Low-pass filters time-varying signal.

    Parameters
    ----------
    value_trace : np.array
        array of time_varying signal to filter.
        
    time_trace : np.array
        Array of time in second.
        
    filter : Float, optional
        Cut-off frequency in kHz . The default is 5.
        
    filter_order : int, optional
        Order of the filter to apply. The default is 2.
        
    do_plot : Boolean, optional
        Do Plot. The default is False.

    Raises
    ------
    ValueError
        Raise error if the sampling frequency of the time varying signal is lower thant the Nyquist frequency.

    Returns
    -------
    filtered_signal : np.array
        Value trace filtered.

    '''


    delta_t = time_trace[1] - time_trace[0]
    sample_freq = 1. / delta_t

    filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency

    if filt_coeff < 0 or filt_coeff >= 1:
        raise ValueError("Butterworth coeff ({:f}) is outside of valid range [0,1]; cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter))
    b, a = scipy.signal.butter(filter_order, filt_coeff, "low")

    zi = scipy.signal.lfilter_zi(b, a)
    
    filtered_signal =  scipy.signal.lfilter(b, a, value_trace,zi=zi*value_trace[0], axis=0)[0]
    
    
   
    
    if do_plot:
        signal_df = pd.DataFrame({'Time_s':np.array(time_trace),
                                  'Values':np.array(value_trace)})
        filtered_df = pd.DataFrame({'Time_s':np.array(time_trace),
                                  'Values':np.array(filtered_signal)})
        
        signal_df ['Trace']='Original_Trace'
        filtered_df ['Trace']='Filtered_Trace'
        
        signal_df=signal_df.append(filtered_df,ignore_index=True)
        
        filter_plot = p9.ggplot(signal_df, p9.aes(x='Time_s',y='Values',color='Trace',group='Trace'))+p9.geom_line()
        print(filter_plot)
        
    return filtered_signal

def get_derivative(value_trace, time_trace):
    '''
    Get time derivative of a signal trace

    Parameters
    ----------
    value_trace : np.array
        Array of time_varying signal.
        
    time_trace : np.array
        Array of time in second.

    Returns
    -------
    dvdt : np.array
        Time derivative of the time varying signal trace.

    '''
    
    dv = np.diff(value_trace)

    dt = np.diff(time_trace)
    dvdt = 1e-3 * dv / dt # in V/s = mV/ms

    # Remove nan values (in case any dt values == 0)
    dvdt = dvdt[~np.isnan(dvdt)]

    return dvdt

def write_cell_file_h5(cell_file_path,
                       saving_dict,
                       overwrite=False,
                       selection=['All']):
    '''
    Create a hdf5 file for a given cell file path (or overwrite existing one if required) and store results of the analysis.

    Parameters
    ----------
    cell_file_path : str
        File path to which store cell file.
        
    original_Full_SF_dict : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep') contains Sweep_id, and 
        second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table)
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    original_cell_sweep_QC : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        
    original_cell_fit_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the fit parameters to reconstruct I/O curve and adaptation curve.
        
    original_cell_feature_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the Adaptation and I/O features.
        
    original_Metadata_table : pd.DataFrame
        DataFrame containing cell metadata.
    
    original_processing_table : pd.DataFrame
        DataFrame containing for each part of the analysis the processing time 
        as well as one row per Error or Warning encountered during the analysis
        
    overwrite : Bool, optional
        Wether to overwrite already existing information, if for a given cell_id, a cell file already exists in saving_folder_path
        The default is False.



    '''


    f = h5py.File(cell_file_path, "a")
    
    if "All" in selection:
        selection = ["Metadata","Sweep analysis", "Spike analysis", "Firing analysis"]
    
    if "Metadata" in saving_dict.keys():
        original_Metadata_table = saving_dict["Metadata"]
        
        if 'Metadata' in f.keys() and overwrite == True:
            del f["Metadata"]
            if isinstance(original_Metadata_table,dict) == True:
                Metadata_group = f.create_group('Metadata')
                for elt in original_Metadata_table.keys():
        
                    Metadata_group.create_dataset(
                        str(elt), data=original_Metadata_table[elt])
        elif 'Metadata' not in f.keys():
            if isinstance(original_Metadata_table,dict) == True:
                Metadata_group = f.create_group('Metadata')
                for elt in original_Metadata_table.keys():
        
                    Metadata_group.create_dataset(
                        str(elt), data=original_Metadata_table[elt])
                
            
    if "Spike analysis" in saving_dict.keys():
        original_Full_SF_dict = saving_dict["Spike analysis"]
        
        if 'Spike analysis' in f.keys() and overwrite == True:
            del f["Spike analysis"]
            if isinstance(original_Full_SF_dict,pd.DataFrame) == True:
                SF_group = f.create_group("Spike analysis")
                
                sweep_list = np.array(original_Full_SF_dict['Sweep'])
                for current_sweep in sweep_list:
                    
                    current_SF_dict = original_Full_SF_dict.loc[current_sweep, "SF_dict"]
                    current_SF_group = SF_group.create_group(str(current_sweep))
                    for elt in current_SF_dict.keys():
        
                        if len(current_SF_dict[elt]) != 0:
                            current_SF_group.create_dataset(
                                str(elt), data=current_SF_dict[elt])
    
        elif 'Spike analysis' not in f.keys():
            if isinstance(original_Full_SF_dict,pd.DataFrame) == True:
                sweep_list = np.array(original_Full_SF_dict['Sweep'])
                SF_group = f.create_group("Spike analysis")
                
                for current_sweep in sweep_list:
        
                    current_SF_dict = original_Full_SF_dict.loc[current_sweep, "SF_dict"]
                    current_SF_group = SF_group.create_group(str(current_sweep))
                    for elt in current_SF_dict.keys():
        
                        if len(current_SF_dict[elt]) != 0:
                            current_SF_group.create_dataset(
                                str(elt), data=current_SF_dict[elt])
               
    if "Sweep analysis" in saving_dict.keys():
        sweep_analysis_dict = saving_dict["Sweep analysis"]
        original_cell_sweep_info_table = sweep_analysis_dict['Sweep info']
        original_cell_sweep_QC = sweep_analysis_dict['Sweep QC']
        
        if 'Sweep analysis' in f.keys() and overwrite == True:
            
            del f["Sweep analysis"]
            if isinstance(original_cell_sweep_info_table,pd.DataFrame) == True:
                cell_sweep_info_table_group = f.create_group('Sweep analysis')
                sweep_list=np.array(original_cell_sweep_info_table['Sweep'])
        
                for elt in np.array(original_cell_sweep_info_table.columns):
                    cell_sweep_info_table_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_info_table[elt]))
                   
    
        elif 'Sweep analysis' not in f.keys():
            if isinstance(original_cell_sweep_info_table,pd.DataFrame) == True:
                cell_sweep_info_table_group = f.create_group('Sweep analysis')
                for elt in np.array(original_cell_sweep_info_table.columns):
                    cell_sweep_info_table_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_info_table[elt]))
                    
            
        
        if isinstance(original_cell_sweep_QC,pd.DataFrame) == True:
            convert_dict = {'Sweep': str,
                            'Passed_QC' : bool
                            }
        
            original_cell_sweep_QC = original_cell_sweep_QC.astype(convert_dict)
    
        if 'Sweep_QC' in f.keys() and overwrite == True:
            del f['Sweep_QC']
            if isinstance(original_cell_sweep_QC,pd.DataFrame) == True:
                cell_sweep_QC_group = f.create_group("Sweep_QC")
                for elt in np.array(original_cell_sweep_QC.columns):
                    
                    cell_sweep_QC_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_QC[elt]))
        elif 'Sweep_QC' not in f.keys():
            if isinstance(original_cell_sweep_QC,pd.DataFrame) == True:
                cell_sweep_QC_group = f.create_group("Sweep_QC")
                for elt in np.array(original_cell_sweep_QC.columns):
                    
                    cell_sweep_QC_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_QC[elt]))
            
    
    if 'Firing analysis' in saving_dict.keys():
        Firing_analysis_dict = saving_dict["Firing analysis"]
        original_cell_feature_table = Firing_analysis_dict['Cell_feature']
        original_cell_fit_table = Firing_analysis_dict['Cell_fit']
        if 'Cell_Feature' in f.keys() and overwrite == True:
            del f["Cell_Feature"]
            if isinstance(original_cell_feature_table,pd.DataFrame) == True:
                cell_feature_group = f.create_group('Cell_Feature')
                for elt in np.array(original_cell_feature_table.columns):
                    cell_feature_group.create_dataset(
                        elt, data=np.array(original_cell_feature_table[elt]))
            
            del f['Cell_Fit']
            if isinstance(original_cell_fit_table,pd.DataFrame) == True:
                cell_fit_group = f.create_group('Cell_Fit')
                for elt in np.array(original_cell_fit_table.columns):
                    cell_fit_group.create_dataset(
                        elt, data=np.array(original_cell_fit_table[elt]))
    
            
    
        elif 'Cell_Feature' not in f.keys():
            if isinstance(original_cell_feature_table,pd.DataFrame) == True:
                cell_feature_group = f.create_group('Cell_Feature')
                for elt in np.array(original_cell_feature_table.columns):
                    cell_feature_group.create_dataset(
                        elt, data=np.array(original_cell_feature_table[elt]))
           
    
            # Store Cell fit table
            if isinstance(original_cell_fit_table,pd.DataFrame) == True:
                cell_fit_group = f.create_group('Cell_Fit')
                
        
                for elt in np.array(original_cell_fit_table.columns):
                    
                    cell_fit_group.create_dataset(
                        elt, data=np.array(original_cell_fit_table[elt]))
                
    #store processing report
    original_processing_table = saving_dict["Processing report"]
    if 'Processing_report' in f.keys() and overwrite == True:
        del f["Processing_report"]
        if isinstance(original_processing_table,pd.DataFrame) == True:
            processing_report_group = f.create_group('Processing_report')
            for elt in np.array(original_processing_table.columns):
                processing_report_group.create_dataset(
                    elt, data=np.array(original_processing_table[elt]))
            
    elif 'Processing_report' not in f.keys():
        if isinstance(original_processing_table,pd.DataFrame) == True:
            processing_report_group = f.create_group('Processing_report')
            for elt in np.array(original_processing_table.columns):
                processing_report_group.create_dataset(
                    elt, data=np.array(original_processing_table[elt]))
           
                
    

    f.close()

def open_json_config_file(config_file):
    '''
    Open JSON configuration file and return a DataFrame

    Parameters
    ----------
    config_file : str
        Path to JSON configuration file (ending .json).

    Returns
    -------
    config_df : pd.DataFrame
        DataFrame containing the information of the JSON configuration file.

    '''
    config_json = pd.read_json(config_file)
    colnames = config_json.keys().to_list()[:-1]
    colnames = colnames+pd.DataFrame(config_json.loc[0,"DB_parameters"],index=[0]).columns.to_list()
    config_df = pd.DataFrame(columns=colnames)
    for db in range(config_json.shape[0]):
        new_line = pd.DataFrame(config_json.loc[db,"DB_parameters"],index=[0])
        
        new_line["path_to_saving_file"] = config_json['path_to_saving_file']
        
        config_df=pd.concat([config_df,new_line],axis=0,ignore_index=True)
    return config_df

def create_TVC(time_trace,voltage_trace,current_trace):
    '''
    Create table containing Time, Voltage, Current traces

    Parameters
    ----------
    time_trace : np.array
        Time points array in s.
        
    voltage_trace : np.array
        Array of membrane voltage recording in mV.
        
    current_trace : np.array
        Array of input current in pA.

    Raises
    ------
    ValueError
        All array must be of the same length, if not raises ValueError.

    Returns
    -------
    TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.

    '''
    length=len(time_trace)
    
    if length!=len(voltage_trace) or length!=len(current_trace):

        raise ValueError('All lists must be of equal length. Current lengths: Time {}, Potential {}, Current {}'.format(len(time_trace),len(voltage_trace),len(current_trace)))
    
    
    TVC_table=pd.DataFrame({'Time_s':time_trace,
                               'Membrane_potential_mV':voltage_trace,
                               'Input_current_pA':current_trace},
                           dtype=np.float64) 
    return TVC_table

def estimate_trace_stim_limits(TVC_table_original,stimulus_duration,do_plot=False):
    '''
    Estimate for a given trace the start time and end of the stimulus, by performing autocorrelation

    Parameters
    ----------
    TVC_table_original : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    stimulus_duration : float
        Duration of the stimulus in second.
        
    do_plot : Bool, optional
        Do plot. The default is False.

    Returns
    -------
    best_stim_start : float
        Start time of the stimulus in second.
        
    best_stim_end : float
        End time of the stimulus in second.

    '''
    
    TVC_table=TVC_table_original.copy()
    

    current_derivative = get_derivative(np.array(TVC_table['Input_current_pA']),
                                        np.array(TVC_table['Time_s']))
    current_derivative=np.insert(current_derivative,0,np.nan)
    TVC_table["Filtered_Stimulus_trace_derivative_pA/ms"]=np.array(current_derivative)
    # remove last 50ms of signal (potential step)
    limit = TVC_table.shape[0] - \
        int(0.05/(TVC_table.iloc[1, 0]-TVC_table.iloc[0, 0]))

    TVC_table.loc[limit:,
                   "Filtered_Stimulus_trace_derivative_pA/ms"] = np.nan
    
    TVC_table = get_autocorrelation(TVC_table, stimulus_duration, do_plot=do_plot)
    
    best_stim_start = TVC_table[TVC_table['Autocorrelation'] == np.nanmin(
        TVC_table['Autocorrelation'])].iloc[0, 0]

    best_stim_end = best_stim_start+stimulus_duration
    
    return best_stim_start, best_stim_end

def get_autocorrelation(table, time_shift, do_plot=False):
    '''
    Compute autocorrelation at each time point for a stimulus trace table

    Parameters
    ----------
    table : DataFrame
        Stimulus trace table, 3 columns
        First column = 'Time_s'
        Second column = 'Input_current_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
    time_shift : float
        Time shift in s .
    do_plot : Bool, optional
        The default is False.

    Returns
    -------
    table : DataFrame
        Stimulus trace table, 6 columns
        First column = 'Time_s'
        Second column = 'Input_current_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
        4th column = "Filtered_Stimulus_trace_derivative_pA/ms"
        5th column = "Shifted_trace" --> Filtered stiumulus trace derivatove shifted by 'time_shift'
        6th column = 'Autocorrelation --> Autocorrelation between 4th and 5th column
    '''

    shift = int(time_shift/(table.iloc[1, 0]-table.iloc[0, 0]))

    table["Shifted_trace"] = table['Filtered_Stimulus_trace_derivative_pA/ms'].shift(
        -shift)

    table['Autocorrelation'] = table['Filtered_Stimulus_trace_derivative_pA/ms'] * \
        table["Shifted_trace"]

    if do_plot == True:

        myplot = p9.ggplot(table, p9.aes(x=table.loc[:, 'Time_s'], y=table.loc[:, 'Filtered_Stimulus_trace_derivative_pA/ms']))+p9.geom_line(
            color='blue')+p9.geom_line(table, p9.aes(x=table.loc[:, 'Time_s'], y=table.loc[:, 'Autocorrelation']), color='red')

        myplot += p9.xlab(str("Time_s; Time_shift="+str(time_shift)))
        print(myplot)

    return table

def find_time_index(t, t_0):
    """ 
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Find the index value of a given time (t_0) in a time series (t).


    Parameters
    ----------
    t   : time array
    t_0 : time point to find an index

    Returns
    -------
    idx: index of t closest to t_0
    """
    assert t[0] <= t_0 <= t[-1], "Given time ({:f}) is outside of time range ({:f}, {:f})".format(t_0, t[0], t[-1])

    idx = np.argmin(abs(t - t_0))
    return idx

def read_cell_file_h5(cell_file, config_line, selection=['All']):
    '''
    Open cell h5 and returns the different elements

    Parameters
    ----------
    cell_file : str
        Path to the cell HDF5 file (ending .h5).
        
    config_json_file : str
        Path to JSON configuration file (ending .json).
        
        
    selection : List, optional
        Indicates which elements from the files to return (Elements must correspond to h5 groups). The default is ['All'].

    Returns
    -------
    Full_TVC_table : pd.DataFrame
        2 columns DataFrame, containing in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time, Current and Potential Traces.
        
    Full_SF_dict_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep') contains Sweep_id, and 
        second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table)
    
    Full_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    Metadata_table : pd.DataFrame
        DataFrame containing cell metadata.
        
    sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    Sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit
        
    cell_fit_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the fit parameters to reconstruct I/O curve and adaptation curve.
        
    cell_feature_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the Adaptation and I/O features.
        
    Processing_report_df : pd.DataFrame
        DataFrame containing for each part of the analysis the processing time 
        as well as one row per Error or Warning encountered during the analysis

    '''
    
    current_file = h5py.File(cell_file, 'r')

    Full_TVC_table = pd.DataFrame()
    Full_SF_dict_table = pd.DataFrame()
    Full_SF_table = pd.DataFrame()
    Metadata_table = pd.DataFrame()
    sweep_info_table = pd.DataFrame()
    Sweep_QC_table = pd.DataFrame()
    cell_fit_table = pd.DataFrame()
    cell_feature_table = pd.DataFrame()
    Processing_report_df = pd.DataFrame()
    
    if 'All' in selection:
        selection = ['TVC_SF', 'Sweep analysis','Sweep QC', 'Metadata', 'Firing analysis','Processing_report']
    
    sub1 = "Cell_"
    sub2 = ".h5"

    # getting index of substrings
    idx1 = cell_file.index(sub1)
    idx2 = cell_file.index(sub2)
 
    # length of substring 1 is added to
    # get string from next character
    cell_id = cell_file[idx1 + len(sub1) : idx2]
    
    if 'Metadata' in selection and 'Metadata' in current_file.keys():
        ## Metadata ##

        Metadata_group = current_file['Metadata']
        Metadata_dict = {}
        for data in Metadata_group.keys():
            if type(Metadata_group[data][()]) == bytes:
                Metadata_dict[data] = Metadata_group[data][()].decode('ascii')
            else:
                Metadata_dict[data] = Metadata_group[data][()]
        Metadata_table = pd.DataFrame(Metadata_dict, index=[0])
    
    elif 'Metadata' in selection and 'Metadata' not in current_file.keys():
        print('File does not contains Metadata group')

    if 'TVC_SF' in selection and 'Spike analysis' in current_file.keys():
        
        SF_group = current_file['Spike analysis']

        sweep_list = list(SF_group.keys())
        
        
        print(config_line)
        if isinstance(config_line,pd.DataFrame) == True:
            config_line = config_line.reset_index()
            config_line = config_line.to_dict('index')
            config_line = config_line[0]
        print(config_line)    
        path_to_python_folder = config_line['path_to_db_script_folder']
        python_file = config_line['python_file_name']
        module=python_file.replace('.py',"")
        full_path_to_python_script=str(path_to_python_folder+python_file)
        print(module)
        print(full_path_to_python_script)
        spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)
        print(spec)
        DB_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DB_module)
        
        
        
        
        db_original_file_directory = config_line['original_file_directory']
        

        
        db_cell_sweep_file = pd.read_csv(config_line['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
        
        
        Full_SF_dict_table = pd.DataFrame(columns=['Sweep', 'SF_dict'])
        
        args_list = [[module,
                      full_path_to_python_script,
                      config_line["db_function_name"],
                      db_original_file_directory,
                      cell_id,
                      x,
                      db_cell_sweep_file,
                      config_line["stimulus_time_provided"],
                      config_line["db_stimulus_duration"]] for x in sweep_list]
        
        Full_TVC_table = pd.DataFrame(columns=['Sweep','TVC'])
        
        for x in args_list:
            result = sw_an.get_TVC_table(x)
            Full_TVC_table = pd.concat([
                Full_TVC_table,result[0]], ignore_index=True)
            
        
        
            
        for current_sweep in sweep_list:

            current_SF_table = SF_group[str(current_sweep)]
            SF_dict = {}
    
            for feature in current_SF_table.keys():
                SF_dict[feature] = np.array(current_SF_table[feature])
    
            new_line_SF = pd.DataFrame([str(current_sweep), SF_dict]).T
            new_line_SF.columns=['Sweep', 'SF_dict']
            Full_SF_dict_table = pd.concat([Full_SF_dict_table,
                        new_line_SF], ignore_index=True)
           
        Full_TVC_table.index = Full_TVC_table.loc[:,"Sweep"]
        Full_TVC_table.index = Full_TVC_table.index.astype(str)
        
        Full_SF_dict_table.index = Full_SF_dict_table["Sweep"]
        Full_SF_dict_table.index = Full_SF_dict_table.index.astype(str)
        
        Sweep_analysis_group = current_file['Sweep analysis']
        Sweep_analysis_dict = {}
        for data in Sweep_analysis_group.keys():
            Sweep_analysis_dict[data] = Sweep_analysis_group[data][()]
        Sweep_analysis_dict['Sweep']=Sweep_analysis_dict['Sweep'].astype(str)
        sweep_info_table = pd.DataFrame(
            Sweep_analysis_dict, index=Sweep_analysis_dict['Sweep'])

        Full_SF_table = sp_an.create_Full_SF_table(
            Full_TVC_table, Full_SF_dict_table.copy(), sweep_info_table.copy())
        
        

    if 'Sweep analysis' in selection and 'Sweep analysis' in current_file.keys():

        ## Sweep_analysis_table ##

        Sweep_analysis_group = current_file['Sweep analysis']
        Sweep_analysis_dict = {}
        for data in Sweep_analysis_group.keys():
            Sweep_analysis_dict[data] = Sweep_analysis_group[data][()]
        Sweep_analysis_dict['Sweep']=Sweep_analysis_dict['Sweep'].astype(str)
        sweep_info_table = pd.DataFrame(
            Sweep_analysis_dict, index=Sweep_analysis_dict['Sweep'])
        
        if 'Sweep_QC' in current_file.keys():
            Sweep_QC_group = current_file['Sweep_QC']
            Sweep_QC_dict = {}
            for data in Sweep_QC_group.keys():
                Sweep_QC_dict[data] = Sweep_QC_group[data][()]
            Sweep_QC_dict['Sweep']=Sweep_QC_dict['Sweep'].astype(str)
            Sweep_QC_table = pd.DataFrame(
                Sweep_QC_dict, index=Sweep_QC_dict['Sweep'])
            for col in Sweep_QC_table.columns:
                if col != 'Sweep':
                    Sweep_QC_table=Sweep_QC_table.astype({str(col):"bool"})
        else:
            print('File does not contains Sweep_QC group')

    elif 'Sweep analysis' in selection and 'Sweep analysis' not in current_file.keys():
        print('File does not contains Sweep analysis group')
        
    

    if 'Firing analysis' in selection and 'Cell_Fit' in current_file.keys():
        ## Cell_fit ##

        Cell_fit_group = current_file['Cell_Fit']
        Cell_fit_dict = {}
        for data in Cell_fit_group.keys():
            if type(Cell_fit_group[data][(0)]) == bytes:
                Cell_fit_dict[data] = np.array(
                    [x.decode('ascii') for x in Cell_fit_group[data][()]], dtype='str')
            else:
                Cell_fit_dict[data] = Cell_fit_group[data][()]

        cell_fit_table = pd.DataFrame(
            Cell_fit_dict)
        cell_fit_table.index = cell_fit_table.index.astype(int)

        ## Cell_feature ##

        Cell_feature_group = current_file['Cell_Feature']
        cell_feature_dict={}
        for data in Cell_feature_group.keys():
            if type(Cell_feature_group[data][(0)]) == bytes:
                cell_feature_dict[data] = np.array(
                    [x.decode('ascii') for x in Cell_feature_group[data][()]], dtype='str')
            else:
                cell_feature_dict[data] = Cell_feature_group[data][()]

        cell_feature_table = pd.DataFrame(
            cell_feature_dict)
        cell_feature_table.index = cell_feature_table.index.astype(int)

        
        
    elif 'Firing analysis' in selection and 'Cell_Fit' not in current_file.keys():
        print('File does not contains Cell_Fit group')
        
        
    if 'Processing_report' in selection and 'Processing_report' in current_file.keys():
        Processing_report_group = current_file['Processing_report']
        Processing_report_dict={} 
        for data in Processing_report_group.keys():
            if type(Processing_report_group[data][(0)]) == bytes:
                Processing_report_dict[data] = np.array(
                    [x.decode('ascii') for x in Processing_report_group[data][()]], dtype='str')
        Processing_report_df = pd.DataFrame(Processing_report_dict)

    current_file.close()
    return Full_TVC_table, Full_SF_dict_table, Full_SF_table, Metadata_table, sweep_info_table, Sweep_QC_table, cell_fit_table, cell_feature_table,Processing_report_df

