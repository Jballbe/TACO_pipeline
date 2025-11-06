#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:35:26 2023

@author: julienballbe
"""

import scipy
import pandas as pd
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache

def get_traces_Lantyer(original_cell_file_folder, cell_id, sweep_list, cell_sweep_table):
    
    # Initialize return lists
    time_trace_list = []
    potential_trace_list = []
    input_current_trace_list = []
    
    
    #Get file name
    file_name = cell_sweep_table.loc[cell_sweep_table['Cell_id'] == cell_id,
                                     'Original_file'].unique()[0]

    #open File
    current_file = scipy.io.loadmat(str(
        str(original_cell_file_folder)+str(file_name)))
    current_unique_cell_id = cell_id[-1]
    
    #For each sweep, access the corresponding traces, and append them to the return lists
    for sweep_id in sweep_list:
        
        #Get sweep from file 
        #Parse the sweep_id to get the experiment number, protocol id and sweep id as used in the database's files
        experiment,current_Protocol,current_sweep = sweep_id.split("_")
        #recreate the ids used in the original matlab files
        current_id = str('Trace_'+current_unique_cell_id+'_' +
                         current_Protocol+'_'+str(current_sweep))
        
        #Get traces 
        input_current_stim_trace = pd.DataFrame(
            current_file[str(current_id+'_1')], columns=['Time_s', 'Input_current_pA'])
        input_current_stim_trace.loc[:, 'Input_current_pA'] *= 1e12 #To pA
    
        current_membrane_trace = pd.DataFrame(
            current_file[str(current_id+'_2')], columns=['Time_s', 'Membrane_potential_mV'])
        current_membrane_trace.loc[:, 'Membrane_potential_mV'] *= 1e3 #to mV
    
        time_trace = np.array(current_membrane_trace.loc[:, 'Time_s'])
        potential_trace = np.array(
            current_membrane_trace.loc[:, 'Membrane_potential_mV'])
        
        input_current_trace = np.array(input_current_stim_trace.loc[:, 'Input_current_pA'])
        
        time_trace_list.append(time_trace)
        potential_trace_list.append(potential_trace)
        input_current_trace_list.append(input_current_trace)
        
    return time_trace_list , potential_trace_list, input_current_trace_list

def get_traces_Allen_CTD(original_cell_file_folder, cell_id, sweep_list, cell_sweep_table):
    
    # Initialize return lists 
    time_trace_list = []
    potential_trace_list = []
    current_trace_list = []
    stim_start_time_list = []
    stim_end_time_list= []
    
    # Follow the AllenSDK tutorials to access files throught CellTypesCache using the manifest.json file
    ctc = CellTypesCache(
        manifest_file=str(str(original_cell_file_folder)+"manifest.json"))
    cell_id = int(cell_id)
    my_Cell_data = ctc.get_ephys_data(cell_id)
    
    #For each sweep, access the corresponding traces, and stimulus start and end times, and append them to the return lists
    for sweep_id in sweep_list:
    
        index_range = my_Cell_data.get_sweep(int(sweep_id))["index_range"]
        sampling_rate = my_Cell_data.get_sweep(int(sweep_id))["sampling_rate"]
        current_trace = (my_Cell_data.get_sweep(int(sweep_id))["stimulus"][0:index_range[1]+1]) * 1e12  # to pA
        potential_trace = (my_Cell_data.get_sweep(int(sweep_id))["response"][0:index_range[1]+1]) * 1e3  # to mV
        potential_trace = np.array(potential_trace)
        current_trace = np.array(current_trace)
        stim_start_index = index_range[0]+next( x for x, val in enumerate(current_trace[index_range[0]:]) if val != 0)
        time_trace = np.arange(0, len(current_trace)) * (1.0 / sampling_rate)
        stim_start_time = time_trace[stim_start_index]
        stim_end_time = stim_start_time+1.
        
        time_trace_list.append(time_trace)
        potential_trace_list.append(potential_trace)
        current_trace_list.append(current_trace)
        stim_start_time_list.append(stim_start_time)
        stim_end_time_list.append(stim_end_time)
    
    return time_trace_list, potential_trace_list, current_trace_list, stim_start_time_list, stim_end_time_list


def get_traces_NVC(original_cell_file_folder, cell_id, sweep_list, cell_sweep_table):
    # Initialize return lists 
    time_trace_list = []
    potential_trace_list = []
    current_trace_list = []
    stim_start_time_list = []
    stim_end_time_list= []
    
    for sweep_id in sweep_list:

        #To access the NVC databases files, multiple information are required to follow the database's organization
        #the sweep_id contains the experiment type, the protocol id and the trace id
        # these information are used to refer to the cell sweep table to access the cell's animal species, date, and cell folder's name
        experiment,current_Protocol,current_trace_id = sweep_id.split("_")
        current_trace_df = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)& (cell_sweep_table['Sweep_id']==sweep_id),:]
        current_trace_df=current_trace_df.reset_index()
        species = current_trace_df.loc[0,"Species"]
        date = current_trace_df.loc[0,"Date"]
        cell = current_trace_df.loc[0,"Cell/Electrode"]
    
        # Altogether, these information enable to build the path toward the cell's folder and the appropriate protocol file
        Full_cell_file=original_cell_file_folder+species+'/'+date+'/'+cell+'/G clamp/'+current_Protocol+'.'+experiment
        
        # The file is then parsed to extract the header (protocol information), then the traces
        float_array=np.fromfile(Full_cell_file,dtype='>f')
        separator_indices=np.argwhere(float_array==max(float_array))   
        header=float_array[:separator_indices[0][0]]
    
        header_information={"Iterations":header[0],
                  "I_start_pulse_pA":header[1],
                  "I_increment_pA":header[2],
                  "I_pulse_width_ms":header[3],
                  "I_delay_ms":header[4],
                  "Sweep_duration_ms":header[5],
                  "Sample_rate_Hz":header[6],
                  "Cycle_time_ms":header[7],
                  "Bridge_MOhms":header[8],
                  "Cap_compensation":header[9]}
        
    
    
        
        current_Protocol = str(current_Protocol)
        current_trace_id=int(current_trace_id)
        
        # According to the trace_id get the appropriate indexes to get the raw traces
        if current_trace_id==len(separator_indices):
            previous_index=separator_indices[int(current_trace_id)-1][0]
            raw_traces=float_array[previous_index:]
        else:
            previous_index=separator_indices[int(current_trace_id)-1][0]
            index=separator_indices[current_trace_id][0]
            raw_traces=float_array[previous_index:index]
        

        traces=raw_traces[1:]
        potential_current_sep_index=int(len(traces)/2)
        potential_trace=traces[:potential_current_sep_index]
    
        if current_trace_df.shape[0]>1:
            raise ValueError(f'Cell {cell_id} has more than one row for sweep {sweep_id} in cell_sweep_csv')
        else:
            #In some case the recorded input curren trace is not correct, thus it is indicated in the cell_sweep table
            # and the appropriate information are provided in the cell sweep table to construct a synthetic input curren trace
            if current_trace_df.loc[0,"Synthesize Current"] == True:
                stim_start_time = float(current_trace_df.loc[0,"Stimulus Start (msec)"])
                stim_start_time*=1e-3
                
                stim_end_time = float(current_trace_df.loc[0,"Stimulus End (msec)"])
                stim_end_time*=1e-3
                
                stim_amp = float(current_trace_df.loc[0,"Stimulus Amplitude (pA)"])
                sampling_rate = int(current_trace_df.loc[0,"Sample Rate (Hz)"])
                
                sweep_duration = len(potential_trace)/sampling_rate
                time_trace=np.arange(0,sweep_duration,1/sampling_rate)
                
                current_trace=np.zeros(len(potential_trace))
                
                stim_start_index=np.argmin(abs(time_trace-stim_start_time))
                stim_end_index = np.argmin(abs(time_trace- stim_end_time))
                current_trace[stim_start_index:stim_end_index]+=stim_amp
                
            #Otherwise the input current trace is obtained from the file
            else:
                current_trace=traces[potential_current_sep_index:]
                
                sweep_duration=len(current_trace)/header_information["Sample_rate_Hz"]
                
                time_trace=np.arange(0,sweep_duration,(1/header_information["Sample_rate_Hz"]))
                
                stim_start_time=header_information["I_delay_ms"]*1e-3
                stim_end_time=stim_start_time+header_information["I_pulse_width_ms"]*1e-3
                
        time_trace_list.append(time_trace)
        potential_trace_list.append(potential_trace)
        current_trace_list.append(current_trace)
        stim_start_time_list.append(stim_start_time)
        stim_end_time_list.append(stim_end_time)
            
    return time_trace_list, potential_trace_list, current_trace_list, stim_start_time_list, stim_end_time_list
            
            
    