#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:26:28 2024

@author: julienballbe
"""
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache

def get_traces_Allen_CTD(original_cell_file_folder,cell_id,sweep_id,cell_sweep_table):
    #Import Cell Type Manifest
    ctc = CellTypesCache(
        manifest_file=str(str(original_cell_file_folder)+"manifest.json"))
    #import Cell file
    cell_id = int(cell_id)
    my_Cell_data = ctc.get_ephys_data(cell_id)
    #Get sweep information
    index_range = my_Cell_data.get_sweep(int(sweep_id))["index_range"]
    sampling_rate = my_Cell_data.get_sweep(int(sweep_id))["sampling_rate"]
    current_trace = (my_Cell_data.get_sweep(int(sweep_id))["stimulus"][0:index_range[1]+1]) * 1e12  # to pA
    potential_trace = (my_Cell_data.get_sweep(int(sweep_id))["response"][0:index_range[1]+1]) * 1e3  # to mV
    potential_trace = np.array(potential_trace)
    current_trace = np.array(current_trace)
    stim_start_index = index_range[0]+next( x for x, 
                                           val in enumerate(current_trace[index_range[0]:]) 
                                           if val != 0)
    time_trace = np.arange(0, len(current_trace)) * (1.0 / sampling_rate)
    stim_start_time = time_trace[stim_start_index]
    stim_end_time = stim_start_time+1.
    
    return time_trace, potential_trace, current_trace, stim_start_time, stim_end_time
  