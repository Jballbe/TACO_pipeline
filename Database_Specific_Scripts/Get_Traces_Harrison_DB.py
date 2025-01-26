#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:37:48 2024

@author: julienballbe
"""

import numpy as np



def get_trace_Harisson(original_cell_file_folder,cell_id,sweep_list,cell_sweep_table):
    
    time_trace_list = []
    potential_trace_list = []
    current_trace_list = []
    stim_start_time_list = []
    stim_end_time_list= []
    
    current_step_values = np.fromfile(f'{original_cell_file_folder}Istep_{cell_id}.bin',dtype='<f')
    potential_traces = np.fromfile(f'{original_cell_file_folder}Vstep_{cell_id}.bin',dtype='<f')
    
    potential_trace_split = int(len(potential_traces)/len(current_step_values))

    stim_start_time = float(cell_sweep_table.loc[cell_sweep_table['Cell_id']==cell_id,'Stimulus_start_time_s'].values[0])
    stim_end_time = float(cell_sweep_table.loc[cell_sweep_table['Cell_id']==cell_id,'Stimulus_end_time_s'].values[0])
    
    for sweep_id in sweep_list:
    
        CC, protocol, trace_nb = sweep_id.split("_")
        first_index = int(trace_nb)*potential_trace_split
        last_index = (int(trace_nb)+1)*potential_trace_split
        potential_trace =  potential_traces[first_index:last_index]


    
        sampling_frequency = 20000 #from Harisson et al 
        sweep_duration = len(potential_trace)/sampling_frequency
        time_trace=np.arange(0,sweep_duration,(1/sampling_frequency))
    
        stim_amp = float(current_step_values[int(trace_nb)])
    
        current_trace = np.zeros(len(time_trace))
        stim_start_index=np.argmin(abs(time_trace-stim_start_time))
        stim_end_index = np.argmin(abs(time_trace- stim_end_time))
        current_trace[stim_start_index:stim_end_index]+=stim_amp
        
        time_trace_list.append(time_trace)
        potential_trace_list.append(potential_trace)
        current_trace_list.append(current_trace)
        stim_start_time_list.append(stim_start_time)
        stim_end_time_list.append(stim_end_time)
    
    return time_trace_list, potential_trace_list, current_trace_list, stim_start_time_list, stim_end_time_list
        
import os
def get_trace_Scala_2019(original_cell_file_folder,cell_id,sweep_list,cell_sweep_table):
    
    
    time_trace_list = []
    potential_trace_list = []
    current_trace_list = []
    stim_start_time_list = []
    stim_end_time_list= []
    
    cell_file_name = cell_sweep_table.loc[cell_sweep_table['Cell_id']==cell_id,"file_name"].values[0]
    cell_patch = cell_sweep_table.loc[cell_sweep_table['Cell_id']==cell_id,"Patch"].values[0]
    cell_folder = cell_sweep_table.loc[cell_sweep_table['Cell_id']==cell_id,"Folder"].values[0]
    
    full_cell_file = os.path.join(original_cell_file_folder, cell_patch, cell_folder, cell_file_name)
    current_file = scipy.io.loadmat(full_cell_file)
    
    for sweep_id in sweep_list:
    
    
        cell_trace = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==sweep_id),"Trace"].values[0]
    
    
        stim_start_time = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==sweep_id),"Stim_start_s"].values[0]
        stim_end_time = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==sweep_id),"Stim_end_s"].values[0]
    
    
    
        current_time_potential_table = pd.DataFrame(current_file[cell_trace], columns = ['Time_s', 'Membrane_potential_mV'])
    
        current_time_potential_table.loc[:,'Membrane_potential_mV']*=1e3
    
        cell_holding_current = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==sweep_id),"Holding_current_pA"].values[0]
        cell_stim_amp = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==sweep_id),"Stim_amp_pA"].values[0]
        
        time_trace = np.array(current_time_potential_table.loc[:,'Time_s'])
        potential_trace = np.array(current_time_potential_table.loc[:,'Membrane_potential_mV'])
        
        current_trace = np.zeros(len(time_trace))
        current_trace = current_trace + cell_holding_current
        stim_start_index=np.argmin(abs(time_trace-stim_start_time))
        stim_end_index = np.argmin(abs(time_trace- stim_end_time))
    
        current_trace[stim_start_index:stim_end_index] = cell_stim_amp # Simulus amplitude is absolute (not relative to holding current)
        
        time_trace_list.append(time_trace)
        potential_trace_list.append(potential_trace)
        current_trace_list.append(current_trace)
        stim_start_time_list.append(stim_start_time)
        stim_end_time_list.append(stim_end_time)
    
    return time_trace_list, potential_trace_list, current_trace_list, stim_start_time_list, stim_end_time_list
    
    
    
    
        
        
import scipy
import plotnine as p9
import pandas as pd
def plot_cell_traces(cell_file, cell_id):
    
    current_file = scipy.io.loadmat(cell_file)
    variance_table = dict()
    my_plot = p9.ggplot()
    for key, table in current_file.items():
        if "Trace" not in key:
            continue
        current_table = pd.DataFrame(table, columns=['Time_s', "Membrane_potetnial_mV"])
        my_plot += p9.geom_line(current_table,p9.aes(x="Time_s",y="Membrane_potetnial_mV"))
        my_plot+= p9.ggtitle(cell_id)
        # my_plot.show()
        
    #variance_table = pd.DataFrame(variance_table, columns=['Trace',"Variance"])
    
    return variance_table


def get_cell_recording_channel(cell_file,cell_id, do_plot = False):
    
    current_file = scipy.io.loadmat(cell_file)
    cell_table = pd.DataFrame(columns=['Trace', 'Experiment', "Protocol", 'Sweep','Channel', "Variance"])
    my_plot = p9.ggplot()
    for key, table in current_file.items():
        if "Trace" not in key:
            continue
        Trace, Experiment, Protocol, Sweep, Channel = key.split("_")
        current_table = pd.DataFrame(table, columns=['Time_s', "Membrane_potential_mV"])
        # my_plot = p9.ggplot(current_table,p9.aes(x="Time_s",y="Membrane_potetnial_mV"))+p9.geom_line()
        # my_plot+= p9.ggtitle(key)
        # my_plot.show()
        variance = np.var(np.array(current_table.loc[:,'Membrane_potential_mV']))
        
        new_line = pd.DataFrame([key, Experiment, Protocol, Sweep, Channel, variance]).T
        new_line.columns = cell_table.columns
        cell_table = pd.concat([cell_table, new_line], ignore_index = True)
   
    cell_table = cell_table.astype({"Variance":"float"})
    mean_variance_per_channel = cell_table.groupby('Channel')['Variance'].mean()
    mean_variance_per_channel_second = mean_variance_per_channel.reset_index()
    # Find the index of the maximum value in the second column
    max_index = mean_variance_per_channel_second['Variance'].idxmax()

    # Get the corresponding value from the first column
    Recording_channel = mean_variance_per_channel_second.loc[max_index, 'Channel']

    sub_cell_table = cell_table.loc[cell_table['Channel']==Recording_channel,:]
    sub_cell_table = sub_cell_table.reset_index(drop=True)
    for line in sub_cell_table.index:
        Protocol = sub_cell_table.loc[line, 'Protocol']
        Sweep = sub_cell_table.loc[line, 'Sweep']
        
        new_sweep = f'CC_{Protocol}_{Sweep}'
        sub_cell_table.loc[line, "Sweep_id"] = new_sweep
    
    min_index = sub_cell_table['Variance'].idxmin()
    min_variance_sweep = int(sub_cell_table.loc[min_index, 'Sweep'])

    # Initialize the input_current_pA column
    sub_cell_table['Stim_amp_pA'] = 0
    sub_cell_table['Holding_current_pA'] = 0
    # Assign input currents
    for i, row in sub_cell_table.iterrows():
        diff = int(row['Sweep']) - min_variance_sweep
        sub_cell_table.loc[i, 'Stim_amp_pA'] = diff * 20 
        
    if do_plot:
        for key, table in current_file.items():
            if "Trace" not in key:
                continue
            Trace, Experiment, Protocol, Sweep, Channel = key.split("_")
            if Channel != Recording_channel :
                continue
            current_table = pd.DataFrame(table, columns=['Time_s', "Membrane_potential_mV"])
            
            my_plot += p9.geom_line(current_table,p9.aes(x="Time_s",y="Membrane_potential_mV"))
            my_plot+= p9.ggtitle(cell_id)
            
        my_plot+=p9.geom_vline(p9.aes(xintercept = .1),color = "blue")
        my_plot+=p9.geom_vline(p9.aes(xintercept = .7),color = "red")
        my_plot.show()
                
                
    return cell_table, Recording_channel, sub_cell_table

import os
import tqdm
def get_Scala_patch_seq_cell_Sweep_table(directory):
    cell_sweep_table = pd.DataFrame()
    
    
    cell_population_class_table = pd.DataFrame(columns=['Cell_id', 'General_area', 'Sub_area', 'Layer',
           'Dendrite_type', 'Hemisphere', 'cell_reporter_status', 'line_name',
           'Database', 'Experiment', 'Animal', 'Exc_Inh', 'Exc_Inh_FT',
           'Cell_type', 'Layer_second'])
    
    pre_folder_list = os.listdir(directory)
    folder_list = [x for x in pre_folder_list if "." not in x]
    
    
    
    #Get file_list in folder
    for folder in folder_list:
        pre_file_list = os.listdir(os.path.join(directory, folder))
        file_list = [x for x in pre_file_list if '._' not in x]
        
        for file_name in tqdm.tqdm(file_list,desc=f'Folder { folder}'):
            year, day, month, _, slice_id, _, sample, _ = file_name.split(" ",7)
            
            cell_id=f'{year}_{day}_{month}_{slice_id}_{sample}'
            
            #get recording_channel
            cell_table, Recording_channel, current_cell_sweep_table = get_cell_recording_channel(os.path.join(directory, folder,file_name),cell_id,False)
            
            current_cell_sweep_table.loc[:,'Patch']="patch-seq"
            current_cell_sweep_table.loc[:,'Folder']=folder
            current_cell_sweep_table.loc[:,'file_name']=file_name
            current_cell_sweep_table.loc[:,'Cell_id']=cell_id
            current_cell_sweep_table.loc[:,"Stim_start_s"] = .1
            current_cell_sweep_table.loc[:,"Stim_end_s"] = .7
        
            
            cell_sweep_table = pd.concat([cell_sweep_table, current_cell_sweep_table],ignore_index = True)
            
            if 'V1' in folder:
                General_area = 'VIS'
            else:
                General_area = 'SS'
            Sub_area = "Unknown"
            if '5' in folder:
                Layer ="5"
                Layer_second = '5'
            else:
                Layer = "4"
                Layer_second = '4'
            Dendrite_type = "Unknown"
            Hemisphere = "Unknown"
            cell_reporter_Status = "Unknown"
            line_name = "Unknown"
            Database = "Scala_2019_DB"
            Experiment = "In_Vitro"
            Animal = "Mouse"
            Exc_Inh = "Inhibitory"
            Exc_Inh_FT = "--"
            cell_type = "Sst"
            population_class_table_new_line = pd.DataFrame([cell_id, General_area,Sub_area, Layer, Dendrite_type, Hemisphere, cell_reporter_Status,  line_name, Database ,Experiment , Animal, Exc_Inh, Exc_Inh_FT, cell_type, Layer_second]).T
            
            population_class_table_new_line.columns = cell_population_class_table.columns
            cell_population_class_table = pd.concat([cell_population_class_table, population_class_table_new_line], ignore_index = True)

    return cell_population_class_table,cell_sweep_table

def get_Scala_patch_morph_cell_Sweep_table(directory):
    cell_sweep_table = pd.DataFrame()
    
    stimulation_current_table = pd.read_excel("/Volumes/Work_Julien/Scala_2019_data/patch-morph/stimulation-currents.xlsx")
    cell_population_class_table = pd.DataFrame(columns=['Cell_id', 'General_area', 'Sub_area', 'Layer',
           'Dendrite_type', 'Hemisphere', 'cell_reporter_status', 'line_name',
           'Database', 'Experiment', 'Animal', 'Exc_Inh', 'Exc_Inh_FT',
           'Cell_type', 'Layer_second'])
    
    pre_folder_list = os.listdir(directory)
    folder_list = [x for x in pre_folder_list if "." not in x]
    
    #Get file_list in folder
    for folder in folder_list:
        pre_file_list = os.listdir(os.path.join(directory, folder))
        file_list = [x for x in pre_file_list if '._' not in x]
        
        for file_name in tqdm.tqdm(file_list,desc=f'Folder { folder}'):
            file_name = file_name.replace('.mat','')
            sub_cell_id = file_name
            month, day, year, _, slice_id, _, sample = file_name.split(" ")

            cell_id=f'{year}_{day}_{month}_{slice_id}_{sample}'
            
            #get recording_channel
            cell_table, Recording_channel, current_cell_sweep_table = get_cell_recording_channel_morph(os.path.join(directory, folder,file_name),cell_id, sub_cell_id,stimulation_current_table,False)
            current_cell_sweep_table.loc[:,'Patch']="patch-morph"
            current_cell_sweep_table.loc[:,'Folder']=folder
            current_cell_sweep_table.loc[:,'file_name']=file_name
            current_cell_sweep_table.loc[:,'Cell_id']=cell_id
            current_cell_sweep_table.loc[:,"Stim_start_s"] = .1
            current_cell_sweep_table.loc[:,"Stim_end_s"] = .7
            
            if folder == "NMC":
                General_area = "SS"
            else:
                General_area = "VIS"
            Sub_area = "Unknown"
            Layer ="4"
            Dendrite_type = "Unknown"
            Hemisphere = "Unknown"
            cell_reporter_Status = "Unknown"
            line_name = "Unknown"
            Database = "Scala_2019_DB"
            Experiment = "In_Vitro"
            Animal = "Mouse"
            if folder == "PYR":
                Exc_Inh = "Excitatory"
            else:
                Exc_Inh = "Inhibitory"
            
            Exc_Inh_FT = "--"
            
            if folder in ['LBC', "SBC", "DBC" , "HBC"]:
                cell_type = "PValb"
            elif folder in ['MC', "NMC"]:
                cell_type ="Sst"
            elif folder == "BPC":
                cell_type ="VIP"
            elif folder == "NGC":
                cell_type = "--"
            elif folder == "PYR":
                cell_type = "Excitatory"
                
            layer_second = "4"
                
            
            
            population_class_table_new_line = pd.DataFrame([cell_id, General_area,Sub_area, Layer, Dendrite_type, Hemisphere, cell_reporter_Status,  line_name, Database ,Experiment , Animal, Exc_Inh, Exc_Inh_FT, cell_type, layer_second]).T
            
            population_class_table_new_line.columns = cell_population_class_table.columns
            cell_population_class_table = pd.concat([cell_population_class_table, population_class_table_new_line], ignore_index = True)
            cell_sweep_table = pd.concat([cell_sweep_table,current_cell_sweep_table ],ignore_index = True)
    return cell_population_class_table, cell_sweep_table

def get_cell_recording_channel_morph(cell_file,cell_id,sub_cell_id,stim_current_table, do_plot = False):
    
    
    first_stim_amp = stim_current_table.loc[stim_current_table['Cell ID'] == sub_cell_id, "inj current"].values[0]
    holding_current = stim_current_table.loc[stim_current_table['Cell ID'] == sub_cell_id, "Current start (pA)"].values[0]
    
    current_file = scipy.io.loadmat(cell_file)
    cell_table = pd.DataFrame(columns=['Trace', 'Experiment', "Protocol", 'Sweep','Channel', "Variance"])
    my_plot = p9.ggplot()
    for key, table in current_file.items():
        if "Trace" not in key:
            continue
        Trace, Experiment, Protocol, Sweep, Channel = key.split("_")
        current_table = pd.DataFrame(table, columns=['Time_s', "Membrane_potential_mV"])
        
        variance = np.var(np.array(current_table.loc[:,'Membrane_potential_mV']))
        
        new_line = pd.DataFrame([key, Experiment, Protocol, Sweep, Channel, variance]).T
        new_line.columns = cell_table.columns
        cell_table = pd.concat([cell_table, new_line], ignore_index = True)
   
    cell_table = cell_table.astype({"Variance":"float"})
    mean_variance_per_channel = cell_table.groupby('Channel')['Variance'].mean()
    mean_variance_per_channel_second = mean_variance_per_channel.reset_index()
    # Find the index of the maximum value in the second column
    max_index = mean_variance_per_channel_second['Variance'].idxmax()

    # Get the corresponding value from the first column
    Recording_channel = mean_variance_per_channel_second.loc[max_index, 'Channel']

    sub_cell_table = cell_table.loc[cell_table['Channel']==Recording_channel,:]
    sub_cell_table = sub_cell_table.reset_index(drop=True)
    for line in sub_cell_table.index:
        Protocol = sub_cell_table.loc[line, 'Protocol']
        Sweep = sub_cell_table.loc[line, 'Sweep']
        
        new_sweep = f'CC_{Protocol}_{Sweep}'
        sub_cell_table.loc[line, "Sweep_id"] = new_sweep
    sub_cell_table = sub_cell_table.astype({'Protocol':'int','Sweep':'int'})
    sub_cell_table = sub_cell_table.sort_values(by=['Protocol', "Sweep"])
    sub_cell_table = sub_cell_table.reset_index(drop=True)
    
    
    for elt in sub_cell_table.index:
        sub_cell_table.loc[elt, "Holding_current_pA"] = holding_current
        sub_cell_table.loc[elt, "Stim_amp_pA"] = int(elt)*20+first_stim_amp#+holding_current #the stimulus amplitude is relative to the holding current, increemented with 20pA
    
    
        
    if do_plot:
        for key, table in current_file.items():
            if "Trace" not in key:
                continue
            Trace, Experiment, Protocol, Sweep, Channel = key.split("_")
            if Channel != Recording_channel :
                continue
            current_table = pd.DataFrame(table, columns=['Time_s', "Membrane_potential_mV"])
            
            my_plot += p9.geom_line(current_table,p9.aes(x="Time_s",y="Membrane_potential_mV"))
            my_plot+= p9.ggtitle(cell_id)
            
        my_plot+=p9.geom_vline(p9.aes(xintercept = .1),color = "blue")
        my_plot+=p9.geom_vline(p9.aes(xintercept = .7),color = "red")
        my_plot.show()
                
                
    return cell_table, Recording_channel, sub_cell_table


def update_Scala_2019_cell_Sweep_table(Scala_2019_cell_table, Scala_2019_pop_name, current_and_label_table):
    
    
    for index in current_and_label_table.index:
        if current_and_label_table.loc[index, "add info"] == 0:
            file_name = current_and_label_table.loc[index, "Cell ID"]
            Scala_2019_cell_table = Scala_2019_cell_table[Scala_2019_cell_table['file_name']!=file_name]
        elif current_and_label_table.loc[index, "add info"] == 1:
            continue
        elif current_and_label_table.loc[index, "add info"] == 2:
            file_name = current_and_label_table.loc[index, "Cell ID"]
            Scala_2019_cell_table.loc[Scala_2019_cell_table['file_name']==file_name,"Stim_amp_pA"]+=Scala_2019_cell_table.loc[Scala_2019_cell_table['file_name']==file_name,"Holding_current_pA"]
        elif current_and_label_table.loc[index, "add info"] == 3:
            file_name = current_and_label_table.loc[index, "Cell ID"]
            Scala_2019_cell_table.loc[Scala_2019_cell_table['file_name']==file_name,"Stim_amp_pA"]+=20
            
    Scala_2019_pop_name = Scala_2019_pop_name.loc[Scala_2019_pop_name['Cell_id'].isin(Scala_2019_cell_table.loc[:,'Cell_id'].unique()),:]
    
    return Scala_2019_cell_table, Scala_2019_pop_name


import pynwb
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
import Ordinary_functions as ordifunc
def get_Scala_2021_cell_table(directory):
    
    cell_table = pd.DataFrame(columns = ['Folder', 
                               'Cell_file', 
                               "Cell_id",
                               "Animal_id", 
                               'Sample_id',
                               'Protocol_type',
                               "Targeted_layer",
                               "Inferred_layer",
                               "Cre_line",
                               "Genotype",
                               "Sweep_id",
                               "Stimulus_amplitude_pA",
                               "Membrane_trace_size",
                               "Membrane_trace_rate",
                               'Stimulus_trace_size',
                               'Stimulus_trace_rate',
                               "Stim_start_s",
                               "Stim_end_s"])
    
    pre_folder_list = os.listdir(directory)
    folder_list = [x for x in pre_folder_list if 'yaml' not in x]
    problem_cells=[]
    
    #Get file_list in folder
    for folder in tqdm.tqdm(folder_list):
        pre_file_list = os.listdir(os.path.join(directory, folder))
        file_list = [x for x in pre_file_list if '._' not in x]
        
        for file_name in file_list:
            try:
                mouse_id = folder.split("-")[-1]
                cell_id = file_name.split('-')[-3]
                sample_id = file_name.split('-')[-1]
                
                nwb_file_path = (os.path.join(directory, folder,file_name))
                
                with NWBHDF5IO(nwb_file_path, mode='r') as io:
                        nwbfile = io.read()
                        protocol_type = nwbfile.experiment_description
                        
                        targeted_layer = nwbfile.lab_meta_data['DandiIcephysMetadata'].targeted_layer
                        inferred_layer = nwbfile.lab_meta_data['DandiIcephysMetadata'].inferred_layer
                        
                        
                        cre_line = nwbfile.subject.fields['cre']
                        genotype = nwbfile.subject.fields['genotype']
                        
                        sweep_list = list(nwbfile.acquisition.keys())
                        
                        
                        
                        stimulus_sweeps = list(nwbfile.stimulus.keys())
                        
                        # Get the sweep data
                        
                        for sweep in sweep_list:
                            sweep_id = str(sweep[-3:])
                            potential_sweep_name = f'CurrentClampSeries{sweep_id}'
                            input_current_sweep_name = f'CurrentClampStimulusSeries{sweep_id}'
                            
                            membrane_sweep_data = nwbfile.acquisition[potential_sweep_name]
                            membrane_potential_trace = membrane_sweep_data.data[:]
                            membrane_trace_size = len(membrane_potential_trace)
                            membrane_potential_rate = membrane_sweep_data.rate
                            
                            if input_current_sweep_name in list(nwbfile.stimulus.keys()):
                            
                                stimulus_data = nwbfile.stimulus[input_current_sweep_name].data[:]
                                stimulus_trace_size = len(membrane_potential_trace)
                                stimulus_rate = nwbfile.stimulus[input_current_sweep_name].rate
                                
                                stimulus_data_extended = np.pad(stimulus_data, (0, len(membrane_potential_trace) - len(stimulus_data)), constant_values=0)
                                sweep_duration = len(membrane_potential_trace)/membrane_potential_rate
                                time_trace=np.arange(0,sweep_duration,(1/membrane_potential_rate))
                                
                                TVC_table = pd.DataFrame({'Time_s':time_trace,
                                                          'Membrane_potential_mV':membrane_potential_trace,
                                                          'Input_current_pA' : stimulus_data_extended})
                                stim_start, stim_end = ordifunc.estimate_trace_stim_limits(TVC_table, 0.6, False)
                                
                                stim_amp = np.mean(TVC_table.loc[(TVC_table['Time_s']> stim_start)&(TVC_table['Time_s']< stim_end),'Input_current_pA'])
                                
                                new_line = pd.DataFrame([folder, 
                                                         file_name, 
                                                         cell_id, 
                                                         mouse_id,
                                                         sample_id, 
                                                         protocol_type, 
                                                         targeted_layer,
                                                         inferred_layer,
                                                         cre_line,
                                                         genotype,
                                                         sweep_id,
                                                         stim_amp,
                                                         membrane_trace_size,
                                                         membrane_potential_rate,
                                                         stimulus_trace_size,
                                                         stimulus_rate,
                                                         stim_start,
                                                         stim_end]).T
                                new_line.columns=cell_table.columns
                                cell_table = pd.concat([cell_table, new_line], ignore_index = True)
                        
            except Exception as e:
                
                problem_cells.append(file_name)
    
    cell_table['Sample_id'] = cell_table['Sample_id'].apply(lambda x: x.split('_')[0])
    cell_table['Cell_id'] = cell_table.apply(lambda row: f"{row['Animal_id']}_{row['Cell_id']}_{row['Sample_id']}", axis=1)
    return cell_table, problem_cells
                
    
    
    
def create_Scala_2021_pop_class_table(cell_table):
    
    scala_pop_class_table = pd.DataFrame(columns=['Cell_id', 'General_area', 'Sub_area', 'Layer',
           'Dendrite_type', 'Hemisphere', 'cell_reporter_status', 'line_name',
           'Database', 'Experiment', 'Animal', 'Exc_Inh', 'Exc_Inh_FT',
           'Cell_type', 'Layer_second','Recording_Temperature'])
    
    for cell_id in cell_table.loc[:,'Cell_id'].unique():
        General_area = 'MO'
        Sub_area = 'MOp'
        
        sub_cell_table = cell_table.loc[cell_table['Cell_id']==cell_id,:].copy()
        sub_cell_table = sub_cell_table.reset_index(drop = True)
        targetted_layer = sub_cell_table.loc[0,'Targeted_layer']
        infered_layer = sub_cell_table.loc[0,"Inferred_layer"]
        if targetted_layer == infered_layer:
            
            Layer = targetted_layer
            Layer_second = targetted_layer
        else:
            Layer = "Unknown"
            Layer_second = "Unknown"
        
        Dendrite_type = "Unknown"
        Hemisphere = "Unknown"
        cre_line =  sub_cell_table.loc[0,"Cre_line"]
        if "+" in cre_line:
            cell_reporter_status = "positive"
        elif '-' in cre_line:
            cell_reporter_status = "negative"
        else:
            cell_reporter_status = "Unknown"
        
        line_name = sub_cell_table.loc[0,"Genotype"]
        Database = "Scala_2021"
        Experiment = "In_Vitro"
        Animal = "Mouse"
        
        if cre_line in ["PV+", "VIP+", "NPY+", "SST+", "VIPR2+", "VIAAT+"]:
            Exc_Inh = "Inhibitory"
            if 'PV' in cre_line:
                Cell_type = 'PValb'
            elif "VIP" in cre_line:
                Cell_type = "Vip"
            elif "NPY" in cre_line:
                Cell_type = "NPY"
            elif "SST" in cre_line:
                Cell_type = "Sst"
            else:
                Cell_type = "--"
            
        elif cre_line in ["SLC17a8+", "SLC17a8-iCre+","SCNN1"]:
            Exc_Inh = "Excitatory"
            Cell_type = "Excitatory"
        else:
            Exc_Inh = "--"
            Cell_type = "--"
        
        Exc_Inh_FT = "--"
        
        dataset = sub_cell_table.loc[0,"Dandi_Dataset"]
        if "Room" in dataset:
            recording_temperature = "25 C"
        else:
            recording_temperature = "34 C"
        
        new_line = pd.DataFrame([cell_id,
                                 General_area,
                                 Sub_area, 
                                 Layer, 
                                 Dendrite_type, 
                                 Hemisphere, 
                                 cell_reporter_status, 
                                 line_name, 
                                 Database, 
                                 Experiment,
                                 Animal, 
                                 Exc_Inh,
                                 Exc_Inh_FT, 
                                 Cell_type, 
                                 Layer_second,
                                 recording_temperature]).T
        new_line.columns= scala_pop_class_table.columns
        scala_pop_class_table = pd.concat([scala_pop_class_table,new_line ], ignore_index = True)
    return scala_pop_class_table  
            
def get_trace_Scala_2021(original_cell_file_folder,cell_id,sweep_list,cell_sweep_table):

    first_sweep_id = sweep_list[0]

    cell_file = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==first_sweep_id),"Cell_file"].values[0]
    cell_folder = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==first_sweep_id),"Folder"].values[0]
    cell_dandi_folder = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)&(cell_sweep_table['Sweep_id']==first_sweep_id),"Dandi_Dataset"].values[0]
    nwb_file_path = os.path.join(original_cell_file_folder, cell_dandi_folder,cell_folder, cell_file)
    
    time_trace_list = []
    potential_trace_list = []
    current_trace_list = []
    stim_start_time_list = []
    stim_end_time_list = []
    
    with NWBHDF5IO(nwb_file_path, mode='r') as io:
            nwbfile = io.read()
            
            for sweep_id in sweep_list:
                sweep = sweep_id.split('_')[-1]
                potential_sweep_name = f'CurrentClampSeries{sweep}'
                input_current_sweep_name = f'CurrentClampStimulusSeries{sweep}'
                
                membrane_sweep_data = nwbfile.acquisition[potential_sweep_name]
                potential_trace = membrane_sweep_data.data[:]
                membrane_trace_size = len(potential_trace)
                membrane_potential_rate = membrane_sweep_data.rate
                
                stimulus_data = nwbfile.stimulus[input_current_sweep_name].data[:]
                stimulus_trace_size = len(potential_trace)
                stimulus_rate = nwbfile.stimulus[input_current_sweep_name].rate
                
                current_trace = np.pad(stimulus_data, (0, len(potential_trace) - len(stimulus_data)), constant_values=0)
                sweep_duration = len(potential_trace)/membrane_potential_rate
                time_trace=np.arange(0,sweep_duration,(1/membrane_potential_rate))
                
                TVC_table = pd.DataFrame({'Time_s':time_trace,
                                          'Membrane_potential_mV':potential_trace,
                                          'Input_current_pA' : current_trace})
                
                
                
                stim_start_time, stim_end_time = ordifunc.estimate_trace_stim_limits(TVC_table, 0.6, False)
                
                time_trace_list.append(time_trace)
                potential_trace_list.append(potential_trace)
                current_trace_list.append(current_trace)
                stim_start_time_list.append(stim_start_time)
                stim_end_time_list.append(stim_end_time)
            
    return time_trace_list, potential_trace_list, current_trace_list, stim_start_time_list, stim_end_time_list
                
            
            
            
            
            
            
        
    # for root, dirs, files in os.walk(directory):
    #     print(dirs)
    #     # for file in files:
    #     #     file_path = os.path.join(root, file)
    #     #     # Do something with each file
    #     #     print(file_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        