#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:06:37 2023

@author: julienballbe
"""

import argschema as ags

import numpy as np
import pandas as pd

import os
import importlib
from tqdm import tqdm
import warnings
import traceback
import random
import time

import Sweep_analysis as sw_an
import Spike_analysis as sp_an
import Firing_analysis as fir_an
import Ordinary_functions as ordifunc
import Sweep_QC_analysis as sw_qc

import concurrent.futures


class DatabaseClass(ags.schemas.DefaultSchema):
    
    database_name = ags.fields.String(required=True)
    
    original_file_directory = ags.fields.InputDir(required=True,
                                                  description = "Indicate directory to original cell files")
    
    path_to_db_script_folder = ags.fields.String(required = True,
                                                description = "Path to folder containning database-specific python script")
    
    python_file_name = ags.fields.String(required = True,
                                                description = "Name of python file(finishing by .py) containing the function to get traces")
    
    db_function_name = ags.fields.String(required = True,
                                                description = "Name of the function used to get the original database traces")
    
    
    
    db_cell_sweep_csv_file = ags.fields.String(required = True,
                                                description = "Path to csv file indicating for each cell of the database, the sweeps to consider")
    
    
    db_population_class_file = ags.fields.String(required = True,
                                                description = "Path to csv file indicating cells' metadata ")
    
    db_stimulus_duration = ags.fields.Float(required=True,
                                            description = "Duration of the stimulus for a given database")
    
    stimulus_time_provided = ags.fields.Boolean(required=True,
                                                 default=False)
    
    
    

class GeneralParameters(ags.ArgSchema):

    
    
    path_to_saving_file = ags.fields.String()
    
    path_to_QC_file = ags.fields.String()
    
    DB_parameters = ags.fields.Nested(DatabaseClass,
                                  required=True,
                                  many=True,
                                  description="Schema for processing data originating from a specific Database")
    

def main (path_to_saving_file, path_to_QC_file, DB_parameters, **kwargs):
    
    problem_cell = []
    nb_of_workers_to_use = int(input('\u001b[1m \nIndicate the number of CPU cores you want to allocate for the analysis pipeline; Set 0 for default (half of the computer"s CPU cores) \u001b[0m \n' ))
    list_which_analysis_to_perform = str(input('\u001b[1m \nIndicate the analysis you want to perform between "Metadata","Sweep analysis", "Spike analysis", "Firing analysis". If you want multiple, enter each one separated by "-". If you want all the analysis to be performed enter "All".  \u001b[0m \n' ))
    analysis_list = ["Metadata","Sweep analysis", "Spike analysis", "Firing analysis","All"]
    analysis_to_perform_to_test = list_which_analysis_to_perform.split('-')
    analysis_to_perform=[]
    for i in analysis_to_perform_to_test:
        while i.strip() not in analysis_list:
            i = str(input(f'\u001b[1m \n {i} not recognized. Please enter one of "Metadata","Sweep analysis", "Spike analysis", "Firing analysis", "All" \u001b[0m \n' ))
        analysis_to_perform.append(i)
    
    overwrite_cell_files = eval(input('\u001b[1m \n If a cell already has a file in the saving folder, do you want to overwrite the existing file? Answer True or False. If False, then a cell which already has a corresponding file will not be processed by the pipeline. \u001b[0m \n'))
    
    for current_db in DB_parameters:
        current_db['path_to_saving_file'] = path_to_saving_file
        
        
        database = current_db['database_name']
        path_to_python_folder = current_db['path_to_db_script_folder']
        python_file=current_db['python_file_name']
        module=python_file.replace('.py',"")
        full_path_to_python_script=str(path_to_python_folder+python_file)
        
        spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)
        DB_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DB_module)
        
        
        db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
        
        
        
        cell_id_list = db_cell_sweep_file['Cell_id'].unique()
       
        random.shuffle(cell_id_list)
        if nb_of_workers_to_use <= 0:
            nb_of_cpu_cores_available = int(os.cpu_count())
            nb_of_workers_to_use = int(nb_of_cpu_cores_available//2)
            if nb_of_workers_to_use == 0:
                nb_of_workers_to_use = 1
            
            print(f' \u001b[1m No number of CPU cores indicated in the json configuration file. The number of CPU cores available is {nb_of_cpu_cores_available}, thus the number of CPU cores used is {nb_of_workers_to_use} \u001b[0m')
            
        
        
        args_list = [[x,
                      current_db,
                      module,
                      full_path_to_python_script,
                      path_to_QC_file, 
                      path_to_saving_file,
                      overwrite_cell_files,
                      analysis_to_perform] for x in cell_id_list]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers = nb_of_workers_to_use) as executor:
            problem_cell_list = {executor.submit(cell_processing, x): x for x in args_list}
            for f in tqdm(concurrent.futures.as_completed(problem_cell_list),total = len(cell_id_list), desc=f'Processing {database}'):
                cell_id_problem = f.result()
                if cell_id_problem is not None:
                    problem_cell.append(f.result())

        print(f'Database {database} processed')
    
    print('Done')
    return f'Problem occured with cells : {problem_cell}'
    
     
                    
def cell_processing(args_list):
    cell_id,current_db, module, full_path_to_python_script, path_to_QC_file,  path_to_saving_file,overwrite_cell_files, analysis_to_perform = args_list
    try:
        
        
        
        saving_file_cell = str(path_to_saving_file+"Cell_"+str(cell_id)+".h5")
        if overwrite_cell_files == True or overwrite_cell_files == False and os.path.exists(saving_file_cell) == False:
            
            
            db_original_file_directory = current_db['original_file_directory']
            
            db_population_class = pd.read_csv(current_db['db_population_class_file'],sep =',',encoding = "unicode_escape")
            
            db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
            
            nb_of_workers_to_use=0
            
            if 'All' in analysis_to_perform : 
                saving_dict = {}
                processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
            
    
                Full_SF_dict='--'
                cell_sweep_info_table='--'
                cell_Sweep_QC_table='--'
                cell_fit_table='--'
                cell_adaptation_table = '--'
                cell_feature_table='--'
                Metadata_dict='--'
                
                current_process = "Get traces"
                
                current_cell_sweep_list = db_cell_sweep_file.loc[db_cell_sweep_file['Cell_id'] == cell_id, "Sweep_id"].to_list()
                # Gather for the different sweeps of the cell, time, voltage and current traces, as weel as stimulus start and end times
                
                
                # args_list = [[module,
                #               full_path_to_python_script,
                #               current_db["db_function_name"],
                #               db_original_file_directory,
                #               cell_id,
                #               x,
                #               db_cell_sweep_file,
                #               current_db["stimulus_time_provided"],
                #               current_db["db_stimulus_duration"]] for x in current_cell_sweep_list]
                
                args_list = [module,
                              full_path_to_python_script,
                              current_db["db_function_name"],
                              db_original_file_directory,
                              cell_id,
                              current_cell_sweep_list,
                              db_cell_sweep_file,
                              current_db["stimulus_time_provided"],
                              current_db["db_stimulus_duration"]]
                
                
                Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(args_list, nb_of_workers_to_use, processing_table)
                
                current_process = "Sweep analysis"
                cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, nb_of_workers_to_use, processing_table)
                
                
                #cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_sweep_info_table, processing_table)
                cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                
                
                current_process = "Spike analysis"
                Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table)
                
                current_process = "Firing analysis"
                cell_feature_table, cell_fit_table,cell_adaptation_table, processing_table = perform_firing_related_analysis(Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table, processing_table)
                
                current_process = "Metadata"
                Metadata_dict = db_population_class.loc[db_population_class['Cell_id'] == cell_id,:].iloc[0,:].to_dict()
                
                Sweep_analysis_dict = {"Sweep info" : cell_sweep_info_table,
                                       "Sweep QC" : cell_Sweep_QC_table}
                
                firing_dict={"Cell_feature" : cell_feature_table,
                             "Cell_fit" : cell_fit_table,
                             "Cell_Adaptation" : cell_adaptation_table}
                
                saving_dict = {"Sweep analysis" : Sweep_analysis_dict,
                               "Spike analysis" : Full_SF_dict,
                               "Firing analysis" : firing_dict,
                               "Metadata" : Metadata_dict}
                
            else:
                saving_dict = {}
                current_cell_sweep_list = db_cell_sweep_file.loc[db_cell_sweep_file['Cell_id'] == cell_id, "Sweep_id"].to_list()
                if "Sweep analysis" in analysis_to_perform:
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
                    current_process = "Sweep analysis"
                    is_Processing_report_df = pd.DataFrame()
                    if os.path.exists(saving_file_cell) == True:
                        current_db_df = pd.DataFrame(current_db,index=[0])
                        cell_dict = ordifunc.read_cell_file_h5(cell_id, current_db_df, selection = ['Processing_report'])
                        is_Processing_report_df = cell_dict['Processing_table']
                    if is_Processing_report_df.shape[0]!=0:
                        
                        processing_table = is_Processing_report_df
                        processing_table = processing_table[processing_table['Processing_step']!="Sweep analysis"]
                        processing_table = processing_table[processing_table['Processing_step']!="Sweep QC"]
                        
                        
                    # args_list = [[module,
                    #               full_path_to_python_script,
                    #               current_db["db_function_name"],
                    #               db_original_file_directory,
                    #               cell_id,
                    #               x,
                    #               db_cell_sweep_file,
                    #               current_db["stimulus_time_provided"],
                    #               current_db["db_stimulus_duration"]] for x in current_cell_sweep_list]
                    
                    args_list = [module,
                                  full_path_to_python_script,
                                  current_db["db_function_name"],
                                  db_original_file_directory,
                                  cell_id,
                                  current_cell_sweep_list,
                                  db_cell_sweep_file,
                                  current_db["stimulus_time_provided"],
                                  current_db["db_stimulus_duration"]]
                    
                    Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(args_list, nb_of_workers_to_use, processing_table)
                    
                    
                    cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, nb_of_workers_to_use, processing_table)
                    
                    
                    #cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_sweep_info_table, processing_table)
                    cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                
                    Sweep_analysis_dict = {"Sweep info" : cell_sweep_info_table,
                                           "Sweep QC" : cell_Sweep_QC_table}
                    dict_to_add = {"Sweep analysis":Sweep_analysis_dict}
                    saving_dict.update(dict_to_add)
                    saving_dict.update({"Processing report" : processing_table})
                if "Spike analysis" in analysis_to_perform:
                    current_process = "Spike analysis"
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
                    # args_list = [[module,
                    #               full_path_to_python_script,
                    #               current_db["db_function_name"],
                    #               db_original_file_directory,
                    #               cell_id,
                    #               x,
                    #               db_cell_sweep_file,
                    #               current_db["stimulus_time_provided"],
                    #               current_db["db_stimulus_duration"]] for x in current_cell_sweep_list]
                    
                    args_list = [module,
                                  full_path_to_python_script,
                                  current_db["db_function_name"],
                                  db_original_file_directory,
                                  cell_id,
                                  current_cell_sweep_list,
                                  db_cell_sweep_file,
                                  current_db["stimulus_time_provided"],
                                  current_db["db_stimulus_duration"]]
                    
                    Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(args_list, nb_of_workers_to_use, processing_table)
                    
                    is_sweep_info_table = pd.DataFrame()
                    is_sweep_QC_table = pd.DataFrame()
                    is_Processing_report_df = pd.DataFrame()
                    if os.path.exists(saving_file_cell) == True:
                        #_, _, _, _, is_sweep_info_table, is_sweep_QC_table, _, _,is_Processing_report_df = ordifunc.read_cell_file_h5(cell_id, current_db, selection = ['Sweep analysis','Sweep QC','Processing_report'])
                        current_db_df = pd.DataFrame(current_db,index=[0])
                        cell_dict = ordifunc.read_cell_file_h5(cell_id, current_db_df, selection = ['Sweep analysis','Sweep QC','Processing_report'])
                        is_sweep_info_table = cell_dict["Sweep_info_table"]
                        is_sweep_QC_table = cell_dict['Sweep_QC_table']
                        is_Processing_report_df = cell_dict['Processing_table']
                    if is_sweep_info_table.shape[0]==0 or is_sweep_QC_table.shape[0]==0 or is_Processing_report_df.shape[0]==0 or os.path.exists(saving_file_cell) == False:
    
                            cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, nb_of_workers_to_use, processing_table)
                            
                            
                            #cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_sweep_info_table, processing_table)
                            cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                    else: 
                        cell_sweep_info_table = is_sweep_info_table
                        cell_Sweep_QC_table = is_sweep_QC_table
                        processing_table = is_Processing_report_df
                        processing_table = processing_table[processing_table['Processing_step'] != "Spike analysis"]
                        
                    Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table)
                    
                    saving_dict.update({"Spike analysis" : Full_SF_dict})
                    saving_dict.update({"Processing report" : processing_table})
                if "Firing analysis" in analysis_to_perform:
                    current_process = "Firing analysis"
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
                    # args_list = [[module,
                    #               full_path_to_python_script,
                    #               current_db["db_function_name"],
                    #               db_original_file_directory,
                    #               cell_id,
                    #               x,
                    #               db_cell_sweep_file,
                    #               current_db["stimulus_time_provided"],
                    #               current_db["db_stimulus_duration"]] for x in current_cell_sweep_list]
                    
                    args_list = [module,
                                  full_path_to_python_script,
                                  current_db["db_function_name"],
                                  db_original_file_directory,
                                  cell_id,
                                  current_cell_sweep_list,
                                  db_cell_sweep_file,
                                  current_db["stimulus_time_provided"],
                                  current_db["db_stimulus_duration"]]
                    
                    Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(args_list, nb_of_workers_to_use, processing_table)
                    
                    is_sweep_info_table = pd.DataFrame()
                    is_sweep_QC_table = pd.DataFrame()
                    is_SF_table = pd.DataFrame()
                    is_Processing_report_df = pd.DataFrame()
    
                    if os.path.exists(saving_file_cell) == True:
    
                        #_, _, is_SF_table, _, is_sweep_info_table, is_sweep_QC_table, _, _,is_Processing_report_df = ordifunc.read_cell_file_h5(cell_id, current_db, selection = ['Sweep analysis','Sweep QC', 'TVC_SF','Processing_report'])
                        current_db_df = pd.DataFrame(current_db,index=[0])
                        cell_dict = ordifunc.read_cell_file_h5(cell_id, current_db_df, selection = ['Sweep analysis','Sweep QC', 'TVC_SF','Processing_report'])
                        is_SF_table = cell_dict['Full_SF_table']
                        is_sweep_info_table = cell_dict["Sweep_info_table"]
                        is_sweep_QC_table = cell_dict['Sweep_QC_table']
                        is_Processing_report_df = cell_dict['Processing_table']
                    
                    if is_SF_table.shape[0]==0 or is_sweep_info_table.shape[0]==0 or is_sweep_QC_table.shape[0]==0 or is_Processing_report_df.shape[0]==0 or os.path.exists(saving_file_cell) == False:
                            
                            cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, nb_of_workers_to_use, processing_table)
                            
                            #cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_sweep_info_table, processing_table)
                            cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                            
                            Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table)
                    else: 
    
                        cell_sweep_info_table = is_sweep_info_table
                        cell_Sweep_QC_table = is_sweep_QC_table
                        processing_table = is_Processing_report_df
                        Full_SF_table = is_SF_table
                        processing_table = processing_table[processing_table['Processing_step'] != "Firing analysis"]

                    cell_feature_table, cell_fit_table, cell_adaptation_table, processing_table = perform_firing_related_analysis(Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table, processing_table)
                    
                    firing_dict={"Cell_feature" : cell_feature_table,
                                 "Cell_fit" : cell_fit_table,
                                 "Cell_Adaptation" : cell_adaptation_table}
                    
                    saving_dict.update({"Firing analysis" : firing_dict})
                    saving_dict.update({"Processing report" : processing_table})
                if 'Metadata' in analysis_to_perform:
                    current_process = "Metadata"

                    
                    Metadata_dict = db_population_class.loc[db_population_class['Cell_id'] == cell_id,:].iloc[0,:].to_dict()
                    
                    saving_dict.update({"Metadata" : Metadata_dict})
                    
            
    
            ordifunc.write_cell_file_h5(saving_file_cell,
                               saving_dict,
                               overwrite=overwrite_cell_files,
                               selection = analysis_to_perform)
            
                
                
                
        
    except:

        error= traceback.format_exc()
        # problem_cell.append(str(cell_id))
        message = str('Error in '+str(current_process)+': '+str(error))
        new_line = pd.DataFrame([current_process,'--',message]).T

        new_line.columns=processing_table.columns
        processing_table = pd.concat([processing_table,new_line],ignore_index=True,axis=0)
        saving_dict.update({"Processing report" : processing_table})

        ordifunc.write_cell_file_h5(saving_file_cell,
                           saving_dict,
                           overwrite=overwrite_cell_files,
                           selection = analysis_to_perform)
        return cell_id
        

def append_processing_table(current_process,processing_table, warning_list, processing_time):
    
    if len(warning_list) == 0:
        new_line = pd.DataFrame([current_process,str(str(round((processing_time),3))+'s'),'--']).T
        new_line.columns=processing_table.columns
        processing_table = pd.concat([processing_table,new_line],ignore_index=True,axis=0)
    else:
        for current_warning in warning_list:
            message = str('Warnings in '+str(current_warning.filename)+str(' line:')+str(current_warning.lineno)+str(':')+str(current_warning.message))
            new_line = pd.DataFrame([current_process,str(str(round((processing_time),3))+'s'),message]).T
            new_line.columns=processing_table.columns
            processing_table = pd.concat([processing_table,new_line],ignore_index=True,axis=0)
    
    return processing_table
        
    
def get_db_traces(args_list, nb_of_workers_to_use, processing_table):

   
    current_process = "get_TVC_tables"
    start_time=time.time()
    with warnings.catch_warnings(record=True) as warning_TVC:
        Full_TVC_table = pd.DataFrame(columns=['Sweep','TVC'])
        cell_stim_time_table = pd.DataFrame(columns=['Sweep','Stim_start_s', 'Stim_end_s'])
        
        # for x in args_list:
        #     result = sw_an.get_TVC_table(x)
        #     Full_TVC_table = pd.concat([
        #         Full_TVC_table,result[0]], ignore_index=True)
        #     cell_stim_time_table = pd.concat([
        #         cell_stim_time_table , result[1]], ignore_index=True)
        
        Full_TVC_table, cell_stim_time_table = sw_an.get_TVC_table(args_list)
        
        Full_TVC_table.index = Full_TVC_table.loc[:,"Sweep"]
        Full_TVC_table.index = Full_TVC_table.index.astype(str)
        
        cell_stim_time_table.index = cell_stim_time_table.loc[:,"Sweep"]
        cell_stim_time_table.index = cell_stim_time_table.index.astype(str)
        cell_stim_time_table=cell_stim_time_table.astype({'Stim_start_s':float, 'Stim_end_s':float})
    end_time=time.time()
    processing_time = end_time-start_time
    processing_table = append_processing_table(current_process, processing_table, warning_TVC, processing_time)

        
    return Full_TVC_table, cell_stim_time_table, processing_table

def perform_sweep_related_analysis (Full_TVC_table, cell_stim_time_table, nb_of_workers_to_use, processing_table):
    
    current_process = "Sweep Analysis"
    start_time=time.time()
    with warnings.catch_warnings(record=True) as warning_cell_sweep_table:
        
        cell_sweep_info_table = sw_an.sweep_analysis_processing(Full_TVC_table, cell_stim_time_table, nb_of_workers_to_use)

    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_sweep_table, processing_time)
    
    return cell_sweep_info_table, processing_table
        
def perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table):
    ### Sweep_QC_table
    current_process = "Sweep QC"
    start_time=time.time()
    QC_function_module = os.path.basename(path_to_QC_file)
    with warnings.catch_warnings(record=True) as warning_cell_QC_table:
        
        cell_Sweep_QC_table, error_message = sw_qc.run_QC_for_cell(Full_TVC_table, cell_sweep_info_table, QC_function_module, path_to_QC_file)
        
        #cell_Sweep_QC_table = sw_qc.create_cell_sweep_QC_table_new_version(cell_sweep_info_table)
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_QC_table, processing_time)
    
    return cell_Sweep_QC_table, processing_table
    
 

def perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table):
    current_process = "Spike analysis"
    start_time=time.time()
    
    with warnings.catch_warnings(record=True) as warning_Full_SF_table:
        Full_SF_dict = sp_an.create_cell_Full_SF_dict_table(
            Full_TVC_table.copy(), cell_sweep_info_table.copy())

        Full_SF_table = sp_an.create_Full_SF_table(
            Full_TVC_table.copy(), Full_SF_dict.copy(), cell_sweep_info_table.copy())
        

        Full_SF_table.index = Full_SF_table.index.astype(str)
        
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_Full_SF_table, processing_time)
    
    return Full_SF_dict, Full_SF_table, processing_table
    
   

def perform_firing_related_analysis(Full_SF_table, cell_sweep_info_table,cell_Sweep_QC_table, processing_table):
    current_process = "Firing analysis"
    start_time=time.time()

    with warnings.catch_warnings(record=True) as warning_cell_feature_table:
        response_duration_dictionnary={
            'Time_based':[.005, .010, .025, .050, .100, .250, .500],
            'Index_based':list(np.arange(2,18)),
            'Interval_based':list(np.arange(1,17))}

        cell_feature_table, cell_fit_table = fir_an.compute_cell_features(Full_SF_table,
                                                                   cell_sweep_info_table,
                                                                   response_duration_dictionnary,
                                                                   cell_Sweep_QC_table)
        
        cell_adaptation_table = fir_an.compute_cell_adaptation_behavior(Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table)
        
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_feature_table, processing_time)
    
    return cell_feature_table, cell_fit_table, cell_adaptation_table ,processing_table
    
    
    
    
if __name__ == "__main__":
    
    
    mod = ags.ArgSchemaParser(schema_type=GeneralParameters)
    result = main(**mod.args)
    
    
    print('Failed_cells:', result)
    
    