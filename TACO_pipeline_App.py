#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:19:51 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
import os
import importlib
import json
import ast
from shiny import App, Inputs, Outputs, Session, render, ui,reactive,req
import math
import time
import concurrent
import random
import Analysis_pipeline as analysis_pipeline
import Sweep_analysis as sw_an
import Spike_analysis as sp_an
import Firing_analysis as fir_an
import Ordinary_functions as ordifunc
import plotnine as p9
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget
import plotly.express as px
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import io
from sklearn.metrics import r2_score 
import sys
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots
from sklearn.preprocessing import minmax_scale 

# # Load the external script dynamically
# script_path = "Analysis_pipeline.py"
# spec = importlib.util.spec_from_file_location("Analysis_pipeline", script_path)
# script_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(script_module)


# In fast mode, throttle interval in ms.
FAST_INTERACT_INTERVAL = 60

app_ui = ui.page_fluid(
    ui.h1("Welcome to TACO pipeline"),
    ui.navset_tab_card(
        
        ui.nav(
            "Create JSON configuration file",
            
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.h3("Create JSON configuration file"),
                    ui.output_ui("add_database"),
                    ui.input_action_button("save_json", "Save JSON"),
                    ui.output_ui("save_json_status"),
                    
                    
                ),
                ui.panel_main(
                    ui.layout_column_wrap(
                        2,  # Two columns layout
                        # First column
                        ui.panel_main(
                            ui.input_text(
                                "json_file_path",
                                "Enter File Path for Saving JSON file (ending in .json):",
                                placeholder="Enter full path to save the JSON file, e.g., /path/to/folder/file.json",
                            ),
                            ui.input_action_button("check_dir_1", "Check File Path"),
                            ui.br(),
                            ui.output_ui("status_1"),
                        ),
                        # Second column
                        ui.panel_main(
                            ui.input_text(
                                "Saving_folder",
                                "Enter Saving Directory for Analysis:",
                                placeholder="Enter path to saving folder, e.g., /path/to/folder/",
                            ),
                            ui.input_action_button("check_dir_2", "Check Saving Folder Path"),
                            ui.br(),
                            ui.output_ui("status_2"),
                        ),
                        # Second column
                        ui.panel_main(
                            ui.input_text(
                                "Path_to_QC_file",
                                "Enter Path to Quality Criteria File:",
                                placeholder="Enter path to QC file, e.g., /path/to/folder/file.py",
                            ),
                            ui.input_action_button("check_QC_file_path", "Check QC file path"),
                            ui.br(),
                            ui.output_ui("status_QC_file"),
                        ),
                        

                    ),
                    ui.hr(),
                    ui.layout_column_wrap(2,
                        # Third column
                        ui.panel_main(
                            ui.row(
                                # Row containing the buttons and input field
                                
                                ui.input_text(
                                    "database_name",
                                    "Enter Database Name:",
                                    placeholder="Enter the name of the database, e.g., MyDatabase",
                                ),
                            ),
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "original_file_path",
                                              "Path to Original File:",
                                              placeholder="Enter full path to the original file, e.g., /path/to/original/file.csv",
                                          ),),
                                ui.column(4, 
                                          ui.input_action_button("Check_original_file_path", "Check original file path"),),
                                ui.column(4,
                                          ui.output_ui("status_original_file_path"),),),
                                # Row containing the 'original_file_path' input and 'Check original file path' button
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "database_python_script_path",
                                              "Path to database python script:",
                                              placeholder="e.g., /path/to/original/script.py",
                                          )),
                                ui.column(4,
                                          ui.input_action_button("check_database_python_script_path", "Check database python script")),
                                ui.column(4,
                                          ui.output_ui("status_database_script_file")),
                                ui.column(4,
                                          ui.output_ui('import_database_script_file'))
                                ),
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "population_class_file",
                                              "Path to database population class csv file:",
                                              placeholder="e.g., /path/to/original/file.csv",
                                          ),),
                                ui.column(4,
                                          ui.input_action_button("check_population_class_file_path", "Check database population class csv file")),
                                ui.column(4,
                                          ui.output_ui("status_population_class_file"))),
                            
                            
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "cell_sweep_table_file",
                                              "Path to database cell sweep csv file:",
                                              placeholder="e.g., /path/to/original/file.csv",
                                          ),),
                                ui.column(4,
                                          ui.input_action_button("check_cell_sweep_table_file_path", "Check database cell sweep csv file")),
                                ui.column(4,
                                          ui.output_ui("status_cell_sweep_table_file"))),
                            
                            ui.row(
                                   ui.column(4,
                                             ui.input_checkbox("stimulus_time_provided", "Are stimulus start and end times provided?")),
                                   ui.column(4,
                                             ui.input_numeric("stimulus_duration", "Indicate stimulus duration in s", 0.5))
                           
                                
                                
                                
                                
                            ),
                            
                            ui.input_action_button("Add_Database", "Add Database"),
                            ui.br(),
                            
                            
                        ),
                    )
                ),
            ),
            ),
        
        ui.nav(
            "Run Analysis",
            ui.input_numeric("n_CPU", "number of CPU cores to use", value=int(os.cpu_count()/2)),
            ui.input_file("json_file_input", "Upload Json configuration File:"),
            ui.input_selectize("select_analysis", "Choose Analysis to perform:", {"All":"All" ,
                                                                               "Metadata":"Metadata",
                                                                               "Sweep analysis": "Sweep analysis",
                                                                               "Spike analysis":"Spike analysis",
                                                                               "Firing analysis":"Firing analysis"
                                                                               }, multiple=True, selected="All"),
            ui.input_select("overwrite_existing_files", "Overwrite existing files?", ['Yes','No']),
            
            ui.input_action_button("run_analysis", "Run Analysis"),
            ui.output_text_verbatim("script_output"),
            ui.input_text(
                "summary_folder_path",
                "Enter Folder Path for Saving Summary tables file :",
                placeholder="Enter full path to save the summary tables e.g., /path/to/folder/",
            ),
            ui.input_action_button("check_saving_summaryfolder", "Check Folder Path"),
            
            ui.output_ui("summary_folder_status"),
            ui.input_action_button("summarise_analysis", "Summarize Analysis"),
            ui.output_text_verbatim("summarize_analysis_output")),
        
        
        
        ui.nav(
            "Cell Visualization",
            ui.navset_pill_card(
                # Encapsulate existing tabs into the "Cell Visualization" nav_pill
                ui.nav(
                    "Select Cell",
                    ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.input_file("JSON_config_file", 'Choose JSON configuration files'),
                            ui.input_selectize("Cell_file_select", "Select Cell to Analyse", choices=[]),
                            ui.input_action_button('Update_cell_btn', 'Update Cell'),
                        ),
                        ui.panel_main(
                            ui.output_table('config_json_table')
                        ),
                    ),
                ),
                ui.nav(
                    "Cell Information",
                    ui.layout_sidebar(
                        ui.panel_sidebar(width=0),
                        ui.panel_main(
                            ui.navset_pill_card(
                                ui.nav(
                                    "Cell summary",
                                    ui.row(
                                        ui.column(2, ui.output_table("Cell_Metadata_table")),
                                        ui.column(2, ui.output_table("Cell_linear_properties_table")),
                                        ui.column(
                                            4,
                                            ui.output_table("Cell_Firing_properties_table"),
                                            ui.output_table("Cell_Adaptation_table"),
                                        ),
                                    ),
                                ),
                                ui.nav(
                                    "Linear properties",
                                    ui.row(
                                        ui.column(
                                            4,
                                            output_widget("cell_Holding_potential_v_Holding_current_plolty"),
                                            output_widget("cell_SS_potential_v_stim_amp_plolty"),
                                        ),
                                        ui.column(
                                            4,
                                            output_widget("cell_input_resistance_v_stim_amp_plolty"),
                                            output_widget('cell_time_constant_v_stim_amp_plolty'),
                                        ),
                                        ui.column(
                                            4,
                                            output_widget("cell_Holding_current_pA_v_stim_amp_plolty"),
                                            output_widget("cell_Holding_potential_mV_v_stim_amp_plolty"),
                                        ),
                                    ),
                                ),
                                ui.nav(
                                    "Processing report",
                                    ui.output_table('cell_processing_time_table')
                                ),
                            ),
                        ),
                    ),
                ),
                ui.nav(
                    "Sweep Analysis",
                    ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.input_selectize("Sweep_selected", "Select Sweep to Analyse", choices=[]),
                            ui.input_checkbox("BE_correction", "Correct for Bridge Error"),
                            ui.input_checkbox("Superimpose_BE_Correction", "Superimpose BE Correction"),
                            width=2
                        ),
                        ui.panel_main(
                            ui.navset_tab(
                                ui.nav(
                                    "Sweep information",
                                    ui.output_data_frame('Sweep_info_table')
                                ),
                                ui.nav(
                                    'Spike Analysis',
                                    output_widget('Sweep_spike_plotly'),
                                    ui.output_data_frame('Sweep_spike_features_table')
                                ),
                                ui.nav(
                                    "Spike phase plot",
                                    ui.row(
                                        ui.column(6, output_widget("Spike_superposition_plot")),
                                        ui.column(6, output_widget("Spike_phase_plane_plot")),
                                    ),
                                ),
                                ui.nav(
                                    'Sweep_QC',
                                    ui.output_table('Sweep_QC_table')
                                ),
                                ui.nav(
                                    'Bridge Error analysis',
                                    output_widget("BE_plotly"),
                                    ui.output_text_verbatim("BE_value"),
                                    ui.output_table('BE_conditions_table'),
                                ),
                                ui.nav(
                                    'Linear properties analysis',
                                    ui.output_table("linear_properties_conditions_table"),
                                    ui.output_plot("linear_properties_fit"),
                                    ui.output_table('linear_properties_fit_table'),
                                    ui.output_table("linear_properties_table")
                                ),
                            ),
                        ),
                    ),
                ),
                ui.nav(
                    "Firing Analysis",
                    ui.layout_sidebar(
                        ui.panel_sidebar(width=0),
                        ui.panel_main(
                            ui.navset_pill(
                                ui.nav(
                                    "I/O",
                                    ui.input_select(
                                        'Firing_response_type',
                                        "Select response type",
                                        choices=['Time_based', 'Index_based', 'Interval_based']
                                    ),
                                    output_widget("Time_based_IO_plotly"),
                                    ui.row(
                                        ui.column(2, ui.tags.b("Gain"), output_widget("Gain_regression_plot_time_based_Second")),
                                        ui.column(2, ui.tags.b("Threshold"), output_widget("Threshold_regression_plot_Second")),
                                        ui.column(2, ui.tags.b("Saturation Frequency"), output_widget("Saturation_Frequency_regression_plot_Second")),
                                        ui.column(2, ui.tags.b("Saturation Stimulus"), output_widget("Saturation_Stimulus_regression_plot_Second")),
                                        ui.column(2, ui.tags.b("Response Failure Frequency"), output_widget("Response_Failure_Frequency_regression_plot_Second")),
                                        ui.column(2, ui.tags.b("Response Failure Stimulus"), output_widget("Response_Failure_Stimulus_regression_plot_Second"))
                                    ),
                                    ui.row(
                                        ui.column(2, ui.output_table("Gain_regression_table_time_based_Second")),
                                        ui.column(2, ui.output_table("Threshold_regression_table_Second")),
                                        ui.column(2, ui.output_table("Saturation_Frequency_regression_Second")),
                                        ui.column(2, ui.output_table("Saturation_Stimulus_regression_Second")),
                                        ui.column(2, ui.output_table("Response_Fail_Frequency_regression_Second")),
                                        ui.column(2, ui.output_table("Response_Fail_Stimulus_regression_Second"))
                                    )
                                ),
                                ui.nav(
                                    "Details",
                                    ui.row(
                                        ui.input_select(
                                            'Firing_response_type_new',
                                            'Select response type',
                                            choices=["Time_based", "Index_based", "Interval_based"]
                                        ),
                                        ui.input_select('Firing_output_duration_new', 'Select response type', choices=[]),
                                        ui.input_selectize('Plot_new', "Select plot to display", choices=[]),
                                    ),
                                    output_widget("Time_based_IO_plotly_new"),
                                    ui.output_table("IO_fit_parameter_table")
                                ),
                                ui.nav(
                                    "Adaptation",
                                    ui.input_select('Adaptation_feature_to_display', "Select feature ", choices=[]),
                                    output_widget("Adaptation_plotly"),
                                    ui.output_table("Adaptation_table"),
                                ),
                            ),
                        ),
                    ),
                ),
                ui.nav(
                    "In progress",
                    ui.layout_sidebar(
                        ui.panel_sidebar(
                            ui.input_selectize("Sweep_selected_in_progress", "Select Sweep to Analyse", choices=[]),
                            ui.input_selectize(
                                "x_Spike_feature",
                                "Select spike feature X axis",
                                choices=['Downstroke', 'fAHP', 'Fast_Trough', 'Peak', 'Spike_heigth', 'Spike_width_at_half_heigth', 'Threshold', 'Trough', 'Upstroke']
                            ),
                            ui.input_selectize(
                                "y_Spike_feature",
                                "Select spike feature Y axis",
                                choices=["Peak", 'Downstroke', 'fAHP', 'Fast_Trough', 'Spike_heigth', 'Spike_width_at_half_heigth', 'Threshold', 'Trough', 'Upstroke']
                            ),
                            ui.input_selectize("y_Spike_feature_index", "Which spike index for Y axis", choices=['N', "N-1", "N+1"]),
                            ui.input_checkbox("BE_correction_in_process", "Correct for Bridge Error"),
                            width=2
                        ),
                        ui.panel_main(
                            ui.navset_pill(
                                ui.nav(
                                    "Spike feature correlation",
                                    output_widget("Spike_features_index_correlation"),
                                ),
                            ),
                            width=10
                        ),
                    ),
                ),
            ),
        ),
    ),
)

def import_json_config_file(json_config_file):
    i=0
    for file in json_config_file:
        file_path = file['datapath']
        if i == 0:
            config_df = ordifunc.open_json_config_file(file_path)
            i+=1
            
        else:
            new_table = ordifunc.open_json_config_file(file_path)
            config_df=pd.concat([config_df, new_table],ignore_index = True)
            
    return config_df

def estimate_processing_time(start_time, total_iteration,i):
    process_time = time.time()-start_time
    hours = math.floor(process_time / 3600)
    minutes = math.floor((process_time % 3600) / 60)
    seconds = int(process_time % 60)
    formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
    
    estimated_total_time = process_time*total_iteration/i
    estimated_remaining_time = estimated_total_time-process_time
    remaining_hours = math.floor(estimated_remaining_time / 3600)
    remaining_minutes = math.floor((estimated_remaining_time % 3600) / 60)
    remaining_seconds = int(estimated_remaining_time % 60)
    remaining_formatted_time = f"{remaining_hours}:{remaining_minutes:02d}:{remaining_seconds:02d}"
    
    return formatted_time, remaining_formatted_time

def highlight_late(s):
    
    return ['background-color: green' if s_ == True else 'background-color: red' if s_ ==False else 'background-color: white' for s_ in s]

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def get_firing_analysis_tables(cells,response_type,output_duration):
    

    Full_SF_table = cells[2]
    Sweep_info_table = cells[4]
    sweep_QC_table = cells[5]
    cell_fit_table = cells[6]
    cell_feature_table = cells[7]
    
    response_type=response_type.replace(' ','_')
    
    sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
    sub_cell_feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == response_type,:]

    
    
    
    if response_type == 'Time_based':
        response = str(str(int(output_duration*1e3))+'ms')
    else:
        
        response = str(response_type.replace('_based',' ') + str(int(output_duration)))
    
    current_stim_freq_table = fir_an.get_stim_freq_table(
        Full_SF_table.copy(), Sweep_info_table.copy(),sweep_QC_table.copy(), output_duration,response_type)
    current_stim_freq_table['Response'] = response
    
    model = sub_cell_fit_table.loc[sub_cell_fit_table['Output_Duration'] == output_duration,'I_O_obs'].values[0]
    if model == 'Hill' or model == 'Hill-Sigmoid':
        min_stim = np.nanmin(current_stim_freq_table.loc[:,'Stim_amp_pA'])
        max_stim = np.nanmax(current_stim_freq_table.loc[:,'Stim_amp_pA'])
        stim_array = np.arange(min_stim,max_stim,.1)
        Hill_Half_cst = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == output_duration,'Hill_Half_cst'].values[0]
        Hill_amplitude = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == output_duration,'Hill_amplitude'].values[0]
        Hill_coef = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == output_duration,'Hill_coef'].values[0]
        Hill_x0 = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == output_duration,'Hill_x0'].values[0]
        freq_array = fir_an.hill_function(stim_array, Hill_x0, Hill_coef, Hill_Half_cst)
        freq_array *= Hill_amplitude
        
        if model == 'Hill-Sigmoid':
            sigmoid_sigma = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == output_duration,'Sigmoid_sigma'].values[0]
            sigmoid_x0 = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == output_duration,'Sigmoid_x0'].values[0]
            
            freq_array *= fir_an.sigmoid_function(stim_array , sigmoid_x0, sigmoid_sigma)
        
        model_table = pd.DataFrame({'Stim_amp_pA' : stim_array,
                                    "Frequency_Hz" : freq_array})
        model_table['Response'] = response
        Gain = sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == output_duration,'Gain'].values[0]
        Threshold = sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == output_duration,'Threshold'].values[0]
        Intercept = -Gain*Threshold
        IO_freq = stim_array*Gain+Intercept
        
        IO_table = pd.DataFrame({'Stim_amp_pA' : stim_array,
                                    "Frequency_Hz" : IO_freq})
        IO_table['Response'] = response
        IO_table=IO_table.loc[IO_table['Frequency_Hz']>=-10.,:]
        
        if not np.isnan(sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == output_duration,'Saturation_Stimulus'].values[0]):
            Saturation_stimulus =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == output_duration,'Saturation_Stimulus'].values[0]
            Saturation_freq =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == output_duration,'Saturation_Frequency'].values[0]
            Saturation_table = pd.DataFrame({'Stim_amp_pA' : Saturation_stimulus,
                                        "Frequency_Hz" : Saturation_freq,
                                        'Response' : response})
            
        else:
            Saturation_table = pd.DataFrame()
    
        
        current_stim_freq_table=1
        model_table=1
        IO_table=1
        Saturation_table=1
        print("Response_type=",response_type)
        print(current_stim_freq_table,model_table,IO_table,Saturation_table)
        return current_stim_freq_table, model_table, IO_table, Saturation_table
    
def get_BE_value_and_table(cell_dict, sweep_id):
    
    Full_TVC_table = cell_dict['Full_TVC_table']
    Full_SF_table = cell_dict['Full_SF_table']
    sweep_info_table = cell_dict['Sweep_info_table']
    Sweep_QC_table = cell_dict['Sweep_QC_table']
    if np.isnan(sweep_info_table.loc[sweep_info_table['Sweep']==sweep_id,"Bridge_Error_GOhms"].values[0]) == False:
        TVC_table = Full_TVC_table.loc[sweep_id,"TVC"]
        stim_amp = sweep_info_table.loc[sweep_info_table['Sweep']==sweep_id,"Stim_SS_pA"].values[0]
        Stim_start_s = sweep_info_table.loc[sweep_info_table['Sweep']==sweep_id,"Stim_start_s"].values[0]
        Stim_end_s = sweep_info_table.loc[sweep_info_table['Sweep']==sweep_id,"Stim_end_s"].values[0]
        BE_val, condition_table = sw_an.estimate_bridge_error_test(TVC_table, stim_amp, Stim_start_s, Stim_end_s,do_plot=False)
        
        return BE_val, condition_table
        
    
def get_firing_analysis(cell_dict,response_type):
    
    
    
    Sweep_info_table = cell_dict['Sweep_info_table']
    Full_SF_table = cell_dict["Full_SF_table"]
    
    sweep_QC_table = cell_dict["Sweep_QC_table"]
    cell_fit_table = cell_dict["Cell_fit_table"]
    cell_feature_table = cell_dict['Cell_feature_table']
    
    


    response_type=response_type.replace(' ','_')
    
    sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
    sub_cell_feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == response_type,:]
    output_response_list = list(sub_cell_fit_table.loc[:,'Output_Duration'])
    
    stim_freq_table_list=[]
    model_table_list = []
    IO_table_list = []
    Saturation_table_list = []

    for current_response_duration in output_response_list:
        
        if response_type == 'Time_based':
            response = str(str(int(current_response_duration*1e3))+'ms')
        else:
            
            response = str(response_type.replace('_based',' ') + str(int(current_response_duration)))
            current_response_duration = int(current_response_duration)
        current_stim_freq_table = fir_an.get_stim_freq_table(
            Full_SF_table.copy(), Sweep_info_table.copy(),sweep_QC_table.copy(), current_response_duration,response_type)
        
        
        current_stim_freq_table['Response'] = response
        stim_freq_table_list.append(current_stim_freq_table)
        
        model = sub_cell_fit_table.loc[sub_cell_fit_table['Output_Duration'] == current_response_duration,'I_O_obs'].values[0]
        if model == 'Hill' or model == 'Hill-Sigmoid':
            
            model_table = fir_an.fit_IO_relationship_NEW_TEST(current_stim_freq_table, "--")[1]
            
            min_stim = np.nanmin(current_stim_freq_table.loc[:,'Stim_amp_pA'])
            max_stim = np.nanmax(current_stim_freq_table.loc[:,'Stim_amp_pA'])
            stim_array = np.arange(min_stim,max_stim,.1)
            
            model_table['Response'] = response
            Gain = sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Gain'].values[0]
            
            plot_list = fir_an.get_IO_features_NEW_TEST(current_stim_freq_table, response_type, current_response_duration, True,True)
            IO_plot = plot_list["3-IO_fit"]
            Intercept = IO_plot['intercept']
            
            IO_freq = stim_array*Gain+Intercept
            
            IO_table = pd.DataFrame({'Stim_amp_pA' : stim_array,
                                        "Frequency_Hz" : IO_freq})
            IO_table['Response'] = response
            IO_table=IO_table.loc[IO_table['Frequency_Hz']>=-10.,:]
            
            if not np.isnan(sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Stimulus'].values[0]):
                Saturation_stimulus =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Stimulus'].values[0]
                Saturation_freq =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Frequency'].values[0]
                Saturation_table = pd.DataFrame({'Stim_amp_pA' : [Saturation_stimulus],
                                            "Frequency_Hz" : [Saturation_freq],
                                            'Response' : [response]})
                Saturation_table_list.append(Saturation_table)
        
            model_table_list.append(model_table)
            IO_table_list.append(IO_table)
    
    
    
    Full_stim_freq_table = pd.concat([*stim_freq_table_list],axis=0,ignore_index=True)
    Full_model_table = pd.concat([*model_table_list],axis=0,ignore_index=True)
    Full_IO_table = pd.concat([*IO_table_list],axis=0,ignore_index=True)
    Full_IO_table['Response'] = Full_IO_table['Response'].astype('category')
    if len(Saturation_table_list)!=0:
        Full_Saturation_table = pd.concat([*Saturation_table_list],axis=0,ignore_index=True)
    else:
        Full_Saturation_table=pd.DataFrame()
    
        
        
    if Full_stim_freq_table.shape[0]>0:
        color_keys = Full_stim_freq_table.loc[:,'Response'].unique()
        color_values = get_color_gradient("#f3e79b", "#5d53a5", len(color_keys))
        color_dict = {color_keys[i]: color_values[i] for i in range(len(color_keys))}
        scatter_plot =  px.scatter(Full_stim_freq_table,x='Stim_amp_pA', y='Frequency_Hz',color="Response",
                                   color_discrete_map=color_dict)
        line_plot =  px.line(Full_model_table,x='Stim_amp_pA', y='Frequency_Hz',color="Response",
                                   color_discrete_map=color_dict)
        IO_plot = px.line(Full_IO_table,x='Stim_amp_pA', y='Frequency_Hz',color="Response",
                                   color_discrete_map=color_dict,line_dash="Response")
        IO_plot.update_traces(line=dict(dash='dash'))

        if Full_Saturation_table.shape[0]>0:
            Sat_plot = px.line(Full_Saturation_table,x='Stim_amp_pA', y='Frequency_Hz',color="Response",
                                       color_discrete_map=color_dict)
            IO_figure = go.Figure(data=scatter_plot.data + line_plot.data + IO_plot.data + Sat_plot.data)
        else:
            IO_figure = go.Figure(data=scatter_plot.data + line_plot.data + IO_plot.data)
            
        # Loop through all traces and hide the legend for duplicate entries
        seen_labels = set()
        for trace in IO_figure.data:
            if trace.name in seen_labels:
                trace.showlegend = False  # Hide legend if already seen
            else:
                seen_labels.add(trace.name)  # Mark the label as seen
                
        IO_figure.update_layout(
            autosize=False,
            width=1200,
            height=800,
            xaxis_title="Input Current (pA)", 
            yaxis_title="Frequency (Hz)",
            template="plotly_white")

        return IO_figure

def get_feature_regression_plot_tables(cell_dict,response_type,feature):
    
    
    cell_feature_table = cell_dict['Cell_feature_table']


    response_type=response_type.replace(' ','_')
    
    sub_cell_feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == response_type,[feature,'Output_Duration']]
    sub_cell_feature_table = sub_cell_feature_table.dropna(subset=[feature,'Output_Duration'])
    response_type_legend_dict = {'Time_based':'ms',
                                 'Index_based':'Spike Index',
                                 'Interval_based':'Spike Interval'}
    
   
        
    if sub_cell_feature_table.shape[0]>=2:

        a,b = fir_an.linear_fit(np.array(sub_cell_feature_table.loc[:,'Output_Duration']),np.array(sub_cell_feature_table.loc[:,feature]))

        xgrid = np.linspace(start=np.nanmin(sub_cell_feature_table.loc[:,'Output_Duration']), stop=np.nanmax(sub_cell_feature_table.loc[:,'Output_Duration']), num=30)
        ygrid_extended = b + a * xgrid
        
        
        
        figure = go.FigureWidget(
            data=[
                go.Scattergl(
                    x=np.array(sub_cell_feature_table.loc[:,'Output_Duration']),
                    y=np.array(sub_cell_feature_table.loc[:,feature]),
                    mode="markers",
                    marker=dict(color="rgba(0, 0, 0, 1)", size=5),
                    ),
                go.Scattergl(
                    x=xgrid,
                    y=ygrid_extended,
                    mode="lines",
                    line=dict(color="red", width=2),
                    ),
                ],
            layout={"showlegend": False},
            )
        
        ygrid_original = b + a * np.array(sub_cell_feature_table.loc[:,feature])
        
        r2 = r2_score(np.array(sub_cell_feature_table.loc[:,feature]), ygrid_original)
        
        
        table = pd.DataFrame([np.nanmax(np.array(sub_cell_feature_table.loc[:,feature])),
                              np.nanmin(np.array(sub_cell_feature_table.loc[:,feature])),
                              a,
                              b,
                              r2])
        table=pd.concat([pd.DataFrame(['Maximum','Minimum','Slope','Intercept','R2']),table], axis=1)
    
    
        
        
    else:
        figure = go.FigureWidget(
            data=[
                go.Scattergl(
                    x=np.array(sub_cell_feature_table.loc[:,'Output_Duration']),
                    y=np.array(sub_cell_feature_table.loc[:,feature]),
                    mode="markers",
                    marker=dict(color="rgba(0, 0, 0, 1)", size=5),
                    ),
                ],
            layout={"showlegend": False},
            )
        
        if sub_cell_feature_table.shape[0]>0:
            table = pd.DataFrame([np.nanmax(np.array(sub_cell_feature_table.loc[:,feature])),
                                  np.nanmin(np.array(sub_cell_feature_table.loc[:,feature]))])
        else:
            table=pd.DataFrame([np.nan,np.nan])
        table=pd.concat([pd.DataFrame(['Maximum','Minimum']),table], axis=1)
    
    figure.update_layout(
        autosize=False,
        width=300,
        height=400,
        xaxis_title=response_type_legend_dict[response_type], 
        yaxis_title=feature)
    
    return figure,table






def server(input: Inputs, output: Outputs, session: Session):
    

    
    print("Hello World!")
    
    
    ################# Create JSON configuration file
    
    
    database_list = reactive.Value([])
    db_done = reactive.Value([])
    # Check for the first directory
    @output
    @render.ui
    @reactive.event(input.check_dir_1)
    def status_1():
        full_path = input.json_file_path()
        if not full_path:
            return ui.HTML("<b>No path provided for Path 1.</b>")
        
        dir_path, file_name = os.path.split(full_path)
        if os.path.isdir(dir_path):
            if file_name.endswith(".json"):
                if os.path.isfile(full_path):
                    return ui.HTML(
                        f"""<span style='color:green;'><b>Directory exists:</b> `{dir_path}`</span><br>
                        <span style='color:red;'><b>File </b> {file_name} already exists </span>"""
                    )
                else:
                    return ui.HTML(
                        f"""<span style='color:green;'><b>Directory exists:</b> {dir_path}</span><br>
                        <b>File will be saved as:</b> {file_name}"""
                    )
            else:
                return ui.HTML(
                    f"""<span style='color:green;'><b>Directory exists:</b> {dir_path}</span><br>
                    <span style='color:red;'><b>Warning:</b> {file_name} does not have a .json extension.</span>
                    """
                )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory does not exist:</b> {dir_path}</span>"""
            )

    # Check for the second directory
    @output
    @render.ui
    @reactive.event(input.check_dir_2)
    def status_2():
        full_path = input.Saving_folder()
        if not full_path:
            return ui.HTML("<b>No path provided for Path 2.</b>")
        

        if os.path.isdir(full_path):
            return ui.HTML(
                f"""<span style='color:green;'><b>Directory exists:</b> {full_path}</span><br>"""
            )
            
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory does not exist:</b> {full_path}</span>"""
            )
        
    # Check for the second directory
    @output
    @render.ui
    @reactive.event(input.check_QC_file_path)
    def status_QC_file():
        
        full_path = input.Path_to_QC_file()
        if not full_path:
            return ui.HTML("<b>No path provided for Quality Criteria file.</b>")
        
        dir_path, file_name = os.path.split(full_path)
        if os.path.isfile(full_path):
            if file_name.endswith(".py"): 
                return ui.HTML(
                    f"""<span style='color:green;'><b>Quality Criteria file:</b> `{full_path}` exists </span>"""
                )
            else:
                return ui.HTML(
                    f"""<span style='color:red;'><b>Warning:</b> {full_path} is not a python file.</span>
                    """
                )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>File :</b> {full_path} does not exist</span>"""
            )


    
    @output
    @render.ui
    @reactive.event(input.Check_original_file_path)
    def status_original_file_path():
        original_file_path = input.original_file_path()
        if not original_file_path:
            return ui.HTML("<b>No path provided for original file path.</b>")
        

        if os.path.isdir(original_file_path):
            return ui.HTML(
                f"""<span style='color:green;'><b>Directory exists:</b> `{original_file_path}`</span>
               """
            )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory does not exist:</b> {original_file_path}</span>"""
            )
        
    @output
    @render.ui
    @reactive.event(input.check_database_python_script_path)
    def status_database_script_file():
        database_script_file = input.database_python_script_path()

        if not database_script_file:
            return ui.HTML("<b>No path provided for original file path.</b>")
    
        if os.path.isfile(database_script_file):
            # When the directory exists, display a success message and a selectInput
            
            
            return ui.HTML(f"""<span style='color:green;'><b>File: </b> `{database_script_file}` exists</span><br>""")
            
        else:
            # When the directory doesn't exist, display an error message
            return ui.HTML(
                f"""<span style='color:red;'><b>File {database_script_file} does not exist:</span>"""
            )
    
    @output
    @render.ui
    @reactive.event(input.check_database_python_script_path)
    def import_database_script_file():
        original_file_path = input.database_python_script_path()
        
        if not original_file_path:
            return ui.HTML("<b>No path provided for original file path.</b>")
    
        if os.path.isfile(original_file_path):
            # When the directory exists, display a success message and a selectInput
            
            with open(original_file_path, "r") as file:
                tree = ast.parse(file.read(), filename=original_file_path)
            
                # Extract function names
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            return ui.input_select(
                "select_database_function", "Choose database's function", 
                choices=functions  # Replace with dynamic options if needed
            )
    
    # Check for the first directory
    @output
    @render.ui
    @reactive.event(input.check_population_class_file_path)
    def status_population_class_file():
        full_path = input.population_class_file()
        if not full_path:
            return ui.HTML("<b>No path provided for Population class file.</b>")
        
        dir_path, file_name = os.path.split(full_path)
        if os.path.isfile(full_path):
            if file_name.endswith(".csv"): 
                return ui.HTML(
                    f"""<span style='color:green;'><b>Population class file:</b> `{full_path}` exists </span>"""
                )
            else:
                return ui.HTML(
                    f"""<span style='color:red;'><b>Warning:</b> {full_path} is not a csv.</span>
                    """
                )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>File :</b> {full_path} does not exist</span>"""
            )
        
    # Check for the first directory
    @output
    @render.ui
    @reactive.event(input.check_cell_sweep_table_file_path)
    def status_cell_sweep_table_file():
        full_path = input.cell_sweep_table_file()
        if not full_path:
            return ui.HTML("<b>No path provided for Cell Sweep table file.</b>")
        
        dir_path, file_name = os.path.split(full_path)
        if os.path.isfile(full_path):
            if file_name.endswith(".csv"): 
                return ui.HTML(
                    f"""<span style='color:green;'><b>Cell Sweep table file:</b> `{full_path}` exists </span>"""
                )
            else:
                return ui.HTML(
                    f"""<span style='color:red;'><b>Warning:</b> {full_path} is not a csv.</span>
                    """
                )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>File :</b> {full_path} does not exist</span>"""
            )
        
        
    # Add Database button action
    @output
    @render.ui
    @reactive.event(input.Add_Database)
    def add_database():
        
        
        database_name = input.database_name()
        if not database_name : 
            return ui.HTML("<span style='color:red;'><b>Database Name must be provided</b></span>")
    
        original_file_path = input.original_file_path()
        if not original_file_path : 
            return ui.HTML("<span style='color:red;'><b>Folder of original files must be provided</b></span>")
        if not original_file_path.endswith(os.path.sep):
            original_file_path += os.path.sep
        
        path_to_db_script_folder = input.database_python_script_path()
        if not path_to_db_script_folder : 
            return ui.HTML("<span style='color:red;'><b>Path to database script must be provided</b></span>")
        
        database_script_path, database_script_file_name = os.path.split(path_to_db_script_folder)
        if not database_script_path.endswith(os.path.sep):
            database_script_path += os.path.sep
        database_function = input.select_database_function()
        if not database_function : 
            return ui.HTML("<span style='color:red;'><b>Function name for database must be provided</b></span>")
        
        
        database_population_class_file = input.population_class_file()
        if not database_population_class_file : 
            return ui.HTML("<span style='color:red;'><b>Population class file must be provided</b></span>")
        
        database_cell_sweep_table_file = input.cell_sweep_table_file()
        if not database_cell_sweep_table_file : 
            return ui.HTML("<span style='color:red;'><b>Cell Sweep table must be provided</b></span>")
        
        stimulus_provided = input.stimulus_time_provided()
        stimulus_duration = input.stimulus_duration()
        


        # Store in the reactive dictionary
        current_list = database_list()
        current_db_done = db_done()
        for i, d in enumerate(current_list):
            if d.get("database_name") == database_name:
                del current_list[i]
                break  # Exit loop after removing the first match
                
        
        db_list = {
            "database_name" : database_name,
            "original_file_directory" :original_file_path,
            "path_to_db_script_folder" : database_script_path,
            "python_file_name" :database_script_file_name,
            'db_function_name':database_function,
            "db_population_class_file" : database_population_class_file,
            'db_cell_sweep_csv_file':database_cell_sweep_table_file,
            "db_stimulus_duration" : float(stimulus_duration),
            "stimulus_time_provided": stimulus_provided}
        current_list.append(db_list)
        current_db_done.append(database_name)
        
        
        database_list.set(current_list)
        db_done.set(current_db_done)

        # Reset inputs manually by setting them to empty strings
        ui.update_text("database_name", value="")
        ui.update_text("original_file_path", value="")
        ui.update_text('database_python_script_path', value = "")

        ui.update_text("population_class_file", value="")
        ui.update_text("cell_sweep_table_file", value="")
        ui.update_numeric("stimulus_duration", value=0.5)

        # Display the updated dictionary
        return ui.HTML(f"<b>Updated Database Dictionary:</b><br><pre>{json.dumps(current_list, indent=4)}</pre>")
    
        
    # Save JSON file if inputs are valid
    @output
    @render.ui
    @reactive.event(input.save_json)
    def save_json_status():
        
        DB_parameters = database_list()
        json_file_path = input.json_file_path()
        
        if not json_file_path : 
            return ui.HTML("<span style='color:red;'><b>Saving file for JSON config file must be provided</b></span>")
        
        
        Saving_folder = input.Saving_folder()
        
        if not Saving_folder : 
            return ui.HTML("<span style='color:red;'><b>Saving folder for analysis must be provided</b></span>")
        
        Path_to_QC_file = input.Path_to_QC_file()
        if not Path_to_QC_file : 
            return ui.HTML("<span style='color:red;'><b>Quality Criteria Python Script must be provided</b></span>")
        
        if not Saving_folder.endswith(os.path.sep):
            Saving_folder += os.path.sep
        
        configuration_file = {
            
            'path_to_saving_file' :Saving_folder,
            'path_to_QC_file' : Path_to_QC_file,
            'DB_parameters':DB_parameters
            }
        
        # Serializing json
        configuration_file_json = json.dumps(configuration_file, indent=4)
         
        # Writing to sample.json
        with open(json_file_path, "w") as outfile:
            outfile.write(configuration_file_json)
        
        return ui.HTML(f"<span style='color:green;'><b>JSON file saved successfully to:</b> {json_file_path}</span>")
        
    ################## Run Analysis
    
        
    @output
    @render.text
    @reactive.event(input.run_analysis)
    def script_output():
        # Get inputs
        problem_cell = []

        if not input.json_file_input():
            return "No file uploaded."
        else:
            json_config_file = input.json_file_input()
            
            config_df = import_json_config_file(json_config_file)
            
            nb_of_workers_to_use = int(input.n_CPU())
            if nb_of_workers_to_use == 0:
                nb_of_workers_to_use = 1
            
            analysis_to_perform = input.select_analysis()
            overwrite_cell_files_yes_no = input.overwrite_existing_files()
            if overwrite_cell_files_yes_no == 'Yes':
                overwrite_cell_files = True
            elif overwrite_cell_files_yes_no == 'No':
                overwrite_cell_files = False
            
            path_to_saving_file = config_df.loc[0,'path_to_saving_file']
            path_to_QC_file = config_df.loc[0,'path_to_QC_file']
            
            for elt in config_df.index:
                
                current_db = config_df.loc[elt,:].to_dict()
                database_name = current_db["database_name"]
                path_to_python_folder = current_db["path_to_db_script_folder"]
                
                python_file=current_db['python_file_name']
                module=python_file.replace('.py',"")
                full_path_to_python_script=str(path_to_python_folder+python_file)
                
                spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)
                DB_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(DB_module)
                
                db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
                
                cell_id_list = db_cell_sweep_file['Cell_id'].unique()
                random.shuffle(cell_id_list)
                
                args_list = [[x,
                              current_db,
                              module,
                              full_path_to_python_script,
                              path_to_QC_file, 
                              path_to_saving_file,
                              overwrite_cell_files,
                              analysis_to_perform] for x in cell_id_list]
                
                
                start_time = time.time()
                with ui.Progress(min=0, max=len(cell_id_list)) as progress:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_of_workers_to_use) as executor:
                        problem_cell_list = {executor.submit(analysis_pipeline.cell_processing, x): x for x in args_list}
                        i=1
                        for f in concurrent.futures.as_completed(problem_cell_list):
                            cell_id_problem = f.result()
                            if cell_id_problem is not None:
                                problem_cell.append(cell_id_problem)
                            
                            
                            
                            formatted_time, remaining_formatted_time = estimate_processing_time(start_time, len(cell_id_list), i)

                            
                            
                            progress.set(i, message=f"Processing {database_name}:", detail=f" {i}/{len(cell_id_list)};  Time spent: {formatted_time}, Estimated remaining time:{remaining_formatted_time}")
                            i+=1
                            


                print(f'Database {database_name} processed')
            
            print('Done')
            return f'Problem occured with cells : {problem_cell}'
    
    @output
    @render.ui
    @reactive.event(input.check_saving_summaryfolder)
    def summary_folder_status():
        full_path = input.summary_folder_path()
        if not full_path:
            return ui.HTML("<b>No path provided for Path 1.</b>")
        
        
        if os.path.isdir(full_path):
            return ui.HTML(
                f"""<span style='color:green;'><b>Directory :</b> `{full_path}` exists</span>"""
            )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory :</b> `{full_path}` Doesn't exist</span>"""
            )
            
    
    @output
    @render.text
    @reactive.event(input.summarise_analysis)
    def summarize_analysis_output():
        if not input.json_file_input():
            return "No file uploaded."
        else:
            json_config_file = input.json_file_input()
            
            config_json_file = import_json_config_file(json_config_file)
            
        saving_path = input.summary_folder_path()
        if not saving_path.endswith(os.path.sep):
            saving_path += os.path.sep

        unit_line=pd.DataFrame(['--','--','--','Hz/pA','pA','Hz','pA','Hz', "pA"]).T
        Full_feature_table=pd.DataFrame(columns=['Cell_id','Obs','I_O_NRMSE','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Response_Fail_Frequency','Response_Fail_Stimulus','Response_type',"Output_Duration"])
        
        unit_fit_line = pd.DataFrame(['--','--','--','--','--','--', "--",'--']).T
        Full_fit_table=pd.DataFrame(columns=['Cell_id','Obs','Hill_amplitude','Hill_coef', 'Hill_Half_cst','Hill_x0','Sigmoid_x0','Sigmoid_k','Response_type',"Output_Duration"])

        cell_linear_values = pd.DataFrame(columns=['Cell_id','Input_Resistance_GOhms','Input_Resistance_GOhms_SD','Time_constant_ms','Time_constant_ms_SD', "Resting_potential_mV", 'Resting_potential_mV_SD'])
        linear_values_unit = pd.DataFrame(['--','GOhms','GOhms','ms','ms','mV','mV']).T
        linear_values_unit.columns=['Cell_id','Input_Resistance_GOhms','Input_Resistance_GOhms_SD','Time_constant_ms','Time_constant_ms_SD', "Resting_potential_mV", 'Resting_potential_mV_SD']
        cell_linear_values = pd.concat([cell_linear_values,linear_values_unit],ignore_index=True)
        
        processing_time_table = pd.DataFrame(columns=['Cell_id','Processing_step','Processing_time'])
        processing_time_unit_line = pd.DataFrame(['--','--','s']).T
        processing_time_unit_line.columns=processing_time_table.columns
        processing_time_table = pd.concat([processing_time_table,processing_time_unit_line],ignore_index=True)
        
        Adaptation_table = pd.DataFrame(columns = ["Cell_id", "Adaptation_Obs", "Adaptation_Instantaneous_Frequency_Hz",
                                                   'Adaptation_Spike_width_at_half_heigth_s',
                                                   "Adaptation_Spike_heigth_mV", 
                                                   "Adaptation_Threshold_mV",
                                                   'Adaptation_Upstroke_mV/s',
                                                   "Adaptation_Peak_mV", 
                                                   'Adaptation_Downstroke_mV/s',
                                                   "Adaptation_Fast_Trough_mV",
                                                   'Adaptation_fAHP_mV',
                                                   'Adaptation_Trough_mV'])
        Adaptation_table_unit_line = pd.DataFrame(["--","--","Index","Index","Index","Index","Index","Index","Index","Index","Index","Index"]).T
        Adaptation_table_unit_line.columns = Adaptation_table.columns
        Adaptation_table = pd.concat([Adaptation_table, Adaptation_table_unit_line],ignore_index = True)
        problem_cell=[]
        problem_df = pd.DataFrame(columns = ['Cell_id','Error_message'])
        
        Full_population_calss_table = pd.DataFrame()
        
        for line in config_json_file.index:
            current_db_population_class_table = pd.read_csv(config_json_file.loc[line,'db_population_class_file'])
            Full_population_calss_table= pd.concat([Full_population_calss_table,current_db_population_class_table],ignore_index = True)
        Full_population_calss_table=Full_population_calss_table.astype({'Cell_id':'str'})
        cell_id_list = Full_population_calss_table.loc[:,'Cell_id'].unique()
        
        i=1
        start_time = time.time()
        with ui.Progress(min=0, max=len(cell_id_list)) as progress:
            for cell_id in cell_id_list:
                try:
                    current_DB = Full_population_calss_table.loc[Full_population_calss_table['Cell_id']==cell_id,'Database'].values[0]
                    config_line = config_json_file.loc[config_json_file['database_name']==current_DB,:]
                    cell_dict = ordifunc.read_cell_file_h5(str(cell_id),config_line,['Sweep analysis','Firing analysis','Processing_report', "Sweep QC"])
                    sweep_info_table = cell_dict['Sweep_info_table']
                    cell_fit_table = cell_dict['Cell_fit_table']
                    cell_feature_table = cell_dict['Cell_feature_table']
                    Processing_df = cell_dict['Processing_table']
                    cell_adaptation_table = cell_dict['Cell_Adaptation']
                    sweep_QC_table = cell_dict['Sweep_QC_table']
                    sweep_info_QC_table = pd.merge(sweep_info_table, sweep_QC_table.loc[:,['Passed_QC', "Sweep"]], on = "Sweep")
                    sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True,:]
                    sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
                    
                    sub_sweep_info_QC_table_SS_potential = sub_sweep_info_QC_table.dropna(subset=['SS_potential_mV'])
                    SS_potential_mV_list = list(sub_sweep_info_QC_table_SS_potential.loc[:,'SS_potential_mV'])
                    Stim_amp_pA_list = list(sub_sweep_info_QC_table_SS_potential.loc[:,'Stim_SS_pA'])
                    

                    IR_regression_fit = ordifunc.compute_cell_input_resistance(cell_dict)
                    Cell_IR = IR_regression_fit[0]
                    

                    IR_SD = np.nanstd(sub_sweep_info_QC_table['Input_Resistance_GOhms'])
                    if Cell_IR <=0:
                        Cell_IR = np.nan
                        IR_SD = np.nan
                   
                    Time_cst_mean, Time_cst_SD = ordifunc.compute_cell_time_constant(cell_dict)
                    
                    
                    
                    Resting_potential_mean, Resting_potential_SD = ordifunc.compute_cell_resting_potential(cell_dict)
                    
                    
                    cell_linear_values_line = pd.DataFrame([str(cell_id),Cell_IR,IR_SD,Time_cst_mean,Time_cst_SD,Resting_potential_mean, Resting_potential_SD ]).T
                    cell_linear_values_line.columns = ['Cell_id','Input_Resistance_GOhms','Input_Resistance_GOhms_SD','Time_constant_ms','Time_constant_ms_SD', "Resting_potential_mV", 'Resting_potential_mV_SD']
                    cell_linear_values=pd.concat([cell_linear_values,cell_linear_values_line],ignore_index=True)
                    
                    response_duration_dictionnary={
                        'Time_based':[.005, .010, .025, .050, .100, .250, .500],
                        'Index_based':list(np.arange(2,18)),
                        'Interval_based':list(np.arange(1,17))}

                    if cell_fit_table.shape[0]==1:

                        for response_type in response_duration_dictionnary.keys():
                            output_duration_list=response_duration_dictionnary[response_type]
                            for output_duration in output_duration_list:
                                I_O_obs=cell_fit_table.loc[(cell_fit_table['Response_type']==response_type )& (cell_fit_table['Output_Duration']==output_duration),"I_O_obs"]
                                if len(I_O_obs)!=0:
                                    Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Response_Failure_Frequency,Response_Failure_Stimulus = np.array(cell_feature_table.loc[(cell_feature_table['Response_type']==response_type )&
                                                                                                                                               (cell_feature_table['Output_Duration']==output_duration),
                                                                                                                                        ["Gain","Threshold","Saturation_Frequency","Saturation_Stimulus","Response_Fail_Frequency", "Response_Fail_Stimulus"]]).tolist()[0]
                                    I_O_obs=I_O_obs.tolist()[0]
                                    
                                    Hill_Half_cst, Hill_amplitude, Hill_coef, Hill_x0, Output_Duration, Response_type, Sigmoid_k,Sigmoid_x0 = np.array(cell_fit_table.loc[(cell_fit_table['Response_type']==response_type )&
                                                                                                                                               (cell_fit_table['Output_Duration']==output_duration),
                                                                                                                                        ["Hill_Half_cst", "Hill_amplitude", "Hill_coef", "Hill_x0", "Output_Duration", "Response_type", "Sigmoid_k","Sigmoid_x0"]]).tolist()[0]
                                else:
                                    I_O_obs="No_I_O_Adapt_computed"
                                    empty_array = np.empty(6)
                                    empty_array[:] = np.nan
                                    Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Response_Failure_Frequency,Response_Failure_Stimulus = empty_array
                                    
                                    empty_array = np.empty(8)
                                    empty_array[:] = np.nan
                                    Hill_Half_cst, Hill_amplitude, Hill_coef, Hill_x0, Output_Duration, Response_type, Sigmoid_k,Sigmoid_x0 = empty_array
                                
                                    
                                if Gain<=0:
                                    Gain=np.nan

                                    
                                    
                                new_line=pd.DataFrame([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,response_type,output_duration]).T
                                new_line.columns=Full_feature_table.columns
                                Full_feature_table = pd.concat([Full_feature_table,new_line],ignore_index=True)
                                
                                new_fit_line = pd.DataFrame([str(cell_id),I_O_obs,Hill_amplitude, Hill_coef, Hill_Half_cst, Hill_x0, Sigmoid_x0, Sigmoid_k, response_type, output_duration]).T
                                new_fit_line.columns = Full_fit_table.columns
                                Full_fit_table = pd.concat([Full_fit_table, new_fit_line], ignore_table = True)
                                
                                
                    else:
                        
                        cell_feature_table = cell_feature_table.merge(
                    cell_fit_table.loc[:,['I_O_obs','Response_type','Output_Duration','I_O_NRMSE']], how='inner', on=['Response_type','Output_Duration'])
            
                        cell_feature_table['Cell_id']=str(cell_id)
                        cell_feature_table=cell_feature_table.rename(columns={"I_O_obs": "Obs"})
                        cell_feature_table=cell_feature_table.reindex(columns=Full_feature_table.columns)
                        Full_feature_table = pd.concat([Full_feature_table,cell_feature_table],ignore_index=True)
                        
                        cell_fit_table['Cell_id'] = str(cell_id)
                        cell_fit_table=cell_fit_table.rename(columns={"I_O_obs": "Obs"})
                        cell_fit_table=cell_fit_table.reindex(columns=Full_fit_table.columns)
                        Full_fit_table = pd.concat([Full_fit_table, cell_fit_table], ignore_index = True)
                        
                    
                    cell_adaptation_table_copy = cell_adaptation_table.copy()
                    cell_adaptation_table_copy['Feature'] = cell_adaptation_table_copy.apply(lambda row: ordifunc.append_last_measure(row['Feature'], row['Measure']), axis=1)
                    feature_dict = cell_adaptation_table_copy.set_index('Feature')['Adaptation_Index'].to_dict()
                    
                    result_df = pd.DataFrame([feature_dict])
                    result_df = result_df.add_prefix('Adaptation_')
                    result_df.loc[:,'Cell_id']=cell_id
                    result_df.loc[:,'Obs']="--"
                    result_df = result_df.rename(columns={'Adaptation_Instantaneous_Frequency_mV':"Adaptation_Instantaneous_Frequency_Hz", 
                                                          "Obs":"Adaptation_Obs"})

                    Adaptation_table = pd.concat([Adaptation_table, result_df], ignore_index=True)
                    

                    for step in Processing_df.loc[:,'Processing_step'].unique():
                        sub_processing_table = Processing_df.loc[Processing_df['Processing_step']==step,:]
                        sub_processing_table=sub_processing_table.reset_index()
                        sub_processing_time = sub_processing_table.loc[0,"Processing_time"]
                        sub_processing_time = float(sub_processing_time.replace("s",""))
                        new_line = pd.DataFrame([cell_id,step,sub_processing_time]).T
                        
                        new_line.columns = processing_time_table.columns
                        processing_time_table=pd.concat([processing_time_table,new_line],ignore_index=True,axis=0)
                except:

                    try:
                        cell_dict = ordifunc.read_cell_file_h5(str(cell_id),config_line,['Processing_report'])
                        
                        Processing_df = cell_dict['Processing_table']
                        for elt in Processing_df.index:
                            if "Error" in Processing_df.loc[elt,'Warnings_encontered'] :
                                new_line = pd.DataFrame([cell_id, Processing_df.loc[elt,'Warnings_encontered']]).T
                                new_line.columns = problem_df.columns
                                problem_df = pd.concat([problem_df, new_line], ignore_index=True)
                                break
                    except:

                        problem_cell.append(cell_id)
                
                formatted_time, remaining_formatted_time = estimate_processing_time(start_time, len(cell_id_list), i)
                progress.set(i, message="Processing ", detail=f" {i}/{len(cell_id_list)}; Time spent: {formatted_time}, Estimated remaining time:{remaining_formatted_time}")
                i+=1
                
            for response_type in response_duration_dictionnary.keys():
                output_duration_list=response_duration_dictionnary[response_type]
                for output_duration in output_duration_list:
                    
                    new_table = pd.DataFrame(columns=['Cell_id','Obs','I_O_NRMSE','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Response_Fail_Frequency','Response_Fail_Stimulus',])
                    unit_line.columns=new_table.columns
                    new_table = pd.concat([new_table,unit_line],ignore_index=True)
                    
                    sub_table=Full_feature_table.loc[(Full_feature_table['Response_type']==response_type)&(Full_feature_table['Output_Duration']==output_duration),]
                    sub_table=sub_table.drop(['Response_type','Output_Duration'], axis=1)
                    sub_table=sub_table.reindex(columns=new_table.columns)
                    sub_table['Gain'] = sub_table['Gain'].where(sub_table['Gain'] >= 0, np.nan)
                    
                    new_table = pd.concat([new_table,sub_table],ignore_index=True)
                    if len(new_table['Cell_id'].unique())!=new_table.shape[0]:
                        return str('problem with table'+str(response_type)+'_'+str(output_duration))
                    
                    
                    
                    new_fit_table = pd.DataFrame(columns = ['Cell_id','Obs','Hill_amplitude','Hill_coef', 'Hill_Half_cst','Hill_x0','Sigmoid_x0','Sigmoid_k'])
                    unit_fit_line.columns = new_fit_table.columns
                    new_fit_table = pd.concat([new_fit_table, unit_fit_line], ignore_index = True)
                    
                    sub_fit_table = Full_fit_table.loc[(Full_fit_table['Response_type']==response_type)&(Full_fit_table['Output_Duration']==output_duration),]
                    sub_fit_table = sub_fit_table.drop(['Response_type','Output_Duration'], axis=1)
                    sub_fit_table = sub_fit_table.reindex(columns=new_fit_table.columns)
                    new_fit_table = pd.concat([new_fit_table,sub_fit_table],ignore_index=True)
                    
                    
                    if response_type == 'Time_based':
                        output_duration*=1000
                        output_duration=str(str(int(output_duration))+'ms')
                    

                    new_table.to_csv(f"{saving_path}Full_Feature_Table_{response_type}_{output_duration}.csv")
                    
                    
                    if len(new_fit_table['Cell_id'].unique())!=new_fit_table.shape[0]:
                        return str('problem with table'+str(response_type)+'_'+str(output_duration))
                    
                    new_fit_table.to_csv(f"{saving_path}Full_Fit_Table_{response_type}_{output_duration}.csv")
                        
            cell_linear_values.to_csv(str(saving_path+ 'Full_Cell_linear_values.csv')) 
            
            processing_time_table.to_csv(str(saving_path+'Full_Processing_Times.csv'))
            Adaptation_table.to_csv(f'{saving_path}Full_Adaptation_Table_Time_based_500ms.csv')
            problem_df.to_csv(f'{saving_path}Problem_report.csv')
            for col in Full_population_calss_table.columns:
                if "Unnamed" in col:
                    Full_population_calss_table = Full_population_calss_table.drop(columns=[col])
            Full_population_calss_table.to_csv(f'{saving_path}Full_Population_Class.csv')
            print("Done")
            return  f'Analysis Summary done. Problem with cells {problem_cell}'
    
    
    ################## Cell Visualization 
    @reactive.Effect
    @reactive.event(input.JSON_config_file)
    def _():
        
        config_table = get_config_json_table()
        saving_folder = config_table.loc[0,"path_to_saving_file"]
        file_list = os.listdir(saving_folder)
        file_df = pd.DataFrame(file_list,columns=['Cell_file'])
        file_df['Cell_id']='--'
        for line in range(file_df.shape[0]):
            cell_id = file_df.loc[line,'Cell_file']
            cell_id = cell_id.replace('Cell_','')
            cell_id = cell_id.replace('.h5','')
            file_df.loc[line,'Cell_id']=cell_id
            
        cell_file_dict = dict(zip(list(file_df.loc[:,'Cell_file']),list(file_df.loc[:,'Cell_id'])))

        ui.update_selectize("Cell_file_select" ,choices=cell_file_dict)
        
    @reactive.Effect
    @reactive.event(input.Update_cell_btn)
    def sweep_list():
        
        if input.JSON_config_file() and input.Cell_file_select() :
            cells_dict = get_cell_tables()
            Sweep_info_table = cells_dict["Sweep_info_table"]
            #Sweep_info_table = cells[4]
            sweep_list = list(Sweep_info_table.loc[:,"Sweep"])
            ui.update_selectize("Sweep_selected" ,choices=sweep_list)
            ui.update_selectize("Sweep_selected_in_progress" ,choices=sweep_list)
            cell_fit_table = cells_dict["Cell_fit_table"]
            #cell_fit_table = cells[6]
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == 'Time_based',:]
            time_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())

            ui.update_select('Firing_output_duration_Time_based', choices=time_output_duration)
            
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == 'Index_based',:]
            index_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
            ui.update_select('Firing_output_duration_Index_based', choices=index_output_duration)
            
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == 'Interval_based',:]
            interval_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
            ui.update_select('Firing_output_duration_Interval_based', choices=interval_output_duration)
            
            
            Adaptation_table = cells_dict['Cell_Adaptation']
            Sub_Adatation_table = Adaptation_table.loc[(Adaptation_table['Obs'] == '--') & (Adaptation_table['Adaptation_Index'].notna()),:]
            validated_features = list(Sub_Adatation_table.loc[:,'Feature'])
            ui.update_select('Adaptation_feature_to_display', choices=validated_features)
            
    @reactive.Effect
    @reactive.event(input.Update_cell_btn, input.Firing_response_type_new)
    def update_IO_output_duration_list():
        
        if input.JSON_config_file() and input.Cell_file_select() :
            cell_dict=get_cell_tables()
            cell_fit_table = cell_dict['Cell_fit_table']
            response_type = input.Firing_response_type_new()
            #cell_fit_table = cells[6]
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
            sub_cell_fit_table = sub_cell_fit_table[sub_cell_fit_table["I_O_obs"].isin(['Hill','Hill-Sigmoid'])]
            time_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
            
            if input.Firing_response_type_new()=='Time_based':
                time_output_duration = [float(x) for x in time_output_duration]
                
            else:
                
                time_output_duration = [int(float(x)) for x in time_output_duration]
            
            
            

            
            ui.update_select(
                "Firing_output_duration_new",
                label="Select output duration",
                choices=time_output_duration,
                
            )
            
            
    @reactive.Effect
    @reactive.event(input.Update_cell_btn, input.Firing_output_duration_new)
    def update_IO_plot_to_display_list():
        
        if input.JSON_config_file() and input.Cell_file_select() and input.Firing_output_duration_new():
            
            cell_dict=get_cell_tables()
            Cell_feature_table = cell_dict['Cell_feature_table']
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            Sweep_QC_table = cell_dict['Sweep_QC_table']
            
            Firing_response_type_new = input.Firing_response_type_new()
            if Firing_response_type_new=='Time_based':
                
                Firing_output_duration_new = float(input.Firing_output_duration_new())
            else:
                Firing_output_duration_new = int(float(input.Firing_output_duration_new()))
            
            
            
            Test_Gain = Cell_feature_table.loc[(Cell_feature_table['Response_type']==Firing_response_type_new)&
                                                            (Cell_feature_table['Output_Duration']==Firing_output_duration_new), 'Gain'].values[0]
            
            
            if not np.isnan(Test_Gain):
                stim_freq_table = fir_an.get_stim_freq_table(Full_SF_table, Sweep_info_table, Sweep_QC_table, Firing_output_duration_new, Firing_response_type_new)
                plot_list = fir_an.get_IO_features_NEW_TEST(stim_freq_table, Firing_response_type_new, Firing_output_duration_new, do_plot=True)
                

                #Update choice input 
                ui.update_selectize('Plot_new',label="Select plot to display", choices=list(plot_list.keys()))
            
            
    
        
    
    @reactive.Calc
    @reactive.event(input.Update_cell_btn)
    def get_cell_tables():

        cell_id = input.Cell_file_select()
        config_json_input = input.JSON_config_file()
        if config_json_input and cell_id :
            config_table = get_config_json_table()
            saving_folder = config_table.loc[0,"path_to_saving_file"]
            cell_file_path = str(saving_folder+str(cell_id))
            
            #Metadata_table = ordifunc.read_cell_file_h5(cell_file_path, "--",selection=["Metadata"])
            Metadata_table = ordifunc.read_cell_file_h5(cell_file_path, "--",selection=["Metadata"])
            
            Database = Metadata_table['Database'].values[0]
            
           
            current_config_line = config_table[config_table['database_name'] == Database]
            cell_id=cell_id.replace('Cell_','')
            cell_id = cell_id.replace('.h5','')

            #Full_TVC_table, Full_SF_dict_table, Full_SF_table, Metadata_table, sweep_info_table, Sweep_QC_table, cell_fit_table, cell_feature_table,Processing_report_df = ordifunc.read_cell_file_h5(cell_id, current_config_line,selection=["All"])
            cell_dict = ordifunc.read_cell_file_h5(cell_id, current_config_line,selection=["All"])
            return cell_dict
            #return Full_TVC_table, Full_SF_dict_table, Full_SF_table, Metadata_table, sweep_info_table, Sweep_QC_table, cell_fit_table, cell_feature_table,Processing_report_df
        
        
    @reactive.Calc
    def get_config_json_table():
        config_json_input=input.JSON_config_file()
        i=0
        
        if config_json_input:
            for file in config_json_input:
                file_path = file['datapath']
                if i == 0:
                    config_df = ordifunc.open_json_config_file(file_path)
                    i+=1
                    
                else:
                    new_table = ordifunc.open_json_config_file(file_path)
                    config_df=pd.concat([config_df, new_table],ignore_index = True)
            return config_df
    @output
    @render.table
    def config_json_table():
        config_table = get_config_json_table()
        
        return config_table
    
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn)
    def cell_processing_time_table():
        if input.JSON_config_file() and input.Cell_file_select() :

                
            cell_dict = get_cell_tables()
            processing_time_table = cell_dict["Processing_table"] 
            #processing_time_table = cells[-1]
            return processing_time_table
        
    @output
    @render.table
    @reactive.event(input.Update_cell_btn)
    def Cell_Metadata_table():
        if input.JSON_config_file() and input.Cell_file_select() :

                
            cell_dict = get_cell_tables()
            Metadata_table = cell_dict['Metadata_table']
            for col in Metadata_table:
                if 'Unnamed' in col:
                    Metadata_table = Metadata_table.drop(columns=[col])
            Metadata_table = Metadata_table.T
            Metadata_table = Metadata_table.reset_index(drop=False)
            Metadata_table.columns = [" Metadata", " "]
            #processing_time_table = cells[-1]
            return Metadata_table
        
    @output
    @render.table
    @reactive.event(input.Update_cell_btn)
    def Cell_linear_properties_table():
        if input.JSON_config_file() and input.Cell_file_select() :

                
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_QC_table = cell_dict["Sweep_QC_table"]
            sweep_info_QC_table = pd.merge(Sweep_info_table, Sweep_QC_table.loc[:, ['Passed_QC', "Sweep"]], on="Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True, :]
            sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
            
            sub_sweep_info_QC_table_SS_potential = sub_sweep_info_QC_table.dropna(subset=['SS_potential_mV'])
            SS_potential_mV_list = list(sub_sweep_info_QC_table_SS_potential['SS_potential_mV'])
            Stim_amp_pA_list = list(sub_sweep_info_QC_table_SS_potential['Stim_SS_pA'])
            
            # Perform linear regression to get Input Resistance and intercept
            #IR_regression_fit = fir_an.linear_fit(Stim_amp_pA_list, SS_potential_mV_list)[0]
            IR_regression_fit = ordifunc.compute_cell_input_resistance(cell_dict)
            Cell_IR = IR_regression_fit[0]
            
            Time_cst_mean, Time_cst_SD = ordifunc.compute_cell_time_constant(cell_dict)
            Capacitance = Time_cst_mean/Cell_IR
            
            Resting_potential_mean, Resting_potential_SD = ordifunc.compute_cell_resting_potential(cell_dict)
            
            
            
            
            Linear_table = pd.DataFrame({'Input Resistance GΩ':[Cell_IR],
                                         'Time constant ms' : [Time_cst_mean],
                                         'Resting membrane potential mV' : [Resting_potential_mean],
                                         'Capacitance pF' : [Capacitance]})
            
            
            Linear_table = Linear_table.T
            Linear_table = Linear_table.reset_index(drop=False)
            Linear_table.columns = [" Linear properties", " "]
            return Linear_table
        
    @output
    @render.table
    @reactive.event(input.Update_cell_btn)
    def Cell_Firing_properties_table():
        if input.JSON_config_file() and input.Cell_file_select() :

                
            cell_dict = get_cell_tables()
            cell_feature_table = cell_dict['Cell_feature_table']
            cell_feature_table = cell_feature_table.loc[:,['Response_type',
                                                           'Output_Duration',
                                                           "Gain", 
                                                           'Threshold',
                                                           'Saturation_Frequency', 
                                                           'Saturation_Stimulus', 
                                                           'Response_Fail_Frequency',
                                                           "Response_Fail_Stimulus"]]
            
            return cell_feature_table
        
    @output
    @render.table
    @reactive.event(input.Update_cell_btn)
    def Cell_Adaptation_table():
        if input.JSON_config_file() and input.Cell_file_select() :

                
            cell_dict = get_cell_tables()
            cell_Adaptation = cell_dict['Cell_Adaptation']
            
            return cell_Adaptation
        
        
        
        
    
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_input_resistance_v_stim_amp_plolty():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_QC_table = cell_dict["Sweep_QC_table"]
            sweep_info_QC_table = pd.merge(Sweep_info_table, Sweep_QC_table.loc[:, ['Passed_QC', "Sweep"]], on="Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True, :]
            sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
            

            
            # Create a scatter plot with color based on 'Protocol_id'
            scatter_plot = px.scatter(
                                sub_sweep_info_QC_table,
                                x='Stim_amp_pA', 
                                y='Input_Resistance_GOhms',
                                color='Protocol_id',  # Colors the points based on 'Protocol_id'
                                labels={'Stim_amp_pA': 'Stimulus Amplitude (pA)', 
                                        'Input_Resistance_GOhms': 'Input Resistance (GOhms)',
                                        'Protocol_id': 'Protocol'},  # Renames the legend title
                                hover_data={'Sweep': True}  # Adds 'Sweep' to hover data
                                    )
            
            # Create a Plotly figure and update layout
            figure = go.Figure(data=scatter_plot.data)
            figure.update_layout(
                autosize=True,
                xaxis_title="Stimulus amplitude (pA)", 
                yaxis_title="Input Resistance (GOhms)",
                template="plotly_white"
            )
            
            return figure
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_time_constant_v_stim_amp_plolty():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_QC_table = cell_dict["Sweep_QC_table"]
            sweep_info_QC_table = pd.merge(Sweep_info_table, Sweep_QC_table.loc[:, ['Passed_QC', "Sweep"]], on="Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True, :]
            sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
            # Create a scatter plot with color based on 'Protocol_id'
            scatter_plot = px.scatter(
                                sub_sweep_info_QC_table,
                                x='Stim_amp_pA', 
                                y='Time_constant_ms',
                                color='Protocol_id',  # Colors the points based on 'Protocol_id'
                                labels={'Stim_amp_pA': 'Stimulus Amplitude (pA)', 
                                        'Time_constant_ms': 'Time_constant (ms)',
                                        'Protocol_id': 'Protocol'},  # Renames the legend title
                                hover_data={'Sweep': True}  # Adds 'Sweep' to hover data
                                    )
            
            # Create a Plotly figure and update layout
            figure = go.Figure(data=scatter_plot.data)
            figure.update_layout(
                autosize=True,
                xaxis_title="Stimulus amplitude (pA)", 
                yaxis_title="Time_constant (ms)",
                template="plotly_white"
            )
            
            return figure
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_SS_potential_v_stim_amp_plolty():
        if input.Cell_file_select():
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_QC_table = cell_dict["Sweep_QC_table"]
            sweep_info_QC_table = pd.merge(Sweep_info_table, Sweep_QC_table.loc[:, ['Passed_QC', "Sweep"]], on="Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True, :]
            sub_sweep_info_QC_table['Protocol_id'] = sub_sweep_info_QC_table['Protocol_id'].astype(str)
            
            sub_sweep_info_QC_table_SS_potential = sub_sweep_info_QC_table.dropna(subset=['SS_potential_mV'])
            SS_potential_mV_list = list(sub_sweep_info_QC_table_SS_potential['SS_potential_mV'])
            Stim_amp_pA_list = list(sub_sweep_info_QC_table_SS_potential['Stim_SS_pA'])
            
            # Perform linear regression to get Input Resistance and intercept
            IR_regression_fit, intercept = fir_an.linear_fit(Stim_amp_pA_list, SS_potential_mV_list)
            
            # Create regression line points
            min_amp = min(Stim_amp_pA_list)
            max_amp = max(Stim_amp_pA_list)
            regression_line_x = np.arange(min_amp, max_amp)
            #regression_line_x = [min_amp, max_amp]
            regression_line_y = IR_regression_fit * regression_line_x + intercept
            
            # Create a scatter plot with color based on 'Protocol_id'
            scatter_plot = px.scatter(
                sub_sweep_info_QC_table,
                x='Stim_SS_pA', 
                y='SS_potential_mV',
                color='Protocol_id',  # Colors the points based on 'Protocol_id'
                labels={
                    'Stim_SS_pA': 'Stimulus Amplitude (pA)', 
                    'SS_potential_mV': 'Steady State potential (mV)',
                    'Protocol_id': 'Protocol'
                },
                hover_data={'Sweep': True}  # Adds 'Sweep' to hover data
            )
            
            # Create a Plotly figure and update layout
            figure = go.Figure(data=scatter_plot.data)
        
            
            
            # Add Input Resistance to the legend
            figure.add_trace(go.Scatter(
                x=regression_line_x,  # Empty data for the legend entry
                y=regression_line_y,
                mode='lines',
                line=dict(color='blue', dash='dash'),
                name=f'Input Resistance: {IR_regression_fit:.2f} GΩ'  # Show Input Resistance in legend
            ))
            
            # Update layout
            figure.update_layout(
                autosize=True,
                xaxis_title="Stimulus amplitude (pA)", 
                yaxis_title='Steady State potential (mV)',
                template="plotly_white"
            )
            
            return figure

        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_Holding_potential_v_Holding_current_plolty():
        if input.Cell_file_select():
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_QC_table = cell_dict["Sweep_QC_table"]
            sweep_info_QC_table = pd.merge(Sweep_info_table, Sweep_QC_table.loc[:, ['Passed_QC', "Sweep"]], on="Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True, :]
            sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
            
            sub_sweep_info_QC_table_Holding_potential = sub_sweep_info_QC_table.dropna(subset=['Holding_potential_mV'])
            Holding_potential_mV = list(sub_sweep_info_QC_table_Holding_potential['Holding_potential_mV'])
            Holding_current_list = list(sub_sweep_info_QC_table_Holding_potential['Holding_current_pA'])
            
            # Perform linear regression to get slope and intercept
            slope, Resting_potential_regression_fit = fir_an.linear_fit(Holding_current_list, Holding_potential_mV)
            
            # Create regression line points
            min_current = min(Holding_current_list)
            max_current = max(Holding_current_list)
            regression_line_x = np.arange(min_current, max_current, 0.01)
            
            regression_line_y = slope * regression_line_x + Resting_potential_regression_fit
                                 
            
            # Get the resting potential point
            resting_potential_x = 0  # Assuming the resting potential corresponds to 0 current
            resting_potential_y = Resting_potential_regression_fit
            
            # Create a scatter plot with color based on 'Protocol_id'
            scatter_plot = px.scatter(
                sub_sweep_info_QC_table_Holding_potential,
                x='Holding_current_pA', 
                y='Holding_potential_mV',
                color='Protocol_id',  # Colors the points based on 'Protocol_id'
                labels={
                    'Holding_current_pA': 'Holding Current (pA)', 
                    'Holding_potential_mV': 'Holding potential (mV)',
                    'Protocol_id': 'Protocol'
                },
                hover_data={'Sweep': True}  # Adds 'Sweep' to hover data
            )
            
            # Create a Plotly figure and update layout
            figure = go.Figure(data=scatter_plot.data)
        
            # Add the linear regression line (without showing in the legend)
            figure.add_trace(go.Scatter(
                x=regression_line_x, 
                y=regression_line_y, 
                mode='lines', 
                line=dict(color='blue', dash='dash'),
                showlegend=True  # Exclude from legend
            ))
            
            # Add the resting potential point (with legend)
            figure.add_trace(go.Scatter(
                x=[resting_potential_x], 
                y=[resting_potential_y], 
                mode='markers',
                marker=dict(color='red', size=10), 
                text=[f'Resting Potential: {resting_potential_y:.2f} mV'],  # Display value in legend
                textposition='top right',
                name=f'Resting Potential: {resting_potential_y:.2f} mV'
            ))
            
            # Update layout
            figure.update_layout(
                autosize=True,
                xaxis_title="Holding current (pA)", 
                yaxis_title='Holding potential (mV)',
                template="plotly_white"
            )
            
            return figure

        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_Holding_potential_mV_v_stim_amp_plolty():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_QC_table = cell_dict["Sweep_QC_table"]
            sweep_info_QC_table = pd.merge(Sweep_info_table, Sweep_QC_table.loc[:, ['Passed_QC', "Sweep"]], on="Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True, :]
            sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
            # Create a scatter plot with color based on 'Protocol_id'
            scatter_plot = px.scatter(
                                sub_sweep_info_QC_table,
                                x='Stim_amp_pA', 
                                y='Holding_potential_mV',
                                color='Protocol_id',  # Colors the points based on 'Protocol_id'
                                labels={'Stim_amp_pA': 'Stimulus Amplitude (pA)', 
                                        'Holding_potential_mV': 'Holding  potential (mV)',
                                        'Protocol_id': 'Protocol'},  # Renames the legend title
                                hover_data={'Sweep': True}  # Adds 'Sweep' to hover data
                                    )
            
            # Create a Plotly figure and update layout
            figure = go.Figure(data=scatter_plot.data)
            figure.update_layout(
                autosize=True,
                xaxis_title="Stimulus amplitude (pA)", 
                yaxis_title='Holding potential (mV)',
                template="plotly_white"
            )
            
            return figure
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_Holding_current_pA_v_stim_amp_plolty():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_QC_table = cell_dict["Sweep_QC_table"]
            sweep_info_QC_table = pd.merge(Sweep_info_table, Sweep_QC_table.loc[:, ['Passed_QC', "Sweep"]], on="Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True, :]
            sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
            # Create a scatter plot with color based on 'Protocol_id'
            scatter_plot = px.scatter(
                                sub_sweep_info_QC_table,
                                x='Stim_amp_pA', 
                                y='Holding_current_pA',
                                color='Protocol_id',  # Colors the points based on 'Protocol_id'
                                labels={'Stim_amp_pA': 'Stimulus Amplitude (pA)', 
                                        'Holding_current_pA': 'Holding  current (pA)',
                                        'Protocol_id': 'Protocol'},  # Renames the legend title
                                hover_data={'Sweep': True}  # Adds 'Sweep' to hover data
                                    )
            
            # Create a Plotly figure and update layout
            figure = go.Figure(data=scatter_plot.data)
            figure.update_layout(
                autosize=True,
                xaxis_title="Stimulus amplitude (pA)", 
                yaxis_title='Holding current (pA)',
                template="plotly_white"
            )
            
            return figure
        
        
    @output
    @render.data_frame
    @reactive.event(input.Update_cell_btn)
    def Sweep_info_table():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            Sweep_info_table['Bridge_Error_extrapolated'] =  Sweep_info_table['Bridge_Error_extrapolated'].astype(str)
            Sweep_info_table = Sweep_info_table.replace(np.nan, 'NaN')
            print(Sweep_info_table)
            return Sweep_info_table
    
    @output
    @render.data_frame
    @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction)
    def Sweep_spike_features_table():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()

            Full_SF_table = cell_dict['Full_SF_table']


            spike_table = pd.DataFrame(Full_SF_table.loc[Full_SF_table['Sweep']==input.Sweep_selected(),"SF"].values[0])
            
            return spike_table
    
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction,input.Superimpose_BE_Correction)
    def Sweep_spike_plotly():
        if input.Sweep_selected() :
            # Sample data
            cell_dict = get_cell_tables()
            Full_TVC_table = cell_dict['Full_TVC_table']
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict["Sweep_info_table"]
            current_sweep = input.Sweep_selected()
            BE = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Bridge_Error_GOhms'].values[0]
            color_dict = {'Input_current_pA': 'black',
                          'Membrane potential': 'black',
                          'Membrane potential BE Corrected': "black",
                          "Membrane potential original" : "red",
                          'Threshold': "#a2c5fc",
                          "Upstroke": "#0562f5",
                          "Peak": "#2f0387",
                          "Downstroke": "#9f02e8",
                          "Fast_Trough": "#c248fa",
                          "fAHP": "#d991fa",
                          "Trough": '#fc3873',
                          "Slow_Trough": "#96022e",
                          "ADP":'#fc0366'
                          }
            
            
            current_TVC_table = ordifunc.get_filtered_TVC_table(Full_TVC_table,current_sweep)
            if input.BE_correction()==True:
                Full_SF_dict_table = cell_dict['Full_SF_dict_table']
                Full_SF_table = sp_an.create_Full_SF_table(Full_TVC_table, Full_SF_dict_table, cell_sweep_info_table = Sweep_info_table,BE_correct=True,)
            else:
                Full_SF_dict_table = cell_dict['Full_SF_dict_table']
                Full_SF_table = sp_an.create_Full_SF_table(Full_TVC_table, Full_SF_dict_table, cell_sweep_info_table = Sweep_info_table,BE_correct=False,)
            
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Membrane Potential plot", "Input Current plot"))
            
            if input.BE_correction() == True:
                if not np.isnan(BE):
                    current_TVC_table.loc[:,'Membrane_potential_mV'] = current_TVC_table.loc[:,'Membrane_potential_mV']-BE*current_TVC_table.loc[:,'Input_current_pA']
                fig.add_trace(go.Scatter(x=current_TVC_table['Time_s'], y=current_TVC_table['Membrane_potential_mV'], mode='lines', name='Membrane potential BE Corrected', line=dict(color=color_dict['Membrane potential'], width =1 )), row=1, col=1)
                if input.Superimpose_BE_Correction() == True:
                    current_TVC_table_second = ordifunc.get_filtered_TVC_table(Full_TVC_table,current_sweep)
                    fig.add_trace(go.Scatter(x=current_TVC_table_second['Time_s'], y=current_TVC_table_second['Membrane_potential_mV'], mode='lines', name='Membrane potential original', line=dict(color=color_dict['Membrane potential original'])), row=1, col=1)
            
            
            else: 
                fig.add_trace(go.Scatter(x=current_TVC_table['Time_s'], y=current_TVC_table['Membrane_potential_mV'], mode='lines', name='Membrane potential ', line=dict(color=color_dict['Membrane potential'], width =1)), row=1, col=1)
            # Plot for Membrane_potential_mV vs Time_s from SF_table if not empty
            
            SF_table = Full_SF_table.loc[current_sweep, "SF"]
            if not SF_table.empty:
                print(SF_table['Feature'].unique())
                for feature in SF_table['Feature'].unique():
                    if feature in ["Spike_heigth", "Spike_width_at_half_heigth"]:
                        continue
                    subset = SF_table[SF_table['Feature'] == feature]
                    
                    fig.add_trace(go.Scatter(x=subset['Time_s'], y=subset['Membrane_potential_mV'], mode='markers', name=feature, marker=dict(color=color_dict[feature])), row=1, col=1)
            
            # Plot for Input_current_pA vs Time_s from TVC
            fig.add_trace(go.Scatter(x=current_TVC_table['Time_s'], y=current_TVC_table['Input_current_pA'], mode='lines', name='Input current', line=dict(color=color_dict['Input_current_pA'] , width =1)), row=2, col=1)
            
            # Update layout
            
            
            # fig.update_layout(height=800, showlegend=True, title_text="TVC and SF Plots", )
            
            fig.update_layout(
            height=800,
            showlegend=True,
            title_text="TVC and SF Plots",
            hovermode='x unified',  # Use 'x unified' to create a unified hover mode
            )
            fig.update_xaxes(title_text="Time_s", row=2, col=1)
            fig.update_yaxes(title_text="Membrane_potential_mV", row=1, col=1)
            fig.update_yaxes(title_text="Input_current_pA", row=2, col=1)
    
     
            return fig
            
                    
   
   
    # @output
    # @render.plot()
    # @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction,input.Superimpose_BE_Correction,input.Update_plot_parameters)
    # def Sweep_spike_plot():
    #     if input.Sweep_selected() :

    #         cell_dict = get_cell_tables()
    #         Full_TVC_table = cell_dict['Full_TVC_table']
    #         Full_SF_table = cell_dict['Full_SF_table']
    #         Sweep_info_table = cell_dict["Sweep_info_table"]
    #         current_sweep = input.Sweep_selected()
            
    #         current_TVC_table = ordifunc.get_filtered_TVC_table(Full_TVC_table,current_sweep)
            
    #         if input.use_custom_x_axis() == False:
            
    #             stim_start_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_start_s'].values[0]
    #             stim_start_time -= .1
    #             stim_end_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_end_s'].values[0]
    #             stim_end_time += .1
    #         else:
    #             stim_start_time = input.SF_plot_min()
    #             stim_end_time = input.SF_plot_max()
                
            
    #         current_TVC_table = current_TVC_table.loc[current_TVC_table['Time_s'] >= (stim_start_time), :]
    #         current_TVC_table = current_TVC_table.loc[current_TVC_table['Time_s'] <= (stim_end_time), :]
            

            

    #         if input.BE_correction():
                
    #             BE = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Bridge_Error_GOhms'].values[0]
    #             #Full_SF_dict_table = cells[1]
    #             Full_SF_dict_table = cell_dict['Full_SF_dict_table']
    #             Full_SF_table = sp_an.create_Full_SF_table(Full_TVC_table, Full_SF_dict_table, cell_sweep_info_table = Sweep_info_table,BE_correct=True,)
    #             current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']
                
    #             if input.Superimpose_BE_Correction() :
    #                 original_TVC_table = current_TVC_table.copy()
    #                 original_TVC_table.loc[:,'Membrane_potential_mV'] -= BE*original_TVC_table.loc[:,'Input_current_pA']
    #                 BE_voltage_table = original_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
    #                 BE_voltage_table.loc[:,'Legend']="Membrane potential BE Corrected"
    #                 BE_voltage_table.columns = ['Time_s', 'Value','Legend']
    #                 BE_voltage_table['Plot'] = 'Membrane_potential_mV'
                    
    #                 Potential_table = current_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
    #                 Potential_table.loc[:,'Legend']="Membrane potential original"
    #                 Potential_table.columns = ['Time_s', 'Value','Legend']
    #                 Potential_table['Plot'] = 'Membrane_potential_mV'
                    
    #             else:
    #                 current_TVC_table.loc[:,'Membrane_potential_mV'] -= BE*current_TVC_table.loc[:,'Input_current_pA']
    #                 Potential_table = current_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
    #                 Potential_table.loc[:,'Legend']="Membrane potential BE Corrected"
    #                 Potential_table.columns = ['Time_s', 'Value','Legend']
    #                 Potential_table['Plot'] = 'Membrane_potential_mV'

    #         else:

    #             current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']
    #             Potential_table = current_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
    #             Potential_table.loc[:,'Legend']="Membrane potential original"
    #             Potential_table.columns = ['Time_s', 'Value','Legend']
    #             Potential_table['Plot'] = 'Membrane_potential_mV'
            
            
            
    #         Current_table = current_TVC_table.loc[:,['Time_s','Input_current_pA']]
    #         Current_table.loc[:,'Legend']="Input current"
    #         Current_table.columns = ['Time_s', 'Value','Legend']
    #         Current_table['Plot'] = 'Input_current_pA'
            
            
            
    #         New_TVC_Table = pd.concat([Potential_table,Current_table], axis = 0, ignore_index = True)
    #         if input.Superimpose_BE_Correction() and input.BE_correction() :
    #             New_TVC_Table = pd.concat([New_TVC_Table,BE_voltage_table], axis = 0, ignore_index = True)
    #         New_TVC_Table['Plot'] = New_TVC_Table['Plot'].astype('category')
    #         New_TVC_Table['Plot'] = New_TVC_Table['Plot'].cat.reorder_categories(['Membrane_potential_mV', 'Input_current_pA'])
    #         color_dict = {'Input_current_pA' : 'black',
    #                       'Membrane potential original' : 'black',
    #                       'Membrane potential BE Corrected' : "red",
    #                       'Threshold':"#a2c5fc",
    #                       "Upstroke" : "#0562f5", 
    #                       "Peak" : "#2f0387",
    #                       "Downstroke":"#9f02e8",
    #                       "Fast_Trough" : "#c248fa",
    #                       "fAHP":"#d991fa",
    #                       "Trough":'#fc3873',
    #                       "Slow_Trough" :  "#96022e"

    #             }
            
    #         New_TVC_Table = New_TVC_Table.astype({'Time_s':'float','Value':'float'})
    #         SF_plot = p9.ggplot(New_TVC_Table,p9.aes(x='Time_s',y='Value',color='Legend'))+p9.geom_line()+p9.scale_color_manual(color_dict)
    #         if current_SF_table.shape[0]>=1:
    #             current_SF_table = current_SF_table.loc[(current_SF_table['Feature']!='Spike_width_at_half_heigth')&(current_SF_table['Feature']!='Spike_heigth'),:]
    #             current_SF_table_voltage = current_SF_table.loc[:,['Time_s','Membrane_potential_mV','Feature']]
    #             current_SF_table_voltage.columns=['Time_s','Value','Feature']
    #             current_SF_table_voltage['Plot'] = 'Membrane_potential_mV'
                
    #             current_SF_table_stim = current_SF_table.loc[:,['Time_s','Input_current_pA','Feature']]
    #             current_SF_table_stim.columns=['Time_s','Value','Feature']
    #             current_SF_table_stim['Plot'] = 'Input_current_pA'
                
    #             current_SF_table_voltage = pd.concat([current_SF_table_voltage,current_SF_table_stim], axis = 0, ignore_index = True)

    #             current_SF_table_voltage['Plot'] = current_SF_table_voltage['Plot'].astype('category')
    #             current_SF_table_voltage['Plot'] = current_SF_table_voltage['Plot'].cat.reorder_categories(['Membrane_potential_mV', 'Input_current_pA'])
                
                
    #             current_SF_table_voltage = current_SF_table_voltage.loc[current_SF_table_voltage['Time_s'] >= (stim_start_time), :]
    #             current_SF_table_voltage = current_SF_table_voltage.loc[current_SF_table_voltage['Time_s'] <= (stim_end_time), :]
    #             current_SF_table_voltage = current_SF_table_voltage[current_SF_table_voltage['Feature'].isin(input.Spike_Feature_to_display())]
    #             current_SF_table_voltage=current_SF_table_voltage.astype({'Time_s':'float','Value':'float'})
    #             if input.Spike_feature_representation() == 'Point':
    #                 SF_plot+=p9.geom_point(current_SF_table_voltage,p9.aes(x='Time_s',y='Value',color='Feature'))
    #             elif input.Spike_feature_representation() == 'Line':
    #                 SF_plot+=p9.geom_line(current_SF_table_voltage,p9.aes(x='Time_s',y='Value',color='Feature'))

    #         SF_plot += p9.facet_grid('Plot ~ .',scales='free')
            

    #         return SF_plot

    @reactive.Calc
    @reactive.event(input.Update_cell_btn)
    def spike_analysis_plot():
        if input.Cell_file_select() :
            
            cell_dict = get_cell_tables()
            Full_TVC_table = cell_dict['Full_TVC_table']
            Full_SF_table = cell_dict['Full_SF_table']
            # Full_TVC_table = cells[0]
            # Full_SF_table = cells[2]
            spike_trace_table = sp_an.get_cell_spikes_traces(Full_TVC_table, Full_SF_table)

            return spike_trace_table
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def Spike_phase_plane_plot():
        spike_trace_table = spike_analysis_plot()
        # Filter table based on the selected Sweep
        sub_spike_super_position_table = spike_trace_table.loc[spike_trace_table['Sweep'] == input.Sweep_selected(), :]
        discrete_colors = []

        # Determine the list of current spike indices
        if sub_spike_super_position_table.shape[0] == 0:
            current_spike_index_list = []
        else:
            sub_spike_super_position_table = sub_spike_super_position_table.astype({'Spike_index': 'category'})
            current_spike_index_list = [str(x) for x in range(sub_spike_super_position_table['Spike_index'].astype(int).min(),
                                                              sub_spike_super_position_table['Spike_index'].astype(int).max() + 1)]
            discrete_colors = sample_colorscale('plasma_r', minmax_scale(current_spike_index_list))

        # Handle colors for spikes not in the filtered table
        for x in range(spike_trace_table['Spike_index'].astype(int).min(), spike_trace_table['Spike_index'].astype(int).max() + 1):
            if x not in sub_spike_super_position_table['Spike_index'].astype(int).tolist():
                discrete_colors.append('rgb(173, 173, 173)')  # Default gray color for missing spikes

        # Prepare color mapping
        new_discrete_colors = []
        colored_spike_index_list = []
        uncolored_spike_index_list = []
        i = 0
        spike_trace_table['Spike_index'] = spike_trace_table['Spike_index'].astype(str)  # Ensure 'Spike_index' is a string
        color_dict = {}

        # Create full spike index list as strings
        full_spike_index_list = [str(x) for x in range(spike_trace_table['Spike_index'].astype(int).min(),
                                                       spike_trace_table['Spike_index'].astype(int).max() + 1)]

        for x in full_spike_index_list:
            if x not in current_spike_index_list:
                new_discrete_colors.append('rgb(173, 173, 173)')
                color_dict[x] = 'rgb(173, 173, 173)'
                uncolored_spike_index_list.append(x)
            else:
                colored_spike_index_list.append(x)

        # Update color dictionary with discrete colors
        for x in colored_spike_index_list:
            color_dict[x] = discrete_colors[i]
            i += 1

        # Combine colored and uncolored lists
        full_spike_index_list_ordered = uncolored_spike_index_list + colored_spike_index_list

        # Sort data for plotting
        spike_trace_table = spike_trace_table.sort_values(by=["Spike_index", 'Time_s'])

        # Create Plotly figure using graph_objects
        fig = go.Figure()

        # Add traces for each Spike_index
        for spike_index in full_spike_index_list_ordered:
            spike_data = spike_trace_table[spike_trace_table['Spike_index'] == spike_index]
            fig.add_trace(
                go.Scatter(
                y=spike_data['Potential_first_time_derivative_mV/s'],
                x=spike_data['Membrane_potential_mV'],
                mode='lines',
                name=f'Spike {spike_index}',
                line=dict(color=color_dict[spike_index]),
                hovertemplate=(
                    'Spike Index: %{text}<br>'
                    'Sweep: %{customdata[0]}<br>'
                    'Membrane deriv: %{y:.2f} s<br>'
                    'Membrane Potential: %{x:.2f} mV'
                ),
                text=[spike_index] * len(spike_data),
                customdata=spike_data[['Sweep']],
                showlegend=True
            )

            )

        # Update figure layout
        fig.update_layout(
            yaxis_title='Membrane potential time derivative (mV/ms)',
            xaxis_title='Membrane Potential (mV)',
            legend_title='Spike Index'
        )
        return fig
        return fig
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def Spike_superposition_plot():
        spike_trace_table = spike_analysis_plot()
        # Filter table based on the selected Sweep
        sub_spike_super_position_table = spike_trace_table.loc[spike_trace_table['Sweep'] == input.Sweep_selected(), :]
        discrete_colors = []

        # Determine the list of current spike indices
        if sub_spike_super_position_table.shape[0] == 0:
            current_spike_index_list = []
        else:
            sub_spike_super_position_table = sub_spike_super_position_table.astype({'Spike_index': 'category'})
            current_spike_index_list = [str(x) for x in range(sub_spike_super_position_table['Spike_index'].astype(int).min(),
                                                              sub_spike_super_position_table['Spike_index'].astype(int).max() + 1)]
            discrete_colors = sample_colorscale('plasma_r', minmax_scale(current_spike_index_list))

        # Handle colors for spikes not in the filtered table
        for x in range(spike_trace_table['Spike_index'].astype(int).min(), spike_trace_table['Spike_index'].astype(int).max() + 1):
            if x not in sub_spike_super_position_table['Spike_index'].astype(int).tolist():
                discrete_colors.append('rgb(173, 173, 173)')  # Default gray color for missing spikes

        # Prepare color mapping
        new_discrete_colors = []
        colored_spike_index_list = []
        uncolored_spike_index_list = []
        i = 0
        spike_trace_table['Spike_index'] = spike_trace_table['Spike_index'].astype(str)  # Ensure 'Spike_index' is a string
        color_dict = {}

        # Create full spike index list as strings
        full_spike_index_list = [str(x) for x in range(spike_trace_table['Spike_index'].astype(int).min(),
                                                       spike_trace_table['Spike_index'].astype(int).max() + 1)]

        for x in full_spike_index_list:
            if x not in current_spike_index_list:
                new_discrete_colors.append('rgb(173, 173, 173)')
                color_dict[x] = 'rgb(173, 173, 173)'
                uncolored_spike_index_list.append(x)
            else:
                colored_spike_index_list.append(x)

        # Update color dictionary with discrete colors
        for x in colored_spike_index_list:
            color_dict[x] = discrete_colors[i]
            i += 1

        # Combine colored and uncolored lists
        full_spike_index_list_ordered = uncolored_spike_index_list + colored_spike_index_list

        # Sort data for plotting
        spike_trace_table = spike_trace_table.sort_values(by=["Spike_index", 'Time_s'])

        # Create Plotly figure using graph_objects
        fig = go.Figure()

        # Add traces for each Spike_index
        for spike_index in full_spike_index_list_ordered:
            spike_data = spike_trace_table[spike_trace_table['Spike_index'] == spike_index]
            fig.add_trace(
                go.Scatter(
                x=spike_data['Time_s'],
                y=spike_data['Membrane_potential_mV'],
                mode='lines',
                name=f'Spike {spike_index}',
                line=dict(color=color_dict[spike_index]),
                hovertemplate=(
                    'Spike Index: %{text}<br>'
                    'Sweep: %{customdata[0]}<br>'
                    'Time: %{x:.4f} s<br>'
                    'Membrane Potential: %{y:.2f} mV'
                ),
                text=[spike_index] * len(spike_data),
                customdata=spike_data[['Sweep']],
                showlegend=True
            )
            )

        # Update figure layout
        fig.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Membrane Potential (mV)',
            legend_title='Spike Index'
        )
        return fig
        
    
    
    @output
    @render.data_frame
    @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction)
    def Spike_feature_table():
        if input.Sweep_selected() :

            cell_dict = get_cell_tables()
            Full_TVC_table = cell_dict['Full_TVC_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            current_sweep = input.Sweep_selected()
            #Full_TVC_table = cells[0]
            
            #Sweep_info_table = cells[4]
            if input.use_custom_x_axis() == False:
            
                stim_start_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_start_s'].values[0]
                stim_start_time -= .1
                stim_end_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_end_s'].values[0]
                stim_end_time += .1
            else:
                stim_start_time = input.SF_plot_min()
                stim_end_time = input.SF_plot_max()
                
            
            current_TVC_table= Full_TVC_table.loc[str(current_sweep),'TVC']
            
            current_TVC_table = current_TVC_table.loc[current_TVC_table['Time_s'] >= (stim_start_time), :]
            current_TVC_table = current_TVC_table.loc[current_TVC_table['Time_s'] <= (stim_end_time), :]

            
            if input.BE_correction():
                
                BE = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Bridge_Error_GOhms'].values[0]
                Full_SF_dict_table = cell_dict['Full_SF_dict_table']
                #Full_SF_dict_table = cells[1]
                original_TVC_table = current_TVC_table.copy()
                original_TVC_table.loc[:,'Membrane_potential_mV'] -= BE*original_TVC_table.loc[:,'Input_current_pA']
                Full_SF_table = sp_an.create_Full_SF_table(Full_TVC_table, Full_SF_dict_table, cell_sweep_info_table = Sweep_info_table, BE_correct=True)
                current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']
            else:
                #Full_SF_table = cells[2]
                Full_SF_table = cell_dict['Full_SF_table']
                current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']

            return current_SF_table
    
    
    @reactive.Calc
    @reactive.event(input.Update_cell_btn,input.Sweep_selected)
    def get_sweep_linear_properties():
        if input.Sweep_selected() :
            cell_dict = get_cell_tables()
            current_sweep = input.Sweep_selected()
            
            Full_TVC_table = cell_dict['Full_TVC_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            stim_start_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_start_s'].values[0]
            stim_end_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_end_s'].values[0]
            current_TVC_table= ordifunc.get_filtered_TVC_table(Full_TVC_table,current_sweep,do_filter=True,filter=5.,do_plot=False)
        
            
            
            
            voltage_spike_table=sp_an.identify_spike(np.array(current_TVC_table.Membrane_potential_mV),
                                            np.array(current_TVC_table.Time_s),
                                            np.array(current_TVC_table.Input_current_pA),
                                            stim_start_time,
                                            (stim_start_time+stim_end_time)/2,do_plot=False)
            
            if len(voltage_spike_table['Peak']) != 0:
                is_spike = True
            else:
                is_spike = False
            
            sub_test_TVC=current_TVC_table[current_TVC_table['Time_s']<=stim_start_time-.005].copy()
            sub_test_TVC=sub_test_TVC[sub_test_TVC['Time_s']>=(stim_start_time-.200)]
            
            CV_pre_stim=np.std(np.array(sub_test_TVC['Membrane_potential_mV']))/np.mean(np.array(sub_test_TVC['Membrane_potential_mV']))
            
            sub_test_TVC=current_TVC_table[current_TVC_table['Time_s']<=(stim_end_time)].copy()
            sub_test_TVC=sub_test_TVC[sub_test_TVC['Time_s']>=((stim_start_time+stim_end_time)/2)]
            
            CV_second_half=np.std(np.array(sub_test_TVC['Membrane_potential_mV']))/np.mean(np.array(sub_test_TVC['Membrane_potential_mV']))
            BE=Sweep_info_table.loc[current_sweep,"Bridge_Error_GOhms"]
            stim_amp = Sweep_info_table.loc[current_sweep,"Stim_amp_pA"]
            if (is_spike == True or
                np.abs(Sweep_info_table.loc[current_sweep,"Stim_amp_pA"])<2.0 or
                np.abs(CV_pre_stim) > 0.01 or 
                np.abs(CV_second_half) > 0.01):
                NRMSE = np.nan
                best_A = np.nan
                best_tau = np.nan,
                SS_potential = np.nan
                holding_potential = np.nan
                resting_potential = np.nan
                time_cst = np.nan
                R_in = np.nan
                
                
            else:
                best_A,best_tau,SS_potential,holding_potential,NRMSE=sw_an.fit_membrane_time_cst(current_TVC_table,
                                                                                 stim_start_time,
                                                                                 (stim_start_time+.300),do_plot=False)
                
                
                if NRMSE > 0.3:
                    SS_potential = np.nan
                    resting_potential = np.nan
                    holding_potential = np.nan
                    R_in = np.nan
                    time_cst = np.nan
                    
                    
                else:
                    Holding_current,SS_current = sw_an.fit_stimulus_trace(
                        current_TVC_table, stim_start_time, stim_end_time, do_plot=False)[:2]
                    
                    
                    R_in=((SS_potential-holding_potential)/stim_amp)-BE
                    
                    time_cst=best_tau*1e3 #convert s to ms
                    resting_potential = holding_potential - (R_in+BE)*Holding_current
                    
                    
                    
            
                    
            linear_fit_plot=sw_an.fit_membrane_time_cst(current_TVC_table, stim_start_time, (stim_start_time+.300),do_plot=True)
            
            
            # if Sweep_info_table.loc[current_sweep,"Bridge_Error_extrapolated"] == "True":
                
            #     is_extrapolated = "Extrapolated"
            # else:
            #     is_extrapolated = "Not extrapolated"
            condition_table=pd.DataFrame([is_spike,
                                          np.abs(Sweep_info_table.loc[current_sweep,"Stim_amp_pA"]),
                                          #is_extrapolated,
                                          np.abs(CV_pre_stim),
                                          np.abs(CV_second_half),
                                          NRMSE]).T
            condition_table.columns=['Potential spike',"Stimulus_amplitude",'CV pre stim','CV second half','NRMSE']
            condition_to_meet = pd.DataFrame(["No spike","|| ≤2pA", '|| < 0.01','|| <0.01', '< 0.3']).T
            condition_to_meet.columns=['Potential spike',"Stimulus_amplitude",'CV pre stim','CV second half','NRMSE']
            
            stim_amp_cond_respected = False if np.abs(Sweep_info_table.loc[current_sweep,"Stim_amp_pA"])<2.0 else True
            #BE_extra_cond_respected = False if is_extrapolated == "Extrapolated" else True   
            CV_pre_stim_cond_respected = False if np.abs(CV_pre_stim) > 0.01 else True 
            CV_second_half_cond_respected = False if np.abs(CV_second_half) > 0.01 else True
            NRMSE_cond_respected = False if np.isnan(NRMSE)==True else False if NRMSE > 0.3 else True
            
            condition_respected = pd.DataFrame([not is_spike, stim_amp_cond_respected, CV_pre_stim_cond_respected, CV_second_half_cond_respected, NRMSE_cond_respected]).T
            condition_respected.columns=['Potential spike',"Stimulus_amplitude",'CV pre stim','CV second half','NRMSE']

            condition_table = pd.concat([condition_to_meet,condition_table, condition_respected],axis=0, ignore_index=True)
            condition_table.index=['condition to meet', 'result', 'condition respected']
            
            linear_fit_table=pd.DataFrame([best_A,
                                          best_tau,
                                          SS_potential,
                                          NRMSE])
            linear_fit_table.index=['A','tau', 'C','NRMSE']
            
            properties_table=pd.DataFrame([BE,
                                          stim_amp,
                                          holding_potential,
                                          resting_potential,
                                          R_in,
                                          time_cst
                                          ])
            properties_table.index= ['Bridge Error', "Stimulus amplitude",'Holding potential mV', 'Resing Potential mV', 'Input Resistance GOhms', 'Time constant ms']
            
            
            return condition_table, linear_fit_plot, linear_fit_table, properties_table
           
    
    @output
    @render.table
    def Sweep_QC_table():
        if input.Sweep_selected() :
            cells_dict = get_cell_tables()
            Sweep_QC_table = cells_dict['Sweep_QC_table']
            return Sweep_QC_table
            
            
    
    @output
    @render.table
    def linear_properties_conditions_table():
        if input.Sweep_selected() :
            
            
            condition_table = get_sweep_linear_properties()[0]
            condition_table_styled = condition_table.style.set_table_attributes('class="dataframe shiny-table table w-auto"').apply(highlight_late)
            return condition_table_styled
                
    @output
    @render.plot()
    def linear_properties_fit():
        if input.Sweep_selected() :
            cell_dict = get_cell_tables()
            current_sweep = input.Sweep_selected()
            
            Sweep_info_table = cell_dict['Sweep_info_table']

            
            stim_start_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_start_s'].values[0]
            stim_end_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_end_s'].values[0]
            
            linear_fit_plot=get_sweep_linear_properties()[1]
            
            linear_fit_plot+=p9.xlim(stim_start_time-.1,stim_end_time+.1)
            return linear_fit_plot
        
    @output
    @render.table()
    def linear_properties_fit_table():
        if input.Sweep_selected() :
            linear_fit_table = get_sweep_linear_properties()[2]

            
            return linear_fit_table.T
        
    @output
    @render.table()
    def linear_properties_table():
        if input.Sweep_selected() :
            properties_table = get_sweep_linear_properties()[3]
            properties_table=properties_table.T
            
            return properties_table
        
    @reactive.Calc
    @reactive.event(input.Update_cell_btn,input.Sweep_selected)
    def get_Bridge_Error_plots():
        if input.Sweep_selected() :
            cell_dict = get_cell_tables()
            Full_TVC_table = cell_dict['Full_TVC_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            current_sweep = input.Sweep_selected()
            #Full_TVC_table = cells[0]
            TVC_table = ordifunc.get_filtered_TVC_table(Full_TVC_table, current_sweep, do_filter=True, filter=5., do_plot=False)
            #Sweep_info_table = cells[4]
            stim_start_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_start_s'].values[0]
            stim_end_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_end_s'].values[0]
            Holding_current,SS_current = sw_an.fit_stimulus_trace(
                TVC_table, stim_start_time, stim_end_time,do_plot=False)[:2]
            
            if np.abs((Holding_current-SS_current))>=20.:
                voltage_plot_start,voltage_plot_end, current_plot_start,current_plot_end = sw_an.estimate_bridge_error(
                    TVC_table, SS_current, stim_start_time, stim_end_time, do_plot=True)
             
            
                return voltage_plot_start,voltage_plot_end, current_plot_start,current_plot_end
            
    @output
    @render.plot()
    def BE_voltage_start():
        if input.Sweep_selected() :
            return get_Bridge_Error_plots()[0]
        
    @output
    @render.plot()
    def BE_voltage_end():
        if input.Sweep_selected() :
            return get_Bridge_Error_plots()[1]
        
    @output
    @render.plot()
    def BE_current_start():
        if input.Sweep_selected() :
            return get_Bridge_Error_plots()[2]
        
    @output
    @render.plot()
    def BE_current_end():
        if input.Sweep_selected() :
            return get_Bridge_Error_plots()[3]
            
                
            
    @reactive.Calc
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def get_firing_analysis_table():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            
            Sweep_info_table = cell_dict['Sweep_info_table']
            Full_SF_table = cell_dict["Full_SF_table"]
            
            sweep_QC_table = cell_dict["Sweep_QC_table"]
            cell_fit_table = cell_dict["Cell_fit_table"]
            cell_feature_table = cell_dict['Cell_feature_table']
            
            
            response_type = input.Firing_response_type()

            response_type=response_type.replace(' ','_')
            
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
            sub_cell_feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == response_type,:]
            output_response_list = list(sub_cell_fit_table.loc[:,'Output_Duration'])
            
            stim_freq_table_list=[]
            model_table_list = []
            IO_table_list = []
            Saturation_table_list = []

            for current_response_duration in output_response_list:
                
                if response_type == 'Time_based':
                    response = str(str(int(current_response_duration*1e3))+'ms')
                else:
                    
                    response = str(response_type.replace('_based',' ') + str(int(current_response_duration)))
                    current_response_duration = int(current_response_duration)
                current_stim_freq_table = fir_an.get_stim_freq_table(
                    Full_SF_table.copy(), Sweep_info_table.copy(),sweep_QC_table.copy(), current_response_duration,response_type)
                
                
                current_stim_freq_table['Response'] = response
                stim_freq_table_list.append(current_stim_freq_table)
                
                model = sub_cell_fit_table.loc[sub_cell_fit_table['Output_Duration'] == current_response_duration,'I_O_obs'].values[0]
                if model == 'Hill' or model == 'Hill-Sigmoid':
                    
                    model_table = fir_an.fit_IO_relationship_NEW_TEST(current_stim_freq_table, "--")[1]
                    
                    min_stim = np.nanmin(current_stim_freq_table.loc[:,'Stim_amp_pA'])
                    max_stim = np.nanmax(current_stim_freq_table.loc[:,'Stim_amp_pA'])
                    stim_array = np.arange(min_stim,max_stim,.1)
                    
                    model_table['Response'] = response
                    Gain = sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Gain'].values[0]
                    
                    plot_list = fir_an.get_IO_features_NEW_TEST(current_stim_freq_table, response_type, current_response_duration, True,True)
                    IO_plot = plot_list["3-IO_fit"]
                    Intercept = IO_plot['intercept']
                    
                    IO_freq = stim_array*Gain+Intercept
                    
                    IO_table = pd.DataFrame({'Stim_amp_pA' : stim_array,
                                                "Frequency_Hz" : IO_freq})
                    IO_table['Response'] = response
                    IO_table=IO_table.loc[IO_table['Frequency_Hz']>=-10.,:]
                    
                    if not np.isnan(sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Stimulus'].values[0]):
                        Saturation_stimulus =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Stimulus'].values[0]
                        Saturation_freq =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Frequency'].values[0]
                        Saturation_table = pd.DataFrame({'Stim_amp_pA' : Saturation_stimulus,
                                                    "Frequency_Hz" : Saturation_freq,
                                                    'Response' : response})
                        Saturation_table_list.append(Saturation_table)
                
                    model_table_list.append(model_table)
                    IO_table_list.append(IO_table)
            
            
            
            Full_stim_freq_table = pd.concat([*stim_freq_table_list],axis=0,ignore_index=True)
            Full_model_table = pd.concat([*model_table_list],axis=0,ignore_index=True)
            Full_IO_table = pd.concat([*IO_table_list],axis=0,ignore_index=True)
            Full_IO_table['Response'] = Full_IO_table['Response'].astype('category')
            if len(Saturation_table_list)!=0:
                Full_Saturation_table = pd.concat([*Saturation_table_list],axis=0,ignore_index=True)
            else:
                Full_Saturation_table=pd.DataFrame()
            return Full_stim_freq_table, Full_model_table, Full_IO_table, Full_Saturation_table
            

    @reactive.Calc
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Time_based_analysis():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            
            IO_figure = get_firing_analysis(cell_dict, input.Firing_response_type())
            
            return IO_figure
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Time_based_IO_plotly():
        
        return Time_based_analysis()
    
    
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Gain_regression_plot_time_based_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            gain_table = cell_feature_table.loc[cell_feature_table['Response_type'] == input.Firing_response_type(),:]
            sub_gain_table = gain_table.dropna(subset=['Gain'])
            if sub_gain_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_gain_table.loc[:,'Output_Duration']),
                                                     np.array(sub_gain_table.loc[:,'Gain']))
                if input.Firing_response_type() == 'Time_based':
                    extended_x_data = np.arange(0,np.nanmax(np.array(gain_table.loc[:,'Output_Duration'])),.001)
                    x_legend = "Response duration ms"
                else:
                    extended_x_data = np.arange(0,np.nanmax(np.array(gain_table.loc[:,'Output_Duration'])),.1)
                    if input.Firing_response_type() == "Index_based":
                        x_legend = "Response duration spike index"
                    else:
                        x_legend = "Response duration spike interval"
                original_data_x = np.array(gain_table['Output_Duration'])  # X values from your original data
                original_data_y = np.array(gain_table['Gain'])             # Y values from your original data

                linear_fit_y = slope * extended_x_data + intercept

                # Create the plot
                fig = go.Figure()
                
                # Add original data points
                fig.add_trace(go.Scatter(
                    x=original_data_x,
                    y=original_data_y,
                    mode='markers',
                    marker=dict(symbol='circle', color='black'),
                    name='Original Data'
                ))
                
                # Add the linear fit line
                fig.add_trace(go.Scatter(
                    x=extended_x_data,  # Use the same x-values
                    y=linear_fit_y,     # Linear fit y-values
                    mode='lines',
                    line=dict(color='red', dash='dash'),  # Dashed red line for linear fit
                    name='Linear Fit'
                ))
                
                # Update the layout
                fig.update_layout(
                    title='Gain Linear Fit',
                    xaxis_title=x_legend,
                    yaxis_title='Gain Hz/pA',
                    width=250,
                    height=400,
                    showlegend=False
                )
                
                # Show the plot

                
            
                return fig
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Gain_regression_table_time_based_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            gain_table = cell_feature_table.loc[cell_feature_table['Response_type'] == "Time_based",:]
            sub_gain_table = gain_table.dropna(subset=['Gain'])
            if sub_gain_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_gain_table.loc[:,'Output_Duration']),
                                                     np.array(sub_gain_table.loc[:,'Gain']))
                
                linear_fit_y = slope * np.array(sub_gain_table.loc[:,'Output_Duration']) + intercept
                r2 = r2_score(sub_gain_table.loc[:,'Gain'], linear_fit_y)
                
                table = pd.DataFrame([slope, intercept, r2]).T
                table.columns = ['Slope', 'Intercept', 'R2']
                table = table.T
                table = table.reset_index(drop=False)
                table.columns = [" Gain", " "]
                return table
            
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Threshold_regression_plot_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            Threshold_table = cell_feature_table.loc[cell_feature_table['Response_type'] == input.Firing_response_type(),:]
            sub_Threshold_table = Threshold_table.dropna(subset=['Threshold'])
            if sub_Threshold_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_Threshold_table.loc[:,'Output_Duration']),
                                                     np.array(sub_Threshold_table.loc[:,'Threshold']))
                if input.Firing_response_type() == 'Time_based':
                    extended_x_data = np.arange(0,np.nanmax(np.array(Threshold_table.loc[:,'Output_Duration'])),.001)
                    x_legend = "Response duration ms"
                else:
                    extended_x_data = np.arange(0,np.nanmax(np.array(Threshold_table.loc[:,'Output_Duration'])),.1)
                    if input.Firing_response_type() == "Index_based":
                        x_legend = "Response duration spike index"
                    else:
                        x_legend = "Response duration spike interval"
                original_data_x = np.array(Threshold_table['Output_Duration'])  # X values from your original data
                original_data_y = np.array(Threshold_table['Threshold'])             # Y values from your original data

                linear_fit_y = slope * extended_x_data + intercept

                # Create the plot
                fig = go.Figure()
                
                # Add original data points
                fig.add_trace(go.Scatter(
                    x=original_data_x,
                    y=original_data_y,
                    mode='markers',
                    marker=dict(symbol='circle', color='black'),
                    name='Original Data'
                ))
                
                # Add the linear fit line
                fig.add_trace(go.Scatter(
                    x=extended_x_data,  # Use the same x-values
                    y=linear_fit_y,     # Linear fit y-values
                    mode='lines',
                    line=dict(color='red', dash='dash'),  # Dashed red line for linear fit
                    name='Linear Fit'
                ))
                
                # Update the layout
                fig.update_layout(
                    title='Threshold_table Linear Fit',
                    xaxis_title=x_legend,
                    yaxis_title='Threshold pA',
                    width=250,
                    height=400,
                    showlegend=False
                )
                
                # Show the plot

                
            
                return fig
    
    
    
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Threshold_regression_table_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            gain_table = cell_feature_table.loc[cell_feature_table['Response_type'] == "Time_based",:]
            Threshold_table = cell_feature_table.loc[cell_feature_table['Response_type'] == "Time_based",:]
            sub_Threshold_table = Threshold_table.dropna(subset=['Threshold'])
            if sub_Threshold_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_Threshold_table.loc[:,'Output_Duration']),
                                                     np.array(sub_Threshold_table.loc[:,'Threshold']))
                extended_x_data = np.arange(0,np.nanmax(np.array(Threshold_table.loc[:,'Output_Duration'])),.001)
                original_data_x = np.array(Threshold_table['Output_Duration'])  # X values from your original data
                original_data_y = np.array(Threshold_table['Threshold'])             # Y values from your original data

                linear_fit_y = slope * np.array(sub_Threshold_table.loc[:,'Output_Duration']) + intercept

                r2 = r2_score(sub_Threshold_table.loc[:,'Threshold'], linear_fit_y)
                
                table = pd.DataFrame([slope, intercept, r2]).T
                table.columns = ['Slope', 'Intercept', 'R2']
                table = table.T
                table = table.reset_index(drop=False)
                table.columns = ["Threshold", " "]
                return table
            
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_Frequency_regression_plot_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == input.Firing_response_type(),:]
            sub_feature_table = feature_table.dropna(subset=['Saturation_Frequency'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Saturation_Frequency']))
                if input.Firing_response_type() == 'Time_based':
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.001)
                    x_legend = "Response duration ms"
                else:
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.1)
                    if input.Firing_response_type() == "Index_based":
                        x_legend = "Response duration spike index"
                    else:
                        x_legend = "Response duration spike interval"
                original_data_x = np.array(feature_table['Output_Duration'])  # X values from your original data
                original_data_y = np.array(feature_table['Saturation_Frequency'])             # Y values from your original data

                linear_fit_y = slope * extended_x_data + intercept

                # Create the plot
                fig = go.Figure()
                
                # Add original data points
                fig.add_trace(go.Scatter(
                    x=original_data_x,
                    y=original_data_y,
                    mode='markers',
                    marker=dict(symbol='circle', color='black'),
                    name='Original Data'
                ))
                
                # Add the linear fit line
                fig.add_trace(go.Scatter(
                    x=extended_x_data,  # Use the same x-values
                    y=linear_fit_y,     # Linear fit y-values
                    mode='lines',
                    line=dict(color='red', dash='dash'),  # Dashed red line for linear fit
                    name='Linear Fit'
                ))
                
                # Update the layout
                fig.update_layout(
                    title='Saturation frequency Linear Fit',
                    xaxis_title=x_legend,
                    yaxis_title='Saturation Frequency Hz',
                    width=250,
                    height=400,
                    showlegend=False
                )
                
                # Show the plot

                
            
                return fig
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_Frequency_regression_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == "Time_based",:]
            sub_feature_table = feature_table.dropna(subset=['Saturation_Frequency'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Saturation_Frequency']))

                linear_fit_y = slope * np.array(sub_feature_table.loc[:,'Output_Duration']) + intercept

                r2 = r2_score(sub_feature_table.loc[:,'Saturation_Frequency'], linear_fit_y)
                
                table = pd.DataFrame([slope, intercept, r2]).T
                table.columns = ['Slope', 'Intercept', 'R2']
                table = table.T
                table = table.reset_index(drop=False)
                table.columns = ["Saturation_Frequency", " "]
                return table
            
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_Stimulus_regression_plot_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == input.Firing_response_type(),:]
            sub_feature_table = feature_table.dropna(subset=['Saturation_Stimulus'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Saturation_Stimulus']))
                if input.Firing_response_type() == 'Time_based':
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.001)
                    x_legend = "Response duration ms"
                else:
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.1)
                    if input.Firing_response_type() == "Index_based":
                        x_legend = "Response duration spike index"
                    else:
                        x_legend = "Response duration spike interval"
                original_data_x = np.array(feature_table['Output_Duration'])  # X values from your original data
                original_data_y = np.array(feature_table['Saturation_Stimulus'])             # Y values from your original data

                linear_fit_y = slope * extended_x_data + intercept

                # Create the plot
                fig = go.Figure()
                
                # Add original data points
                fig.add_trace(go.Scatter(
                    x=original_data_x,
                    y=original_data_y,
                    mode='markers',
                    marker=dict(symbol='circle', color='black'),
                    name='Original Data'
                ))
                
                # Add the linear fit line
                fig.add_trace(go.Scatter(
                    x=extended_x_data,  # Use the same x-values
                    y=linear_fit_y,     # Linear fit y-values
                    mode='lines',
                    line=dict(color='red', dash='dash'),  # Dashed red line for linear fit
                    name='Linear Fit'
                ))
                
                # Update the layout
                fig.update_layout(
                    title='Saturation frequency Linear Fit',
                    xaxis_title=x_legend,
                    yaxis_title='Saturation Stimulus pA',
                    width=250,
                    height=400,
                    showlegend=False
                )
                
                # Show the plot

                
            
                return fig
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_Stimulus_regression_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == "Time_based",:]
            sub_feature_table = feature_table.dropna(subset=['Saturation_Stimulus'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Saturation_Stimulus']))

                linear_fit_y = slope * np.array(sub_feature_table.loc[:,'Output_Duration']) + intercept

                r2 = r2_score(sub_feature_table.loc[:,'Saturation_Stimulus'], linear_fit_y)
                
                table = pd.DataFrame([slope, intercept, r2]).T
                table.columns = ['Slope', 'Intercept', 'R2']
                table = table.T
                table = table.reset_index(drop=False)
                table.columns = ["Saturation_Stimulus", " "]
                return table
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Response_Failure_Stimulus_regression_plot_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == input.Firing_response_type(),:]
            sub_feature_table = feature_table.dropna(subset=['Response_Fail_Stimulus'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Response_Fail_Stimulus']))
                if input.Firing_response_type() == 'Time_based':
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.001)
                    x_legend = "Response duration ms"
                else:
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.1)
                    if input.Firing_response_type() == "Index_based":
                        x_legend = "Response duration spike index"
                    else:
                        x_legend = "Response duration spike interval"
                original_data_x = np.array(feature_table['Output_Duration'])  # X values from your original data
                original_data_y = np.array(feature_table['Response_Fail_Stimulus'])             # Y values from your original data

                linear_fit_y = slope * extended_x_data + intercept

                # Create the plot
                fig = go.Figure()
                
                # Add original data points
                fig.add_trace(go.Scatter(
                    x=original_data_x,
                    y=original_data_y,
                    mode='markers',
                    marker=dict(symbol='circle', color='black'),
                    name='Original Data'
                ))
                
                # Add the linear fit line
                fig.add_trace(go.Scatter(
                    x=extended_x_data,  # Use the same x-values
                    y=linear_fit_y,     # Linear fit y-values
                    mode='lines',
                    line=dict(color='red', dash='dash'),  # Dashed red line for linear fit
                    name='Linear Fit'
                ))
                
                # Update the layout
                fig.update_layout(
                    title='Saturation frequency Linear Fit',
                    xaxis_title=x_legend,
                    yaxis_title='Response Failure Stimulus pA',
                    width=250,
                    height=400,
                    showlegend=False
                )
                
                # Show the plot

                
            
                return fig
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Response_Fail_Stimulus_regression_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == "Time_based",:]
            sub_feature_table = feature_table.dropna(subset=['Response_Fail_Stimulus'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Response_Fail_Stimulus']))

                linear_fit_y = slope * np.array(sub_feature_table.loc[:,'Output_Duration']) + intercept

                r2 = r2_score(sub_feature_table.loc[:,'Response_Fail_Stimulus'], linear_fit_y)
                
                table = pd.DataFrame([slope, intercept, r2]).T
                table.columns = ['Slope', 'Intercept', 'R2']
                table = table.T
                table = table.reset_index(drop=False)
                table.columns = ["Response_Fail_Stimulus", " "]
                return table
            
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Response_Failure_Frequency_regression_plot_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == input.Firing_response_type(),:]
            sub_feature_table = feature_table.dropna(subset=['Response_Fail_Frequency'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Response_Fail_Frequency']))
                if input.Firing_response_type() == 'Time_based':
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.001)
                    x_legend = "Response duration ms"
                else:
                    extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.1)
                    if input.Firing_response_type() == "Index_based":
                        x_legend = "Response duration spike index"
                    else:
                        x_legend = "Response duration spike interval"
                original_data_x = np.array(feature_table['Output_Duration'])  # X values from your original data
                original_data_y = np.array(feature_table['Response_Fail_Frequency'])             # Y values from your original data

                linear_fit_y = slope * extended_x_data + intercept

                # Create the plot
                fig = go.Figure()
                
                # Add original data points
                fig.add_trace(go.Scatter(
                    x=original_data_x,
                    y=original_data_y,
                    mode='markers',
                    marker=dict(symbol='circle', color='black'),
                    name='Original Data'
                ))
                
                # Add the linear fit line
                fig.add_trace(go.Scatter(
                    x=extended_x_data,  # Use the same x-values
                    y=linear_fit_y,     # Linear fit y-values
                    mode='lines',
                    line=dict(color='red', dash='dash'),  # Dashed red line for linear fit
                    name='Linear Fit'
                ))
                
                # Update the layout
                fig.update_layout(
                    title='Saturation frequency Linear Fit',
                    xaxis_title=x_legend,
                    yaxis_title='Response Failure Frequency Hz',
                    width=250,
                    height=400,
                    showlegend=False
                )
                
                # Show the plot

                
            
                return fig
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Response_Fail_Frequency_regression_Second():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_feature_table  = cell_dict['Cell_feature_table']
            
            feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == "Time_based",:]
            sub_feature_table = feature_table.dropna(subset=['Response_Fail_Frequency'])
            if sub_feature_table.shape[0]>1:
                slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                                     np.array(sub_feature_table.loc[:,'Response_Fail_Frequency']))

                linear_fit_y = slope * np.array(sub_feature_table.loc[:,'Output_Duration']) + intercept

                r2 = r2_score(sub_feature_table.loc[:,'Response_Fail_Frequency'], linear_fit_y)
                
                table = pd.DataFrame([slope, intercept, r2]).T
                table.columns = ['Slope', 'Intercept', 'R2']
                table = table.T
                table = table.reset_index(drop=False)
                table.columns = ["Response_Fail_Frequency", " "]
                return table
    
    
    
    @reactive.Effect
    def _():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            cell_fit_table = cell_dict['Cell_fit_table']
            response_type = input.Firing_response_type_detail()
            #cell_fit_table = cells[6]
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
            sub_cell_fit_table = sub_cell_fit_table[sub_cell_fit_table["I_O_obs"].isin(['Hill','Hill-Sigmoid'])]
            time_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
    
    
            ui.update_select(
                "Firing_output_duration_detail",
                label="Select output duration",
                choices=time_output_duration,
                
            )
            ui.update_select(
                "Firing_output_duration_new",
                label="Select output duration",
                choices=time_output_duration,
                
            )
            
    
    @reactive.Calc
    @reactive.event(input.Update_cell_btn, input.Firing_response_type_detail,input.Firing_output_duration_detail)
    def get_IO_fit_plots():
        
        if input.Firing_output_duration_detail() :
            cell_dict=get_cell_tables()
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            sweep_QC_table = cell_dict['Sweep_QC_table']
            
            
            
            response_type = input.Firing_response_type_detail()
            response_duration = input.Firing_output_duration_detail()
            
            stim_freq_table = fir_an.get_stim_freq_table(Full_SF_table, Sweep_info_table,sweep_QC_table,float(response_duration), response_based=response_type)

            plot_lists = fir_an.fit_IO_relationship(
                stim_freq_table, do_plot=True)
            
            return plot_lists
    
                
    
    def render_plot_func(plot_list, j):
        @render.plot
        def f():
            plot = plot_list[j]
            
            return plot
        return f
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Plot_new, input.Firing_output_duration_new)
    def IO_fit_parameter_table():
        if input.JSON_config_file() and input.Cell_file_select() and input.Plot_new():
            print('JIFIferifjoreijforlRIIFIR')
            cell_dict=get_cell_tables()
            Cell_feature_table = cell_dict['Cell_feature_table']
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            Sweep_QC_table = cell_dict['Sweep_QC_table']
            
            Firing_response_type_new = input.Firing_response_type_new()
            if Firing_response_type_new=='Time_based':
                
                Firing_output_duration_new = float(input.Firing_output_duration_new())
            else:
                Firing_output_duration_new = int(float(input.Firing_output_duration_new()))
            

            
            Test_Gain = Cell_feature_table.loc[(Cell_feature_table['Response_type']==Firing_response_type_new)&
                                                            (Cell_feature_table['Output_Duration']==Firing_output_duration_new), 'Gain'].values[0]
            
            
            if not np.isnan(Test_Gain):
                stim_freq_table = fir_an.get_stim_freq_table(Full_SF_table, Sweep_info_table, Sweep_QC_table, Firing_output_duration_new, Firing_response_type_new)
                parameters_table = fir_an.get_IO_features_NEW_TEST(stim_freq_table, Firing_response_type_new, Firing_output_duration_new, do_plot=False)[-1]
                
                if "-Polynomial_fit" in input.Plot_new():
                    sub_parameter_table = parameters_table.loc[parameters_table['Fit']=='3rd_order_polynomial_fit',:]
                    
                elif '-Final_Hill_Sigmoid_Fit' in input.Plot_new():
                    sub_parameter_table = parameters_table.loc[parameters_table['Fit']=='Final fit to data',:]
                    
                elif "-IO_fit" in input.Plot_new():
                    sub_parameter_table = parameters_table
            return sub_parameter_table
                    

    @output
    @render.ui
    def plots():
        if input.Firing_output_duration_detail() :
            plot_output_list = []
            plot_list = get_IO_fit_plots()
            counter = 0
            for i in plot_list:

                plotname = f"IO_details_plot_{counter}"
    
                plot_output_list.append(ui.output_plot(plotname))
                output(render_plot_func(plot_list, i), id=plotname)
                counter += 1
            return ui.TagList(plot_output_list)
        
    
        
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Plot_new, input.Firing_output_duration_new)
    def Time_based_IO_plotly_new():
        if input.JSON_config_file() and input.Cell_file_select() and input.Plot_new():

            cell_dict=get_cell_tables()
            Cell_feature_table = cell_dict['Cell_feature_table']
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            Sweep_QC_table = cell_dict['Sweep_QC_table']
            
            Firing_response_type_new = input.Firing_response_type_new()
            if Firing_response_type_new=='Time_based':
                
                Firing_output_duration_new = float(input.Firing_output_duration_new())
            else:
                Firing_output_duration_new = int(float(input.Firing_output_duration_new()))
            

            
            Test_Gain = Cell_feature_table.loc[(Cell_feature_table['Response_type']==Firing_response_type_new)&
                                                            (Cell_feature_table['Output_Duration']==Firing_output_duration_new), 'Gain'].values[0]
            
            
            if not np.isnan(Test_Gain):
                stim_freq_table = fir_an.get_stim_freq_table(Full_SF_table, Sweep_info_table, Sweep_QC_table, Firing_output_duration_new, Firing_response_type_new)
                plot_list = fir_an.get_IO_features_NEW_TEST(stim_freq_table, Firing_response_type_new, Firing_output_duration_new, do_plot=True)
                
                # )
                if "-Polynomial_fit" in input.Plot_new():
                
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
                        legend_title='Legend',
                        width=1000,  # Set plot width
                        height=800,# Set plot height
                        template="plotly_white"
                    )
                elif '-Final_Hill_Sigmoid_Fit' in input.Plot_new():
                    
                    Hill_Sigmoid_fit_dict = plot_list['2-Final_Hill_Sigmoid_Fit']
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
                        legend_title='Legend',
                        width=1000,  # Set plot width
                        height=800,# Set plot height
                        template="plotly_white"
                    )
                    
                elif "-IO_fit" in input.Plot_new():
                    
                    feature_plot_dict = plot_list['3-IO_fit']
                    
                    # Extract data from the dictionary
                    original_data_table = feature_plot_dict['original_data_table']
                    model_table = feature_plot_dict['model_table']
                    gain_table = feature_plot_dict['gain_table']
                    Intercept = feature_plot_dict['intercept']
                    Gain = feature_plot_dict['Gain']
                    Threshold_table = feature_plot_dict['Threshold']
                    stimulus_for_max_freq = feature_plot_dict["stimulus_for_maximum_frequency"]
                    
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
                    
                    # Add model line
                    fig.add_trace(go.Scatter(x=gain_table['Stim_amp_pA'], 
                                              y=gain_table['Frequency_Hz'], 
                                              mode='lines', 
                                              name='Linear IO portion',
                                              line=dict(color='green', 
                                                        width = 6,
                                                        dash='dot')))
                    
                    # Add the abline (slope and intercept)
                    x_range = np.arange(original_data_table['Stim_amp_pA'].min(), stimulus_for_max_freq, 1)

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
                                              marker=dict(color='green', size = 10, symbol='cross')))
                    
                    # Add Saturation points if not NaN
                    if "Saturation" in feature_plot_dict.keys():
                        Saturation_table = feature_plot_dict['Saturation']
                        fig.add_trace(go.Scatter(x=Saturation_table['Stim_amp_pA'], 
                                                  y=Saturation_table['Frequency_Hz'], 
                                                  mode='markers', 
                                                  name='Saturation', 
                                                  marker=dict(color='green', size = 10, symbol='triangle-up')))
                    
                    if "Response_Failure" in feature_plot_dict.keys():
                        response_failure_table = feature_plot_dict['Response_Failure']
                        fig.add_trace(go.Scatter(x=response_failure_table['Stim_amp_pA'], 
                                                  y=response_failure_table['Frequency_Hz'], 
                                                  mode='markers', 
                                                  name='Response Failure', 
                                                  marker=dict(color='red', size = 10, symbol='x')))
                    
                    fig.update_layout(
                        title='I/O fit',
                        xaxis_title='Stim_amp_pA',
                        yaxis_title='Frequency_Hz',
                        width=1000,  # Set plot width
                        height=800,# Set plot height
                        template="plotly_white"
                    )
                


                # Add a custom button to download the figure as a PDF
                
                return fig
            
            
    # @render.download(filename="Firing_plot.pdf")
    # def Save_Detail_Firing_plot():
    #     # Another way to implement a file download is by yielding bytes; either all at
    #     # once, like in this case, or by yielding multiple times. When using this
    #     # approach, you should pass a filename argument to @render.download, which
    #     # determines what the browser will name the downloaded file.
    #     print('coucouc')
    #     cell_dict=get_cell_tables()
    #     Firing_response_type_new = input.Firing_response_type_new()
    #     plot = get_firing_analysis(cell_dict,Firing_response_type_new)
    #     with io.BytesIO() as buf:
    #         plot.write_image(buf, format='pdf',width=1300, height=900)
    #         #plot.savefig(buf, format="png")
    #         yield buf.getvalue()

                
                
    
            
            
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def BE_plotly():
        if input.Cell_file_select() :
            cell_dict=get_cell_tables()
            Full_TVC_table = cell_dict['Full_TVC_table']
            Full_SF_table = cell_dict['Full_SF_table']
            sweep_info_table = cell_dict['Sweep_info_table']
            Sweep_QC_table = cell_dict['Sweep_QC_table']
            if np.isnan(sweep_info_table.loc[sweep_info_table['Sweep']==input.Sweep_selected(),"Bridge_Error_GOhms"].values[0]) == False:
                TVC_table = Full_TVC_table.loc[input.Sweep_selected(),"TVC"]
                #TVC_table = ordifunc.get_filtered_TVC_table(Full_TVC_table, input.Sweep_selected(), do_filter=False, filter=5., do_plot=False)
                stim_amp = sweep_info_table.loc[sweep_info_table['Sweep']==input.Sweep_selected(),"Stim_SS_pA"].values[0]
                Stim_start_s = sweep_info_table.loc[sweep_info_table['Sweep']==input.Sweep_selected(),"Stim_start_s"].values[0]
                Stim_end_s = sweep_info_table.loc[sweep_info_table['Sweep']==input.Sweep_selected(),"Stim_end_s"].values[0]
                dict_plot = sw_an.estimate_bridge_error_test(TVC_table, stim_amp, Stim_start_s, Stim_end_s,do_plot=True)

                TVC_table = dict_plot['TVC_table']
                #min_time_fit
                min_time_current = dict_plot['min_time_current']
                max_time_current = dict_plot['max_time_current']
                    # Transition_time
                actual_transition_time = dict_plot["Transition_time"]
                alpha_FT = dict_plot['alpha_FT']
                T_FT = dict_plot['T_FT']
                delta_t_ref_first_positive = dict_plot["delta_t_ref_first_positive"]
                T_ref_cell = dict_plot['T_ref_cell']
                
                
                # Membrane_potential_mV
                V_fit_table = dict_plot["Membrane_potential_mV"]["V_fit_table"]
                V_table_to_fit = dict_plot['Membrane_potential_mV']['V_table_to_fit']
                V_pre_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Pre_transition"]
                V_post_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Post_transition"]
                
                # Input_current_pA
                pre_T_current = dict_plot["Input_current_pA"]["pre_T_current"]
                post_T_current = dict_plot["Input_current_pA"]["post_T_current"]
                current_trace_table = dict_plot["Input_current_pA"]["current_trace_table"]
                
                # Create subplots
                fig = make_subplots(rows=5, cols=1,  shared_xaxes=True, subplot_titles=("Membrane Potential plot", "Input Current plot", "Membrane potential first derivative 1kHz LPF","Membrane potential second derivative 5kHz LPF","Input current derivative 5kHz LPF"), vertical_spacing=0.03)
                
                # Membrane potential plot
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Membrane_potential_mV'], mode='lines', name='Cell_trace', line=dict(color='black', width =1 )), row=1, col=1)
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Membrane_potential_0_5_LPF'], mode='lines', name="Membrane_potential_0_5_LPF", line=dict(color='#680396')), row=1, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[V_pre_transition_time], mode='markers', name='Voltage_Pre_transition', marker=dict(color='#fa2df0', size=8)), row=1, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[V_post_transition_time], mode='markers', name='Voltage_Post_transition', marker=dict(color='#9e2102', size=8)), row=1, col=1)
                fig.add_trace(go.Scatter(x=V_table_to_fit['Time_s'], y=V_table_to_fit['Membrane_potential_mV'], mode='lines', line=dict(color='#c96a04'), name='Fitted membrane potential'), row=1, col=1)
                fig.add_trace(go.Scatter(x=V_fit_table['Time_s'], y=V_fit_table['Membrane_potential_mV'], mode='lines', line=dict(color='orange'), name='Post_transition_voltage_fit'), row=1, col=1)
                
                
                
                # Input current plot
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Input_current_pA'], mode='lines', name='Input_current_trace', line=dict(color='black', width =1 )), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(min_time_current+0.005))&(current_trace_table["Time_s"]>=(min_time_current)),"Time_s"]), 
                    y=[pre_T_current]*len(np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(min_time_current+0.005))&(current_trace_table["Time_s"]>=(min_time_current)),"Time_s"])), 
                    mode='lines', name="Pre transition fit Median", line=dict(color="red")), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(max_time_current))&(current_trace_table["Time_s"]>=(max_time_current-0.005)),"Time_s"]), 
                    y=[post_T_current]*len(np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(max_time_current))&(current_trace_table["Time_s"]>=(max_time_current-0.005)),"Time_s"])), 
                    mode='lines', name="Post transition fit Median", line=dict(color='blue')), row=2, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[pre_T_current], mode='markers', name='pre_Transition_current', marker=dict(color='red', size = 8)), row=2, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[post_T_current], mode='markers', name='post_Transition_current', marker=dict(color='blue', size=8)), row=2, col=1)
                
                # V_dot_one_kHz
                first_sign_change = delta_t_ref_first_positive+T_ref_cell
                min_V_dot = np.nanmin(TVC_table['V_dot_one_kHz'])
                max_V_dot = np.nanmax(TVC_table['V_dot_one_kHz'])

                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['V_dot_one_kHz'], mode='lines', name='V_dot_one_kHz', line=dict(color='black', width = 1)), row=3, col=1)
                fig.add_trace(go.Scatter(x=[first_sign_change]*len(np.arange(min_V_dot,max_V_dot,1)), y=np.arange(min_V_dot,max_V_dot,1),
                                         mode='lines', name=f'first_sign_change = {round(first_sign_change,4)}s <br> delta_t_ref_first_positive = {round(delta_t_ref_first_positive,4)}s ', line=dict(color='blueviolet', dash='dash')), row=3, col=1)
                # Add dashed vertical line at time = T_star
                fig.add_trace(go.Scatter(x=[T_ref_cell]*len(np.arange(min_V_dot,max_V_dot,1)), y=np.arange(min_V_dot,max_V_dot,1),
                                         mode='lines', name=f'T_ref_cell = {round(T_ref_cell,4)}s', line=dict(color='red', dash='dash')), row=3, col=1)
                T_start_fit = np.nanmin(V_table_to_fit['Time_s'])
                fig.add_trace(go.Scatter(x=[T_start_fit]*len(np.arange(min_V_dot,max_V_dot,1)), y=np.arange(min_V_dot,max_V_dot,1),
                                         mode='lines', name=f'T_start_fit = {round(T_start_fit,4)}s', line=dict(color='#c96a04', dash='dash')), row=3, col=1)
                
                
                # V_double_dot_5KHz plot
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['V_double_dot_five_kHz'], mode='lines', name='V_double_dot_five_kHz', line=dict(color='black', width = 1)), row=4, col=1)
                # Add dashed vertical line at time = T_star
                Fast_ring_time = np.array(TVC_table.loc[(TVC_table["Time_s"]<=(actual_transition_time+.005))&(TVC_table["Time_s"]>=actual_transition_time),'Time_s'])
                max_V_double_dot_five_kHz = np.nanmax(TVC_table['V_double_dot_five_kHz'])
                min_V_double_dot_five_kHz = np.nanmin(TVC_table['V_double_dot_five_kHz'])
                fig.add_trace(go.Scatter(x=Fast_ring_time, y=[alpha_FT]*len(Fast_ring_time),
                                         mode='lines', name=f'alpha_FT = {round(alpha_FT,4)} mV/s/s', line=dict(color='darkred', dash='dash')), row=4, col=1)
                fig.add_trace(go.Scatter(x=Fast_ring_time, y=[-alpha_FT]*len(Fast_ring_time),
                                         mode='lines', name=f'-alpha_FT = {round(-alpha_FT,4)} mV/s/s', line=dict(color='darkred', dash='dash')), row=4, col=1)
                if not np.isnan(T_FT):
                    fig.add_trace(go.Scatter(x=[T_FT]*len(np.arange(min_V_double_dot_five_kHz, max_V_double_dot_five_kHz, 1.)), y=np.arange(min_V_double_dot_five_kHz, max_V_double_dot_five_kHz, 1.),
                                             mode='lines', name=f'T_FT = {round(T_FT,4)}s', line=dict(color='darkred', dash='dash')), row=4, col=1)

                # I_dot_5_kHz plot
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['I_dot_five_kHz'], mode='lines', name='I_dot_5_kHz', line=dict(color='black', width = 1)), row=5, col=1)
                
                # Add dashed vertical line at time = T_star
                fig.add_trace(go.Scatter(x=[actual_transition_time, actual_transition_time], y=[TVC_table['I_dot_five_kHz'].min(), TVC_table['I_dot_five_kHz'].max()],
                                         mode='lines', name=f'T_star = {round(actual_transition_time,4)}s', line=dict(color='red', dash='dash')), row=5, col=1)
                

                # Update layout
                fig.update_layout(

                    height=1200,
                    showlegend=True,
                    
                )
                
                # Update x and y axes

                fig.update_yaxes(title_text="mV", row=1, col=1)
                fig.update_xaxes(title_text="Time s", row=5, col=1)
                fig.update_yaxes(title_text="pA", row=2, col=1)
                fig.update_yaxes(title_text="mV/ms", row=3, col=1)
                fig.update_yaxes(title_text="mV/ms/ms", row=4, col=1)
                fig.update_yaxes(title_text="pA/ms", row=5, col=1)

                return fig
            
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def BE_conditions_table():
        if input.Cell_file_select() :
            cell_dict=get_cell_tables()
            condition_table = get_BE_value_and_table(cell_dict, input.Sweep_selected())[1]
            condition_table_styled = condition_table.style.set_table_attributes('class="dataframe shiny-table table w-auto"').apply(highlight_late)
            return condition_table_styled
        
    @output
    @render.text
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def BE_value():
        if input.Cell_file_select() :
            cell_dict=get_cell_tables()
            BE_val = get_BE_value_and_table(cell_dict, input.Sweep_selected())[0]
            
            return f"Estimated Bridge error = {round(BE_val, 5)} MOhms"
            
                
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Adaptation_feature_to_display)
    def Adaptation_plotly():
        if input.Cell_file_select() :
            cell_dict=get_cell_tables()
            Adaptation_table = cell_dict['Cell_Adaptation']
            Full_SF_table = cell_dict['Full_SF_table']
            sweep_info_table = cell_dict['Sweep_info_table']
            Sweep_QC_table = cell_dict['Sweep_QC_table']
            
            current_adaptation_measure = Adaptation_table.loc[Adaptation_table['Feature'] == input.Adaptation_feature_to_display(), "Measure" ].values[0]
            print(f"current_adaptation_measure=FZEOFRKE: {current_adaptation_measure}")
            
            interval_based_feature = fir_an.collect_interval_based_features_test(Full_SF_table, sweep_info_table, Sweep_QC_table, 0.5, input.Adaptation_feature_to_display(), current_adaptation_measure)
            
            plot_dict = fir_an.fit_adaptation_test(interval_based_feature, do_plot=True)
            
            ####
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

            
            fig.update_layout(xaxis_title='Spike Interval', yaxis_title='Normalized Feature', title=f"Adaptation of {input.Adaptation_feature_to_display()} ({current_adaptation_measure})")
            
            return fig
    
    @output
    @render.table
    def Adaptation_table():
        if input.Sweep_selected() :
            cell_dict=get_cell_tables()
            Adaptation_table = cell_dict['Cell_Adaptation']
            
            
            return Adaptation_table
        
    @reactive.Calc
    @reactive.event(input.Update_cell_btn, input.x_Spike_feature, input.y_Spike_feature, input.y_Spike_feature_index)
    def get_spike_feature_index():
        if input.Cell_file_select() :
            
            cell_dict=get_cell_tables()
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            Full_TVC_table = cell_dict['Full_TVC_table']
            
            
            Spike_feature_x = input.x_Spike_feature()
            
            Spike_feature_y = input.y_Spike_feature()
            Spike_index_y = input.y_Spike_feature_index()

            First_Spike_feature_col = f'{str(Spike_feature_x)}_mV_Spike_N'
            Second_Spike_feature_col = f'{str(Spike_feature_y)}_mV_Spike_{input.y_Spike_feature_index()}'
            
            spike_feature_table=pd.DataFrame(columns=[First_Spike_feature_col, Second_Spike_feature_col, "Spike_index", "Sweep", "Stim_amp_pA"])

            Sweep_info_table = Sweep_info_table.sort_values(by=['Stim_amp_pA'])
            sweep_list = Sweep_info_table.loc[:,"Sweep"].unique()
            stim_amp_list = Sweep_info_table.loc[:,"Stim_amp_pA"].unique()

            for sweep, stim_amp in zip(sweep_list,stim_amp_list):
                sub_SF_table = Full_SF_table.loc[sweep,'SF']

                if sub_SF_table.shape[0]==0:
                    continue
                
                current_first_feature_table = sub_SF_table.loc[sub_SF_table['Feature']==Spike_feature_x,:]
                
                
                current_second_feature_table = sub_SF_table.loc[sub_SF_table['Feature']==Spike_feature_y,:]
                
                if Spike_feature_x == 'Trough' and Spike_index_y=="N+1":
                    sub_TVC = ordifunc.get_filtered_TVC_table(Full_TVC_table, sweep)
                    stim_start_time = Sweep_info_table.loc[sweep, "Stim_start_s"]
                    First_spike_time = list(sub_SF_table.loc[sub_SF_table['Feature']=="Threshold",'Time_s'])[0]
                    
                    minimum_voltage_before_first_spike = np.nanmin(np.array(sub_TVC.loc[(sub_TVC['Time_s']>=stim_start_time)&(sub_TVC['Time_s']<First_spike_time),"Membrane_potential_mV"]))
                    new_spike_feature_line = pd.DataFrame([np.nan,minimum_voltage_before_first_spike, np.nan, np.nan, np.nan, 'Trough', -1 ]).T
                    new_spike_feature_line.columns = current_first_feature_table.columns
                    
                    current_first_feature_table = pd.concat([new_spike_feature_line,current_first_feature_table], ignore_index=True)
                
                current_first_feature_table.index = current_first_feature_table.loc[:,"Spike_index"]
                current_first_feature_table.index = current_first_feature_table.index.astype(int)
                
                current_second_feature_table.index = current_second_feature_table.loc[:,"Spike_index"]
                current_second_feature_table.index = current_second_feature_table.index.astype(int)
                
                
                Spike_index_list = current_first_feature_table.loc[:,'Spike_index'].unique()
                for spike_index in Spike_index_list:
                    if input.y_Spike_feature_index() == 'N-1':
                        spike_index_second = spike_index-1
                    elif input.y_Spike_feature_index() == 'N+1':
                        spike_index_second = spike_index+1
                    else:
                        spike_index_second = spike_index
                    second_spike_features_index_list = current_second_feature_table.loc[:,"Spike_index"].unique()
                    
                    if spike_index_second not in second_spike_features_index_list:
                        continue
                    
                    
                    first_spike_feature = current_first_feature_table.loc[spike_index, 'Membrane_potential_mV']
                    Second_spike_feature = current_second_feature_table.loc[spike_index_second, 'Membrane_potential_mV']
                    
                    new_line = pd.DataFrame([first_spike_feature, Second_spike_feature, spike_index, sweep, stim_amp]).T
                    new_line.columns = spike_feature_table.columns
                    spike_feature_table = pd.concat([spike_feature_table, new_line], ignore_index=True)
                    
            spike_feature_table = spike_feature_table.astype({First_Spike_feature_col:'float',Second_Spike_feature_col:'float','Spike_index':'int', 'Stim_amp_pA':'float'})
                
            
            
            return spike_feature_table
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.x_Spike_feature, input.y_Spike_feature, input.y_Spike_feature_index)
    def Spike_features_index_correlation():
        
        if input.Cell_file_select() :
            
            Spike_feature_x = input.x_Spike_feature()
            
            Spike_feature_y = input.y_Spike_feature()
            Spike_index_y = input.y_Spike_feature_index()
            
           
            spike_feature_table = get_spike_feature_index()
            
            First_Spike_feature_col = f'{str(Spike_feature_x)}_mV_Spike_N'
            Second_Spike_feature_col = f'{str(Spike_feature_y)}_mV_Spike_{input.y_Spike_feature_index()}'
            
            
            
            fig = px.scatter(spike_feature_table, 
                              x=First_Spike_feature_col, 
                              y=Second_Spike_feature_col, 
                              color='Stim_amp_pA',
                              hover_data=['Sweep',"Spike_index"])
            fig.update_layout(
                width=800, height=500)
    
            return fig
    
    
                
                
            
app = App(app_ui, server)