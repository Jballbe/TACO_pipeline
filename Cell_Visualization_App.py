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

from shiny import App, Inputs, Outputs, Session, render, ui,reactive




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

from sklearn.metrics import r2_score 
import sys
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots
from sklearn.preprocessing import minmax_scale


# In fast mode, throttle interval in ms.
FAST_INTERACT_INTERVAL = 60

app_ui = ui.page_fluid(
    ui.panel_title("Cell Visualization App"),
    ui.layout_sidebar(
        
        
        
        ui.panel_sidebar(
            ui.input_file("JSON_config_file", 'Choose JSON configuration files'),
            ui.input_selectize("Cell_file_select","Select Cell to Analyse",choices=[]),
            ui.input_action_button('Update_cell_btn','Update Cell'),
            
            
        ),
        ui.panel_main(
            ui.output_table('config_json_table'),
        ),
    ),
    
    
    ui.navset_tab_card(
        ui.nav("Cell Information",
               ui.layout_sidebar(
                   ui.panel_sidebar(width=0),
                   ui.panel_main(
                       ui.navset_pill_card(
                           ui.nav('Processing report', 
                                  ui.output_table('cell_processing_time_table')),
                           ui.nav("Sweep information",
                                  ui.output_data_frame('Sweep_info_table'),
                                  ),
                           
                                  
                           ui.nav('Linear properties',
                                  
                                  ui.row(
                                      ui.column(6,ui.output_plot("cell_input_resistance_bxp"),
                                                output_widget("cell_input_resistance_plolty")),
                                      ui.column(6,ui.output_plot("cell_time_cst_bxp"),
                                                output_widget('cell_time_constant_plolty'))
                                  ),
                           ),
                       
                       
                   ),
               ))),

        ui.nav("Sweep Analysis",
               ui.layout_sidebar(
                   ui.panel_sidebar(
                                    ui.input_selectize("Sweep_selected","Select Sweep to Analyse",choices=[]),
                                    ui.input_checkbox("BE_correction", "Correct for Bridge Error"),
                                    ui.input_checkbox("Superimpose_BE_Correction", "Superimpose BE Correction"),
                                
                                    width=2),
                   ui.panel_main(
                       ui.navset_tab(
                           ui.nav('Spike Analysis', 
                                  #ui.output_plot("Sweep_spike_plot", click=True, dblclick=True, hover=True, brush=True),
                                  #output_widget('Sweep_spike_plotly'),
                                  output_widget('Sweep_spike_plotly'),                                  
                                  
                                  
                                  ui.row(ui.column(6,
                                                   output_widget("Spike_superposition_plot",width=30,height=30)),
                                         ui.column(6,
                                                   output_widget("Spike_phase_plane_plot",width=30,height=30))),
                                  
                                  
                                  #ui.input_action_button("Sweep_spike_plot_modal", "Plot parameters"),
                                  ui.output_data_frame("Spike_feature_table")),
                           ui.nav('Sweep_QC',
                                  ui.output_table('Sweep_QC_table')),
                           ui.nav('Bridge Error analysis',
                                  
                                  output_widget("BE_plotly")
                                  
                                  ),
                           ui.nav('Linear properties analysis',
                                  ui.output_table("linear_properties_conditions_table"),
                                  ui.output_plot("linear_properties_fit"),
                                  ui.output_table('linear_properties_fit_table'),
                                  ui.output_table("linear_properties_table")
                                  ),
                           ),
                       
                      
                   ),
               )),

        ui.nav("Firing Analysis",
            ui.layout_sidebar(
                ui.panel_sidebar(width=0),
                ui.panel_main(
                    ui.navset_pill(
                        ui.nav("I/O",
                               ui.input_select('Firing_response_type', "Select response type", choices=['Time_based','Index_based','Interval_based']),
                               output_widget("Time_based_IO_plotly"),
                               
                               
                               ui.row(
                                   ui.column(3,ui.tags.b("Gain"),output_widget("Gain_regression_plot_time_based")),
                                   ui.column(3,ui.tags.b("Threshold"),output_widget("Threshold_regression_plot_time_based")),
                                   ui.column(3,ui.tags.b("Saturation Frequency"),output_widget("Saturation_freq_regression_plot_time_based")),
                                   ui.column(3,ui.tags.b("Saturation Stimulus"),output_widget("Saturation_stim_regression_plot_time_based"))
                                   ),
                               
                               ui.row(
                                   ui.column(3,ui.output_table("Gain_regression_table_time_based")),
                                   ui.column(3,ui.output_table("Threshold_regression_table_time_based")),
                                   ui.column(3,ui.output_table("Saturation_freq_regression_table_time_based")),
                                   ui.column(3,ui.output_table("Saturation_stim_regression_table_time_based"))
                                   )
                               ),
                        
                        ui.nav("I/O_new",
                               ui.row(ui.input_select('Firing_response_type_new','Select response type',choices=["Time_based","Index_based","Interval_based"]),
                                      ui.input_select('Firing_output_duration_new','Select response type',choices=[])),
                               
                               ui.input_selectize('Plot_new', "Select plot to display", choices=[]),
                               output_widget("Time_based_IO_plotly_new"),
                               
                               
                               
                               ),
                        ui.nav("Details",
                               ui.row(ui.input_select('Firing_response_type_detail','Select response type',choices=["Time_based","Index_based","Interval_based"]),
                                      ui.input_select('Firing_output_duration_detail','Select response type',choices=[])),
                               ui.output_ui("plots")
                               
                               ),
                        ui.nav("Adaptation",
                               ui.input_select('Adaptation_feature_to_display', "Select feature ", choices=[]),
                               output_widget("Adaptation_plotly"),
                               ui.output_table("Adaptation_table"),
                               
                               
                               )
                        ),
                    width=12
                    )
                ),
                
            
        ),
        
        ui.nav("In progress",
            ui.layout_sidebar(
                ui.panel_sidebar(
                                 ui.input_selectize("Sweep_selected_in_progress","Select Sweep to Analyse",choices=[]),
                                 ui.input_selectize("x_Spike_feature","Select spike feature X axis",choices=['Downstroke','fAHP','Fast_Trough','Peak','Spike_heigth','Spike_width_at_half_heigth','Threshold','Trough','Upstroke']),
                                 ui.input_selectize("y_Spike_feature","Select spike feature Y axis",choices=["Peak",'Downstroke','fAHP','Fast_Trough','Spike_heigth','Spike_width_at_half_heigth','Threshold','Trough','Upstroke']),
                                 ui.input_selectize("y_Spike_feature_index","Which spike index for Y axis",choices=['N',"N-1","N+1"]),
                                 ui.input_checkbox("BE_correction_in_process", "Correct for Bridge Error"),
                                 
                             
                                 width=2),
                ui.panel_main(
                    ui.navset_pill(
                        ui.nav("Spike feature correlation",
                               
                               output_widget("Spike_features_index_correlation"),
                               
                               
                               
                               
                               ),
                        
                        ),
                    width=10
                    )
                ),
                
            
        ),
        
        
        )
    
    
)



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
    
def get_firing_analysis(cell_dict,response_type):
    
    cell_fit_table = cell_dict["Cell_fit_table"]
    sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
    time_output_duration = sub_cell_fit_table.loc[:,'Output_Duration'].unique()
    Full_stim_freq_table_list=[]
    model_table_list = []
    IO_table_list = []
    Saturation_table_list = []

    for output_duration in time_output_duration:
        
        Full_SF_table = cell_dict["Full_SF_table"]
        Sweep_info_table = cell_dict['Sweep_info_table']
        sweep_QC_table = cell_dict['Sweep_QC_table']
        sweep_QC_table = cell_dict['Sweep_QC_table']
        cell_feature_table = cell_dict['Cell_feature_table']
        
        
        response_type=response_type.replace(' ','_')
        
        sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
        sub_cell_feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == response_type,:]

        
        
        
        if response_type == 'Time_based':
            response = str(str(int(output_duration*1e3))+'ms')
        else:
            output_duration=int(output_duration)
            response = str(response_type.replace('_based',' ') + str(int(output_duration)))
        
        current_stim_freq_table = fir_an.get_stim_freq_table(
            Full_SF_table.copy(), Sweep_info_table.copy(),sweep_QC_table.copy(), output_duration,response_type)
        current_stim_freq_table['Response'] = response
       
        model = sub_cell_fit_table.loc[sub_cell_fit_table['Output_Duration'] == output_duration,'I_O_obs'].values[0]
        Full_stim_freq_table_list.append(current_stim_freq_table)
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
                                            'Response' : response},index=[0])
                
            else:
                Saturation_table = pd.DataFrame()
        
        
            
            model_table_list.append(model_table)
            IO_table_list.append(IO_table)
            if Saturation_table.shape[0]>0:
                Saturation_table_list.append(Saturation_table)
    
    
    Full_stim_freq_table = pd.concat([*Full_stim_freq_table_list],axis=0,ignore_index=True)
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
                                   color_discrete_map=color_dict)
        if Full_Saturation_table.shape[0]>0:
            Sat_plot = px.line(Full_Saturation_table,x='Stim_amp_pA', y='Frequency_Hz',color="Response",
                                       color_discrete_map=color_dict)
            IO_figure = go.Figure(data=scatter_plot.data + line_plot.data + IO_plot.data + Sat_plot.data)
        else:
            IO_figure = go.Figure(data=scatter_plot.data + line_plot.data + IO_plot.data)
        IO_figure.update_layout(
            autosize=False,
            width=1200,
            height=800,
            xaxis_title="Input Current (pA)", 
            yaxis_title="Frequency (Hz)")

        Gain_reg_plot,Gain_reg_table = get_feature_regression_plot_tables(cell_dict,response_type,'Gain')
        Threshold_reg_plot,Threshold_reg_table = get_feature_regression_plot_tables(cell_dict,response_type,'Threshold')
        Sat_freq_reg_plot,Sat_freq_reg_table = get_feature_regression_plot_tables(cell_dict,response_type,'Saturation_Frequency')
        Saturation_stim_reg_plot,Saturation_stim_reg_table = get_feature_regression_plot_tables(cell_dict,response_type,'Saturation_Stimulus')
        
        return IO_figure,Gain_reg_plot,Gain_reg_table,Threshold_reg_plot,Threshold_reg_table,Sat_freq_reg_plot,Sat_freq_reg_table,Saturation_stim_reg_plot,Saturation_stim_reg_table

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
    print('OKFKFROROFKFKROFRKO')
    print(table)
    return figure,table


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





def server(input: Inputs, output: Outputs, session: Session):
    print("coucou")

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
                plot_list = fir_an.extract_IO_features_Test(stim_freq_table, Firing_response_type_new, Firing_output_duration_new, do_plot=True)
                

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
            print('herer',cell_id)
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
    @render.plot
    @reactive.event(input.Update_cell_btn)
    def cell_input_resistance_bxp():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            #Sweep_info_table = cells[4]
            IR_boxplot = p9.ggplot(Sweep_info_table,p9.aes(x='0',y='Input_Resistance_GOhms'))+p9.geom_boxplot()+p9.geom_jitter()
            return IR_boxplot
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_input_resistance_plolty():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            #Sweep_info_table = cells[4]
            line_plot = px.line(Sweep_info_table,x='Stim_amp_pA',y='Input_Resistance_GOhms',line_group='Protocol_id',color='Protocol_id')
            figure = go.Figure(data=line_plot.data )
            figure.update_layout(
                autosize=True,
                
                xaxis_title="Stimulus amplitude pA", 
                yaxis_title="GOhms")
        
            return figure
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_time_constant_plolty():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            line_plot = px.line(Sweep_info_table,x='Stim_amp_pA',y='Time_constant_ms',color='Protocol_id')
            figure = go.Figure(data=line_plot.data )
            figure.update_layout(
                autosize=True,
                
                xaxis_title="Stimulus amplitude pA", 
                yaxis_title="ms")
        
            return figure
        
    @output
    @render.plot
    @reactive.event(input.Update_cell_btn)
    def cell_time_cst_bxp():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            time_cst_boxplot = p9.ggplot(Sweep_info_table,p9.aes(x='0',y='Time_constant_ms'))+p9.geom_boxplot()+p9.geom_jitter()
            return time_cst_boxplot
        
    
    
        
        
    @output
    @render.data_frame
    @reactive.event(input.Update_cell_btn)
    def Sweep_info_table():
        if input.Cell_file_select() :
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            return Sweep_info_table
    
   
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
                fig.add_trace(go.Scatter(x=current_TVC_table['Time_s'], y=current_TVC_table['Membrane_potential_mV'], mode='lines', name='Membrane potential BE Corrected', line=dict(color=color_dict['Membrane potential'])), row=1, col=1)
                if input.Superimpose_BE_Correction() == True:
                    current_TVC_table_second = ordifunc.get_filtered_TVC_table(Full_TVC_table,current_sweep)
                    fig.add_trace(go.Scatter(x=current_TVC_table_second['Time_s'], y=current_TVC_table_second['Membrane_potential_mV'], mode='lines', name='Membrane potential original', line=dict(color=color_dict['Membrane potential original'])), row=1, col=1)
            
            
            else: 
                fig.add_trace(go.Scatter(x=current_TVC_table['Time_s'], y=current_TVC_table['Membrane_potential_mV'], mode='lines', name='Membrane potential ', line=dict(color=color_dict['Membrane potential'])), row=1, col=1)
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
            fig.add_trace(go.Scatter(x=current_TVC_table['Time_s'], y=current_TVC_table['Input_current_pA'], mode='lines', name='Input current', line=dict(color=color_dict['Input_current_pA'])), row=2, col=1)
            
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
            
                    
   
   
    @output
    @render.plot()
    @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction,input.Superimpose_BE_Correction,input.Update_plot_parameters)
    def Sweep_spike_plot():
        if input.Sweep_selected() :

            cell_dict = get_cell_tables()
            Full_TVC_table = cell_dict['Full_TVC_table']
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict["Sweep_info_table"]
            current_sweep = input.Sweep_selected()
            
            current_TVC_table = ordifunc.get_filtered_TVC_table(Full_TVC_table,current_sweep)
            
            if input.use_custom_x_axis() == False:
            
                stim_start_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_start_s'].values[0]
                stim_start_time -= .1
                stim_end_time = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_end_s'].values[0]
                stim_end_time += .1
            else:
                stim_start_time = input.SF_plot_min()
                stim_end_time = input.SF_plot_max()
                
            
            current_TVC_table = current_TVC_table.loc[current_TVC_table['Time_s'] >= (stim_start_time), :]
            current_TVC_table = current_TVC_table.loc[current_TVC_table['Time_s'] <= (stim_end_time), :]
            

            

            if input.BE_correction():
                
                BE = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Bridge_Error_GOhms'].values[0]
                #Full_SF_dict_table = cells[1]
                Full_SF_dict_table = cell_dict['Full_SF_dict_table']
                Full_SF_table = sp_an.create_Full_SF_table(Full_TVC_table, Full_SF_dict_table, cell_sweep_info_table = Sweep_info_table,BE_correct=True,)
                current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']
                
                if input.Superimpose_BE_Correction() :
                    original_TVC_table = current_TVC_table.copy()
                    original_TVC_table.loc[:,'Membrane_potential_mV'] -= BE*original_TVC_table.loc[:,'Input_current_pA']
                    BE_voltage_table = original_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
                    BE_voltage_table.loc[:,'Legend']="Membrane potential BE Corrected"
                    BE_voltage_table.columns = ['Time_s', 'Value','Legend']
                    BE_voltage_table['Plot'] = 'Membrane_potential_mV'
                    
                    Potential_table = current_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
                    Potential_table.loc[:,'Legend']="Membrane potential original"
                    Potential_table.columns = ['Time_s', 'Value','Legend']
                    Potential_table['Plot'] = 'Membrane_potential_mV'
                    
                else:
                    current_TVC_table.loc[:,'Membrane_potential_mV'] -= BE*current_TVC_table.loc[:,'Input_current_pA']
                    Potential_table = current_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
                    Potential_table.loc[:,'Legend']="Membrane potential BE Corrected"
                    Potential_table.columns = ['Time_s', 'Value','Legend']
                    Potential_table['Plot'] = 'Membrane_potential_mV'

            else:

                current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']
                Potential_table = current_TVC_table.loc[:,['Time_s','Membrane_potential_mV']]
                Potential_table.loc[:,'Legend']="Membrane potential original"
                Potential_table.columns = ['Time_s', 'Value','Legend']
                Potential_table['Plot'] = 'Membrane_potential_mV'
            
            
            
            Current_table = current_TVC_table.loc[:,['Time_s','Input_current_pA']]
            Current_table.loc[:,'Legend']="Input current"
            Current_table.columns = ['Time_s', 'Value','Legend']
            Current_table['Plot'] = 'Input_current_pA'
            
            
            
            New_TVC_Table = pd.concat([Potential_table,Current_table], axis = 0, ignore_index = True)
            if input.Superimpose_BE_Correction() and input.BE_correction() :
                New_TVC_Table = pd.concat([New_TVC_Table,BE_voltage_table], axis = 0, ignore_index = True)
            New_TVC_Table['Plot'] = New_TVC_Table['Plot'].astype('category')
            New_TVC_Table['Plot'] = New_TVC_Table['Plot'].cat.reorder_categories(['Membrane_potential_mV', 'Input_current_pA'])
            color_dict = {'Input_current_pA' : 'black',
                          'Membrane potential original' : 'black',
                          'Membrane potential BE Corrected' : "red",
                          'Threshold':"#a2c5fc",
                          "Upstroke" : "#0562f5", 
                          "Peak" : "#2f0387",
                          "Downstroke":"#9f02e8",
                          "Fast_Trough" : "#c248fa",
                          "fAHP":"#d991fa",
                          "Trough":'#fc3873',
                          "Slow_Trough" :  "#96022e"

                }
            
            New_TVC_Table = New_TVC_Table.astype({'Time_s':'float','Value':'float'})
            SF_plot = p9.ggplot(New_TVC_Table,p9.aes(x='Time_s',y='Value',color='Legend'))+p9.geom_line()+p9.scale_color_manual(color_dict)
            if current_SF_table.shape[0]>=1:
                current_SF_table = current_SF_table.loc[(current_SF_table['Feature']!='Spike_width_at_half_heigth')&(current_SF_table['Feature']!='Spike_heigth'),:]
                current_SF_table_voltage = current_SF_table.loc[:,['Time_s','Membrane_potential_mV','Feature']]
                current_SF_table_voltage.columns=['Time_s','Value','Feature']
                current_SF_table_voltage['Plot'] = 'Membrane_potential_mV'
                
                current_SF_table_stim = current_SF_table.loc[:,['Time_s','Input_current_pA','Feature']]
                current_SF_table_stim.columns=['Time_s','Value','Feature']
                current_SF_table_stim['Plot'] = 'Input_current_pA'
                
                current_SF_table_voltage = pd.concat([current_SF_table_voltage,current_SF_table_stim], axis = 0, ignore_index = True)

                current_SF_table_voltage['Plot'] = current_SF_table_voltage['Plot'].astype('category')
                current_SF_table_voltage['Plot'] = current_SF_table_voltage['Plot'].cat.reorder_categories(['Membrane_potential_mV', 'Input_current_pA'])
                
                
                current_SF_table_voltage = current_SF_table_voltage.loc[current_SF_table_voltage['Time_s'] >= (stim_start_time), :]
                current_SF_table_voltage = current_SF_table_voltage.loc[current_SF_table_voltage['Time_s'] <= (stim_end_time), :]
                current_SF_table_voltage = current_SF_table_voltage[current_SF_table_voltage['Feature'].isin(input.Spike_Feature_to_display())]
                current_SF_table_voltage=current_SF_table_voltage.astype({'Time_s':'float','Value':'float'})
                if input.Spike_feature_representation() == 'Point':
                    SF_plot+=p9.geom_point(current_SF_table_voltage,p9.aes(x='Time_s',y='Value',color='Feature'))
                elif input.Spike_feature_representation() == 'Line':
                    SF_plot+=p9.geom_line(current_SF_table_voltage,p9.aes(x='Time_s',y='Value',color='Feature'))

            SF_plot += p9.facet_grid('Plot ~ .',scales='free')
            

            return SF_plot

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
        sub_spike_super_position_table = spike_trace_table.loc[spike_trace_table['Sweep']==input.Sweep_selected(),:]
        discrete_colors = []
        if sub_spike_super_position_table.shape[0]==0:
            current_spike_index_list = []
        else:
            sub_spike_super_position_table = sub_spike_super_position_table.astype({'Spike_index' : 'category'})
            current_spike_index_list = [str(x) for x in range(sub_spike_super_position_table['Spike_index'].astype(int).min(),
                                         sub_spike_super_position_table['Spike_index'].astype(int).max()+1)]
        
            discrete_colors = sample_colorscale('plasma_r', minmax_scale(current_spike_index_list))
        for x in range(spike_trace_table['Spike_index'].astype(int).min(),spike_trace_table['Spike_index'].astype(int).max()+1):
            if x not in sub_spike_super_position_table.loc[:,'Spike_index']:
                discrete_colors.append('rgb(173, 173, 173)')
        new_discrete_colors=[]
        colored_spike_index_list=[]
        uncolored_spike_index_list = []
        i=0
        spike_trace_table=spike_trace_table.sort_values(by=['Spike_index'])
        color_dict = {}
        full_spike_index_list = [str(x) for x in range(spike_trace_table['Spike_index'].astype(int).min(),
                                         spike_trace_table['Spike_index'].astype(int).max()+1)]
        for x in full_spike_index_list:
            if x not in current_spike_index_list:
                new_discrete_colors.append('rgb(173, 173, 173)')
                color_dict[x]='rgb(173, 173, 173)'
                uncolored_spike_index_list.append(x)
            else:
                colored_spike_index_list.append(x)
        for x in colored_spike_index_list:
            color_dict[x]=discrete_colors[i]
            i+=1
        full_spike_index_list_ordered = uncolored_spike_index_list+colored_spike_index_list
        spike_trace_table = spike_trace_table.sort_values(by=["Spike_index",'Time_s'])
        fig = px.line(spike_trace_table,
                      x="Membrane_potential_mV",
                      y="Potential_first_time_derivative_mV/s",
                      color='Spike_index',
                      category_orders={'Spike_index':full_spike_index_list_ordered},
                      color_discrete_map= color_dict)
        
        return fig
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def Spike_superposition_plot():
        spike_trace_table = spike_analysis_plot()
        sub_spike_super_position_table = spike_trace_table.loc[spike_trace_table['Sweep']==input.Sweep_selected(),:]
        discrete_colors = []
        if sub_spike_super_position_table.shape[0]==0:
            current_spike_index_list = []
        else:
            sub_spike_super_position_table = sub_spike_super_position_table.astype({'Spike_index' : 'category'})
            current_spike_index_list = [str(x) for x in range(sub_spike_super_position_table['Spike_index'].astype(int).min(),
                                         sub_spike_super_position_table['Spike_index'].astype(int).max()+1)]
        
            discrete_colors = sample_colorscale('plasma_r', minmax_scale(current_spike_index_list))
        for x in range(spike_trace_table['Spike_index'].astype(int).min(),spike_trace_table['Spike_index'].astype(int).max()+1):
            if x not in sub_spike_super_position_table.loc[:,'Spike_index']:
                discrete_colors.append('rgb(173, 173, 173)')
        new_discrete_colors=[]
        colored_spike_index_list=[]
        uncolored_spike_index_list = []
        i=0
        spike_trace_table=spike_trace_table.sort_values(by=['Spike_index'])
        color_dict = {}
        full_spike_index_list = [str(x) for x in range(spike_trace_table['Spike_index'].astype(int).min(),
                                         spike_trace_table['Spike_index'].astype(int).max()+1)]
        for x in full_spike_index_list:
            if x not in current_spike_index_list:
                new_discrete_colors.append('rgb(173, 173, 173)')
                color_dict[x]='rgb(173, 173, 173)'
                uncolored_spike_index_list.append(x)
            else:
                colored_spike_index_list.append(x)
        for x in colored_spike_index_list:
            color_dict[x]=discrete_colors[i]
            i+=1
        full_spike_index_list_ordered = uncolored_spike_index_list+colored_spike_index_list
        spike_trace_table = spike_trace_table.sort_values(by=["Spike_index",'Time_s'])
        fig = px.line(spike_trace_table,
                      x="Time_s",
                      y="Membrane_potential_mV",
                      color='Spike_index',
                      category_orders={'Spike_index':full_spike_index_list_ordered},
                      color_discrete_map= color_dict)
        
        return fig
    
    
    @output
    @render.data_frame
    @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction,input.Update_plot_parameters)
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
                Sweep_info_table.loc[current_sweep,"Bridge_Error_extrapolated"] ==True or
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
                    
                    
                    
                    R_in=((SS_potential-holding_potential)/stim_amp)-BE
                    
                    time_cst=best_tau*1e3 #convert s to ms
                    resting_potential = holding_potential - (R_in+BE)*stim_amp
            
                    
            linear_fit_plot=sw_an.fit_membrane_time_cst(current_TVC_table, stim_start_time, (stim_start_time+.300),do_plot=True)
                    
            condition_table=pd.DataFrame([is_spike,
                                          np.abs(Sweep_info_table.loc[current_sweep,"Stim_amp_pA"]),
                                          Sweep_info_table.loc[current_sweep,"Bridge_Error_extrapolated"],
                                          np.abs(CV_pre_stim),
                                          np.abs(CV_second_half),
                                          NRMSE]).T
            condition_table.columns=['Potential spike',"Stimulus_amplitude",'BE extrapolated','CV pre stim','CV second half','NRMSE']
            condition_to_meet = pd.DataFrame(["No spike","|| ≤2pA", "Not extrapolated", '|| < 0.01','|| <0.01', '< 0.3']).T
            condition_to_meet.columns=['Potential spike',"Stimulus_amplitude",'BE extrapolated','CV pre stim','CV second half','NRMSE']
            
            stim_amp_cond_respected = False if np.abs(Sweep_info_table.loc[current_sweep,"Stim_amp_pA"])<2.0 else True
            BE_extra_cond_respected = False if Sweep_info_table.loc[current_sweep,"Bridge_Error_extrapolated"] ==True else True   
            CV_pre_stim_cond_respected = False if np.abs(CV_pre_stim) > 0.01 else True 
            CV_second_half_cond_respected = False if np.abs(CV_second_half) > 0.01 else True
            NRMSE_cond_respected = False if np.isnan(NRMSE)==True else False if NRMSE > 0.3 else True
            
            condition_respected = pd.DataFrame([not is_spike, stim_amp_cond_respected, BE_extra_cond_respected, CV_pre_stim_cond_respected, CV_second_half_cond_respected, NRMSE_cond_respected]).T
            condition_respected.columns=['Potential spike',"Stimulus_amplitude",'BE extrapolated','CV pre stim','CV second half','NRMSE']

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
            Full_TVC_table = cell_dict['Full_TVC_table']
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
                
                current_stim_freq_table = fir_an.get_stim_freq_table(
                    Full_SF_table.copy(), Sweep_info_table.copy(),sweep_QC_table.copy(), current_response_duration,response_type)
                current_stim_freq_table['Response'] = response
                stim_freq_table_list.append(current_stim_freq_table)
                
                model = sub_cell_fit_table.loc[sub_cell_fit_table['Output_Duration'] == current_response_duration,'I_O_obs'].values[0]
                if model == 'Hill' or model == 'Hill-Sigmoid':
                    min_stim = np.nanmin(current_stim_freq_table.loc[:,'Stim_amp_pA'])
                    max_stim = np.nanmax(current_stim_freq_table.loc[:,'Stim_amp_pA'])
                    stim_array = np.arange(min_stim,max_stim,.1)
                    Hill_Half_cst = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == current_response_duration,'Hill_Half_cst'].values[0]
                    Hill_amplitude = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == current_response_duration,'Hill_amplitude'].values[0]
                    Hill_coef = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == current_response_duration,'Hill_coef'].values[0]
                    Hill_x0 = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == current_response_duration,'Hill_x0'].values[0]
                    freq_array = fir_an.hill_function(stim_array, Hill_x0, Hill_coef, Hill_Half_cst)
                    freq_array *= Hill_amplitude
                    
                    if model == 'Hill-Sigmoid':
                        sigmoid_sigma = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == current_response_duration,'Sigmoid_sigma'].values[0]
                        sigmoid_x0 = sub_cell_fit_table.loc[cell_fit_table['Output_Duration'] == current_response_duration,'Sigmoid_x0'].values[0]
                        
                        freq_array *= fir_an.sigmoid_function(stim_array , sigmoid_x0, sigmoid_sigma)
                    
                    model_table = pd.DataFrame({'Stim_amp_pA' : stim_array,
                                                "Frequency_Hz" : freq_array})
                    model_table['Response'] = response
                    Gain = sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Gain'].values[0]
                    Threshold = sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Threshold'].values[0]
                    Intercept = -Gain*Threshold
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
            
            IO_figure,Gain_reg_plot,Gain_reg_table,Threshold_reg_plot,Threshold_reg_table,Sat_freq_reg_plot,Sat_freq_reg_table,Saturation_stim_reg_plot,Saturation_stim_reg_table = get_firing_analysis(cell_dict, input.Firing_response_type())
            print('okokokookfokfrkoforkforkrkof')
            print(Gain_reg_plot)
            return IO_figure,Gain_reg_plot,Gain_reg_table,Threshold_reg_plot,Threshold_reg_table,Sat_freq_reg_plot,Sat_freq_reg_table,Saturation_stim_reg_plot,Saturation_stim_reg_table
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Time_based_IO_plotly():
        
        return Time_based_analysis()[0]
    
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Gain_regression_plot_time_based():

        return Time_based_analysis()[1]
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Gain_regression_table_time_based():
        
        return Time_based_analysis()[2]
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Threshold_regression_plot_time_based():
        
        return Time_based_analysis()[3]
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Threshold_regression_table_time_based():
        
        return Time_based_analysis()[4]
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_freq_regression_plot_time_based():
        
        return Time_based_analysis()[5]
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_freq_regression_table_time_based():
        
        return Time_based_analysis()[6]
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_stim_regression_plot_time_based():
        
        return Time_based_analysis()[7]
    
    @output
    @render.table
    @reactive.event(input.Update_cell_btn, input.Firing_response_type)
    def Saturation_stim_regression_table_time_based():
        
        return Time_based_analysis()[8]
    
    
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
        
    
    # @reactive.Calc
    # @reactive.event(input.Update_cell_btn, input.Firing_response_type_new,input.Firing_output_duration_new)
    # def get_IO_fit_plots_new():
        
    #     if input.Cell_file_select() :
    #         cell_dict=get_cell_tables()
    #         Cell_feature_table = cell_dict['Cell_feature_table']
    #         Full_SF_table = cell_dict['Full_SF_table']
    #         Sweep_info_table = cell_dict['Sweep_info_table']
    #         Sweep_QC_table = cell_dict['Sweep_QC_table']
            
    #         Firing_response_type_new = input.Firing_response_type_new()
    #         Firing_output_duration_new = .5
            

            
    #         Test_Gain = Cell_feature_table.loc[(Cell_feature_table['Response_type']==Firing_response_type_new)&
    #                                                         (Cell_feature_table['Output_Duration']==float(Firing_output_duration_new)), 'Gain'].values[0]
            
            
    #         if not np.isnan(Test_Gain):
    #             stim_freq_table = fir_an.get_stim_freq_table(Full_SF_table, Sweep_info_table, Sweep_QC_table, Firing_output_duration_new, Firing_response_type_new)
    #             plot_list = fir_an.extract_IO_features_Test(stim_freq_table, Firing_response_type_new, Firing_output_duration_new, do_plot=True)
    #             plot_dict = dict()
    #             for key in plot_list.keys():
    #                 sub_key = key[1:]
    #                 my_plot = plot_IO_fit(plot_list, sub_key, return_plot = False)
    #                 plot_dict[key] = my_plot

    #             #Update choice input 
    #             ui.input_selectize('Plot_new',label="Select plot to display", choices=list(plot_dict.keys()))
    #             return plot_dict
    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Plot_new)
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
                plot_list = fir_an.extract_IO_features_Test(stim_freq_table, Firing_response_type_new, Firing_output_duration_new, do_plot=True)
                return plot_IO_fit(plot_list, input.Plot_new(), True)
                # polynomial_fit_dict = plot_list["1-Polynomial_fit"]
                # # Extract data from dictionary
                # original_data_table = polynomial_fit_dict["original_data_table"]
                # trimmed_stimulus_frequency_table = polynomial_fit_dict['trimmed_stimulus_frequency_table']
                # trimmed_3rd_poly_table = polynomial_fit_dict['trimmed_3rd_poly_table']
                # color_shape_dict = polynomial_fit_dict['color_shape_dict']

                
                # passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
                # failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
                
                # fig = go.Figure()
                

                # fig.add_trace(go.Scatter(
                #     x=passed_QC_table['Stim_amp_pA'],
                #     y=passed_QC_table['Frequency_Hz'],
                #     mode='markers',
                #     marker=dict(symbol='circle'),
                #     name='Original',
                #     text=original_data_table['Sweep'],
                #     hoverinfo='text+x+y'
                # ))
                # fig.add_trace(go.Scatter(
                #     x=failed_QC_table['Stim_amp_pA'],
                #     y=failed_QC_table['Frequency_Hz'],
                #     mode='markers',
                #     marker=dict(symbol='circle-x-open'),
                #     name='Failed QC Data',
                #     text=original_data_table['Sweep'],
                #     hoverinfo='text+x+y'
                # ))
                
                # # Add trimmed stimulus frequency points
                # fig.add_trace(go.Scatter(
                #     x=trimmed_stimulus_frequency_table['Stim_amp_pA'],
                #     y=trimmed_stimulus_frequency_table['Frequency_Hz'],
                #     mode='markers',
                #     marker=dict(color='black'),
                #     name='Trimmed Stimulus Frequency',
                #     text=trimmed_stimulus_frequency_table['Passed_QC']
                # ))
                
                # # Add polynomial fit lines
                # for legend in trimmed_3rd_poly_table['Legend'].unique():
                #     legend_data = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Legend'] == legend]
                #     fig.add_trace(go.Scatter(
                #         x=legend_data['Stim_amp_pA'],
                #         y=legend_data['Frequency_Hz'],
                #         mode='lines',
                #         name=legend,
                #         line=dict(color=color_shape_dict[legend])
                #     ))
                
                # # Update layout
                # fig.update_layout(
                #     title='3rd order polynomial fit to trimmed_data',
                #     xaxis_title='Stim_amp_pA',
                #     yaxis_title='Frequency_Hz',
                #     legend_title='Legend'
                # )
                # return fig
                # #plot_dict[key] = my_plot
                
            
            
            
            
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
                stim_amp = sweep_info_table.loc[sweep_info_table['Sweep']==input.Sweep_selected(),"Stim_SS_pA"].values[0]
                Stim_start_s = sweep_info_table.loc[sweep_info_table['Sweep']==input.Sweep_selected(),"Stim_start_s"].values[0]
                Stim_end_s = sweep_info_table.loc[sweep_info_table['Sweep']==input.Sweep_selected(),"Stim_end_s"].values[0]
                dict_plot = sw_an.estimate_bridge_error_test(TVC_table, stim_amp, Stim_start_s, Stim_end_s,do_plot=True)
                
                #TVC_table
                TVC_table = dict_plot['TVC_table']
                #min_time_fit
                min_time_fit = dict_plot['min_time_fit']
                max_time_fit = dict_plot['max_time_fit']
                    # Transition_time
                actual_transition_time = dict_plot["Transition_time"]
                
                # Membrane_potential_mV
                V_fit_table = dict_plot["Membrane_potential_mV"]["V_fit_table"]
                V_pre_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Pre_transition"]
                V_post_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Post_transition"]
                
                # Input_current_pA
                pre_T_current = dict_plot["Input_current_pA"]["pre_T_current"]
                post_T_current = dict_plot["Input_current_pA"]["post_T_current"]
                sigmoid_fit_trace_table = dict_plot["Input_current_pA"]["Sigmoid_fit"]
                
                # Create subplots
                fig = make_subplots(rows=2, cols=1,  shared_xaxes=True, subplot_titles=("Membrane Potential plot", "Input Current plot"))
                
                # Membrane potential plot
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Membrane_potential_mV'], mode='lines', name='Cell_trace'), row=1, col=1)
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Membrane_potential_0_5_LPF'], mode='lines', name="Membrane_potential_0_5_LPF"), row=1, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[V_pre_transition_time], mode='markers', name='Voltage_Pre_transition', marker=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[V_post_transition_time], mode='markers', name='Voltage_Post_transition', marker=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=V_fit_table['Time_s'], y=V_fit_table['Membrane_potential_mV'], mode='lines', name='Post_transition_voltage_fit'), row=1, col=1)
                
                # Input current plot
                fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Input_current_pA'], mode='lines', name='Input_current_trace'), row=2, col=1)
                fig.add_trace(go.Scatter(x=sigmoid_fit_trace_table['Time_s'], y=sigmoid_fit_trace_table['Input_current_pA_Fit'], mode='lines', name="Sigmoid Fit To Current Trace"), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=np.array(sigmoid_fit_trace_table.loc[(sigmoid_fit_trace_table["Time_s"]<=(min_time_fit+0.005))&(sigmoid_fit_trace_table["Time_s"]>=(min_time_fit)),"Time_s"]), 
                    y=[pre_T_current]*len(np.array(sigmoid_fit_trace_table.loc[(sigmoid_fit_trace_table["Time_s"]<=(min_time_fit+0.005))&(sigmoid_fit_trace_table["Time_s"]>=(min_time_fit)),"Time_s"])), 
                    mode='lines', name="Pre transition fit Median", line=dict(color="red")), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=np.array(sigmoid_fit_trace_table.loc[(sigmoid_fit_trace_table["Time_s"]<=(max_time_fit))&(sigmoid_fit_trace_table["Time_s"]>=(max_time_fit-0.005)),"Time_s"]), 
                    y=[post_T_current]*len(np.array(sigmoid_fit_trace_table.loc[(sigmoid_fit_trace_table["Time_s"]<=(max_time_fit))&(sigmoid_fit_trace_table["Time_s"]>=(max_time_fit-0.005)),"Time_s"])), 
                    mode='lines', name="Post transition fit Median", line=dict(color='blue')), row=2, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[pre_T_current], mode='markers', name='pre_Transition_current', marker=dict(color='red')), row=2, col=1)
                fig.add_trace(go.Scatter(x=[actual_transition_time], y=[post_T_current], mode='markers', name='post_Transition_current', marker=dict(color='blue')), row=2, col=1)
                
                # Update layout
                fig.update_layout(
                    title_text="Plots",
                    height=800,
                    showlegend=True
                )
                
                # Update x and y axes
                fig.update_xaxes(title_text="Time_s", row=1, col=1)
                fig.update_yaxes(title_text="Membrane_potential_mV", row=1, col=1)
                fig.update_xaxes(title_text="Time_s", row=2, col=1)
                fig.update_yaxes(title_text="Input_current_pA", row=2, col=1)
                
                return fig
                
                
                
    
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