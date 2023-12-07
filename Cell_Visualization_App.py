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

from sklearn.metrics import r2_score 

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
                                  ui.row(ui.column(11,
                                                   ui.output_plot(
                                                       "Sweep_spike_plot")),
                                         ui.column(1,
                                                   ui.input_checkbox_group("Spike_Feature_to_display", "Select Spike Features to Display",
                                                                            choices = ["Threshold","Upstroke", "Peak",'Downstroke','Fast_Trough','fAHP','Slow_Trough','Trough'],
                                                                            selected=["Threshold","Upstroke", "Peak",'Downstroke','Fast_Trough','fAHP','Slow_Trough','Trough']
                                                                            ))),
                                  
                                  ui.row(ui.column(3,
                                                    ui.input_checkbox('use_custom_x_axis', 'Custom Time Limits')),
                                          ui.column(3,
                                                    ui.input_numeric("SF_plot_min",'Minimum',1),
                                                    ui.input_numeric("SF_plot_max",'Maximum',2)),
                                          ui.column(3,
                                                    ui.input_select("Spike_feature_representation", "Feature representation", choices=['Point','Line'],selected = 'Point'),
                                                    ),
                                          ui.column(3,
                                                   
                                                    ui.input_action_button("Update_plot_parameters",'Update'))),
                                  
                                  
                                  
                                  
                                  #ui.input_action_button("Sweep_spike_plot_modal", "Plot parameters"),
                                  ui.output_data_frame("Spike_feature_table")),
                           ui.nav('Bridge Error analysis',
                                  ui.row(ui.column(6,
                                                   ui.output_plot('BE_voltage_start')),
                                         ui.column(6,
                                                   ui.output_plot('BE_voltage_end'))),
                                  ui.row(ui.column(6,
                                                   ui.output_plot('BE_current_start')),
                                         ui.column(6,
                                                   ui.output_plot('BE_current_end')))
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
                               output_widget("Time_based_IO_plotly",width=300,height=800),
                               
                               
                               ui.row(
                                   ui.column(3,ui.tags.b("Gain"),output_widget("Gain_regression_plot_time_based",width=30,height=30)),
                                   ui.column(3,ui.tags.b("Threshold"),output_widget("Threshold_regression_plot_time_based",width=30,height=30)),
                                   ui.column(3,ui.tags.b("Saturation Frequency"),output_widget("Saturation_freq_regression_plot_time_based",width=30,height=30)),
                                   ui.column(3,ui.tags.b("Saturation Stimulus"),output_widget("Saturation_stim_regression_plot_time_based",width=30,height=30))
                                   ),
                               
                               ui.row(
                                   ui.column(3,ui.output_table("Gain_regression_table_time_based")),
                                   ui.column(3,ui.output_table("Threshold_regression_table_time_based")),
                                   ui.column(3,ui.output_table("Saturation_freq_regression_table_time_based")),
                                   ui.column(3,ui.output_table("Saturation_stim_regression_table_time_based"))
                                   )
                               ),
                        ui.nav("Details",
                               ui.row(ui.input_select('Firing_response_type_detail','Select response type',choices=["Time_based","Index_based","Interval_based"]),
                                      ui.input_select('Firing_output_duration_detail','Select response type',choices=[])),
                               ui.output_ui("plots")
                               
                               )
                        ),
                    width=12
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
    
def get_firing_analysis(cells,response_type):
    
    cell_fit_table = cells[6]
    sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
    time_output_duration = sub_cell_fit_table.loc[:,'Output_Duration'].unique()
    Full_stim_freq_table_list=[]
    model_table_list = []
    IO_table_list = []
    Saturation_table_list = []

    for output_duration in time_output_duration:
        
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

        Gain_reg_plot,Gain_reg_table = get_feature_regression_plot_tables(cells,response_type,'Gain')
        Threshold_reg_plot,Threshold_reg_table = get_feature_regression_plot_tables(cells,response_type,'Threshold')
        Sat_freq_reg_plot,Sat_freq_reg_table = get_feature_regression_plot_tables(cells,response_type,'Saturation_Frequency')
        Saturation_stim_reg_plot,Saturation_stim_reg_table = get_feature_regression_plot_tables(cells,response_type,'Saturation_Stimulus')
        
        return IO_figure,Gain_reg_plot,Gain_reg_table,Threshold_reg_plot,Threshold_reg_table,Sat_freq_reg_plot,Sat_freq_reg_table,Saturation_stim_reg_plot,Saturation_stim_reg_table

def get_feature_regression_plot_tables(cells,response_type,feature):
    
    
    cell_feature_table = cells[7]
    

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
            cells = get_cell_tables()
            Sweep_info_table = cells[4]
            sweep_list = list(Sweep_info_table.loc[:,"Sweep"])
            ui.update_selectize("Sweep_selected" ,choices=sweep_list)
            cell_fit_table = cells[6]
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == 'Time_based',:]
            time_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
            print(time_output_duration)
            ui.update_select('Firing_output_duration_Time_based', choices=time_output_duration)
            
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == 'Index_based',:]
            index_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
            ui.update_select('Firing_output_duration_Index_based', choices=index_output_duration)
            
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == 'Interval_based',:]
            interval_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
            ui.update_select('Firing_output_duration_Interval_based', choices=interval_output_duration)
    
    
        
    
    @reactive.Calc
    @reactive.event(input.Update_cell_btn)
    def get_cell_tables():

        cell_id = input.Cell_file_select()
        config_json_input = input.JSON_config_file()
        if config_json_input and cell_id :
            config_table = get_config_json_table()
            saving_folder = config_table.loc[0,"path_to_saving_file"]
            cell_file_path = str(saving_folder+str(cell_id))
            
            Metadata_table = ordifunc.read_cell_file_h5(cell_file_path, "--",selection=["Metadata"])[3]
            Database = Metadata_table['Database'].values[0]
            
            print(config_table)
            print(Database)
            current_config_line = config_table[config_table['database_name'] == Database]
            print('ijij')
            print(current_config_line)
            
            Full_TVC_table, Full_SF_dict_table, Full_SF_table, Metadata_table, sweep_info_table, Sweep_QC_table, cell_fit_table, cell_feature_table,Processing_report_df = ordifunc.read_cell_file_h5(cell_file_path, current_config_line,selection=["All"])
            return Full_TVC_table, Full_SF_dict_table, Full_SF_table, Metadata_table, sweep_info_table, Sweep_QC_table, cell_fit_table, cell_feature_table,Processing_report_df
        
        
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

                
            cells = get_cell_tables()
                
            processing_time_table = cells[-1]
            return processing_time_table
        
        
    @output
    @render.plot
    @reactive.event(input.Update_cell_btn)
    def cell_input_resistance_bxp():
        if input.Cell_file_select() :
            cells = get_cell_tables()
            Sweep_info_table = cells[4]
            IR_boxplot = p9.ggplot(Sweep_info_table,p9.aes(x='0',y='Input_Resistance_GOhms'))+p9.geom_boxplot()+p9.geom_jitter()
            return IR_boxplot
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn)
    def cell_input_resistance_plolty():
        if input.Cell_file_select() :
            cells = get_cell_tables()
            Sweep_info_table = cells[4]
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
            cells = get_cell_tables()
            Sweep_info_table = cells[4]
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
            cells = get_cell_tables()
            Sweep_info_table = cells[4]
            time_cst_boxplot = p9.ggplot(Sweep_info_table,p9.aes(x='0',y='Time_constant_ms'))+p9.geom_boxplot()+p9.geom_jitter()
            return time_cst_boxplot
        
    
    
        
        
    @output
    @render.data_frame()
    @reactive.event(input.Update_cell_btn)
    def Sweep_info_table():
        if input.Cell_file_select() :
            cells = get_cell_tables()
            Sweep_info_table = cells[4]
            return Sweep_info_table
    
   
    
   
   
    @output
    @render.plot()
    @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction,input.Superimpose_BE_Correction,input.Update_plot_parameters)
    def Sweep_spike_plot():
        if input.Sweep_selected() :

            cells = get_cell_tables()
            current_sweep = input.Sweep_selected()
            Full_TVC_table = cells[0]
            Full_SF_table = cells[2]
            Sweep_info_table = cells[4]
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
                Full_SF_dict_table = cells[1]
                Full_SF_table = sp_an.create_Full_SF_table(Full_TVC_table, Full_SF_dict_table, correct_bridge_resistance=True, cell_sweep_info_table = Sweep_info_table)
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
            

            SF_plot = p9.ggplot(New_TVC_Table,p9.aes(x='Time_s',y='Value',color='Legend'))+p9.geom_line()+p9.scale_color_manual(color_dict)
            if current_SF_table.shape[0]>=1:
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
                if input.Spike_feature_representation() == 'Point':
                    SF_plot+=p9.geom_point(current_SF_table_voltage,p9.aes(x='Time_s',y='Value',color='Feature'))
                elif input.Spike_feature_representation() == 'Line':
                    SF_plot+=p9.geom_line(current_SF_table_voltage,p9.aes(x='Time_s',y='Value',color='Feature'))

            SF_plot += p9.facet_grid('Plot ~ .',scales='free')
            

            return SF_plot

    
        
    @output
    @render.data_frame()
    @reactive.event(input.Update_cell_btn,input.Sweep_selected,input.BE_correction,input.Update_plot_parameters)
    def Spike_feature_table():
        if input.Sweep_selected() :

            cells = get_cell_tables()
            current_sweep = input.Sweep_selected()
            Full_TVC_table = cells[0]
            
            Sweep_info_table = cells[4]
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
                Full_SF_dict_table = cells[1]
                original_TVC_table = current_TVC_table.copy()
                original_TVC_table.loc[:,'Membrane_potential_mV'] -= BE*original_TVC_table.loc[:,'Input_current_pA']
                Full_SF_table = sp_an.create_Full_SF_table(Full_TVC_table, Full_SF_dict_table, correct_bridge_resistance=True, cell_sweep_info_table = Sweep_info_table)
                current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']
            else:
                Full_SF_table = cells[2]
                current_SF_table = Full_SF_table.loc[str(current_sweep),'SF']

            return current_SF_table
    
    
    @reactive.Calc
    @reactive.event(input.Update_cell_btn,input.Sweep_selected)
    def get_sweep_linear_properties():
        if input.Sweep_selected() :
            cells = get_cell_tables()
            current_sweep = input.Sweep_selected()
            Full_TVC_table = cells[0]
        
            Sweep_info_table = cells[4]
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
    def linear_properties_conditions_table():
        if input.Sweep_selected() :
            
            
            condition_table = get_sweep_linear_properties()[0]
            condition_table_styled = condition_table.style.set_table_attributes('class="dataframe shiny-table table w-auto"').apply(highlight_late)
            return condition_table_styled
                
    @output
    @render.plot()
    def linear_properties_fit():
        if input.Sweep_selected() :
            cells = get_cell_tables()
            current_sweep = input.Sweep_selected()
            

            Sweep_info_table = cells[4]
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
            cells = get_cell_tables()
            current_sweep = input.Sweep_selected()
            Full_TVC_table = cells[0]
            TVC_table = ordifunc.get_filtered_TVC_table(Full_TVC_table, current_sweep, do_filter=True, filter=5., do_plot=False)
            Sweep_info_table = cells[4]
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
            cells = get_cell_tables()
            Full_SF_table = cells[2]
            Sweep_info_table = cells[4]
            sweep_QC_table = cells[5]
            cell_fit_table = cells[6]
            cell_feature_table = cells[7]
            
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
            
            cells=get_cell_tables()
            
            IO_figure,Gain_reg_plot,Gain_reg_table,Threshold_reg_plot,Threshold_reg_table,Sat_freq_reg_plot,Sat_freq_reg_table,Saturation_stim_reg_plot,Saturation_stim_reg_table = get_firing_analysis(cells, input.Firing_response_type())
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
            
            cells=get_cell_tables()
            
            response_type = input.Firing_response_type_detail()
            cell_fit_table = cells[6]
            sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
            sub_cell_fit_table = sub_cell_fit_table[sub_cell_fit_table["I_O_obs"].isin(['Hill','Hill-Sigmoid'])]
            time_output_duration = list(sub_cell_fit_table.loc[:,'Output_Duration'].unique())
    
    
            ui.update_select(
                "Firing_output_duration_detail",
                label="Select output duration",
                choices=time_output_duration,
                
            )
    
    @reactive.Calc
    @reactive.event(input.Update_cell_btn, input.Firing_response_type_detail,input.Firing_output_duration_detail)
    def get_IO_fit_plots():
        
        if input.Firing_output_duration_detail() :
            cells=get_cell_tables()
            Full_SF_table = cells[2]
            Sweep_info_table = cells[4]
            sweep_QC_table = cells[5]
            
            
            response_type = input.Firing_response_type_detail()
            response_duration = input.Firing_output_duration_detail()
            
            stim_freq_table = fir_an.get_stim_freq_table(Full_SF_table, Sweep_info_table,sweep_QC_table,np.float(response_duration), response_based=response_type)

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
    
                
                
            
app = App(app_ui, server)