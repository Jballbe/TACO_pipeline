#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:00:50 2023

@author: julienballbe
"""

import json


if __name__ == "__main__":
    
    path_to_save_json_file = str(input('\u001b[1m \nIndicate path to save the configuration json file, indicating the file name and ending in .json \u001b[0m \n' ))
    
    path_to_saving_file = str(input('\u001b[1m \nIndicate path to folder in which analysis files will be saved \u001b[0m \n'))

    number_of_DB = int(input("\u001b[1m \nHow many DB do you have to process? \u001b[0m \n"))
    
    path_to_QC_file = str(input('\u001b[1m \nIndicate path to Quality Criteria python file(finishing by .py) containing the function assessing Quality Criteria \u001b[0m \n'))
    
    DB_parameters = []
    for db in range(number_of_DB):
        database_name = str(input('\u001b[1m \nEnter Database Name: \u001b[0m \n'))
        database_path_to_original_files = str(input('\u001b[1m \nEnter path to folder containing original files: \u001b[0m \n'))
        path_to_db_script_folder = str(input("\u001b[1m \nIndicate path to folder containning database-specific python script \u001b[0m \n"))
        python_file_name = str(input("\u001b[1m \nIndicate name of python file(finishing by .py) containing the function to get traces \u001b[0m \n"))
        db_function_name = str(input('\u001b[1m \nIndicate name of the function used \u001b[0m \n'))
        db_population_class_file = str(input('\u001b[1m \nIndicate path to database population_class csv file (.csv) \u001b[0m \n'))
        db_cell_sweep_csv_file = str(input('\u001b[1m \nIndicate path to database  cell-sweep_list csv file (.csv) \u001b[0m \n'))
        db_stimulus_duration = float(input('\u001b[1m \n Indicate stimulus duration (in s) \u001b[0m \n'))
        stimulus_time_provided = eval(input('\u001b[1m \n Are the stimulus start and end time provided for each sweep? Answer True or False. If True, then the database-dependant script must return after the time, potential and current traces the stimulus start time and the stimulus end time in second. If False, the stimulus start and end times will be determined by the database-independant script \u001b[0m \n'))
        
        
        db_list = {
            "database_name" : database_name,
            "original_file_directory" :database_path_to_original_files,
            "path_to_db_script_folder" : path_to_db_script_folder,
            "python_file_name" :python_file_name,
            'db_function_name':db_function_name,
            "db_population_class_file" : db_population_class_file,
            'db_cell_sweep_csv_file':db_cell_sweep_csv_file,
            "db_stimulus_duration" : db_stimulus_duration,
            "stimulus_time_provided": stimulus_time_provided}
        
        DB_parameters.append(db_list)
        
    configuration_file = {
        
        'path_to_saving_file' :path_to_saving_file,
        'path_to_QC_file' : path_to_QC_file,
        'DB_parameters':DB_parameters
        }
    
    # Serializing json
    configuration_file_json = json.dumps(configuration_file, indent=4)
     
    # Writing to sample.json
    with open(path_to_save_json_file, "w") as outfile:
        outfile.write(configuration_file_json)