# Welcome to TACO pipeline!

This python-based pipeline has been designed for the treatment and analysis of current-clamp recordings using long-currents steps protocols, in order to extract biophysical properties coherently across databases while minimizing experimentally induced variability.

## How does it work?
To ensure a coherent and database-independent analysis while minimizing the need for users to adapt their databases, the TACO pipeline extracts raw traces using a user-provided script and associated database information. These elements are specified in a `config_json_file`, which the pipeline helps the user generate (see How to use the TACO pipeline). The script and configuration are then automatically integrated into the pipeline, and the data proceed through the different analysis steps.  
For each cell in the database, the pipeline produces an `.h5` file containing the analysis results. These files can also be loaded back into the TACO pipeline for visual inspection.

## Defining some common terms
Before digging into the details of the pipeline functioning, it is important to start by giving a set of definitions used in the context of the pipeline and referring to different aspect of the current-clamp experiment.  
Notably we define a “trace” as an array of recorded values representing a unique modality of the experiment (i.e. voltage trace, or current trace).  
A “sweep”, is a set of corresponding voltage and current traces for a given cell. A sweep is referred to by a single identifier “sweep id” which is unique for a given cell but can be used for different cell.  
A “protocol” is constituted by a train of sweeps grouped together as they are done one after the other and have unique level of stimulation.  
Finally, an “experiment” refers to the ensemble of the protocols that have been performed on the cell.

## Pre-requisites
Before using the pipeline, it is highly recommended to create a separated python environment (https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)  
This environment must contain the following python package to enable the proper functioning of the pipeline:
- pandas
- numpy
- matplotlib
- scipy
- lmfit
- tqdm
- importlib
- json
- ast
- traceback
- concurrent
- shiny
- shiny widget
- scikit-learn
- plotly
- h5py
- re
- inspect
- plotnine
- anywidget

To help the user to configure the environment, it is possible to create it from the `TACO_pipeline_env.yml` file here provided. By following the instructions indicated here (https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) the user can automatically configure a fully functioning environment for the TACO pipeline.  
Once the pipeline is created, do not forget to activate the environment before using the TACO pipeline.

The pipeline is composed of 8 different scripts:
- `TACO_pipeline_App.py`
- `Sweep_QC_analysis.py`
- `Analysis_pipeline.py`
- `Ordinary_functions.py`
- `Sweep_analysis.py`
- `Spike_analysis.py`
- `Firing_analysis.py`
- `globals_module.py`

All scripts must be installed in a same directory.

## How to use the TACO pipeline
The pipeline is used through a Shiny-based GUI.  
To start the application, the user can follow the steps:
1. Open the Terminal
2. Start the environment with appropriate python packages
3. Go to the folder in which the different python scripts are installed
4. Type :  
   ```bash
   shiny run --reload --launch-browser TACO_pipeline_App.py
5. The application will automatically open in the web browser 

## Application panels

The app is composed of three main panels which help the user **prepare (1), run (2), and investigate (3) the analysis**:

### (1) Preparing the analysis

The TACO pipeline operates based on information specified in a **JSON configuration file**. To prepare this file, the user can rely on the first panel of the TACO app. In this panel, the user can enter multiple information regarding the overall analysis and rely on the app to ensure the paths and files indicated are correctly written (ensure paths/files exist):

- Full path to save the JSON configuration file (the path must contain the name of the JSON file to be created, e.g., `/Path/to/save/JSON_file.json`)
- Path of the folder in which cell-related analysis files will be saved (e.g., `/path/to/saving/folder/`)
- Path to the user-defined Python QC file (e.g., `/Path/to/save/User_defined_QC_file.py`)

For each database to analyze, the user must provide the following information:

- Name of the database
- Path to folder containing original files (e.g., `/Path/to/folder/with/original/files/`)
- Path to database-specific Python script (e.g., `/Path/to/database_python_script.py`)
- Path to database-specific population class table (e.g., `/Path/to/database/Population_class_table.csv`)
- Path to database-specific cell-sweep table (e.g., `/Path/to/database/Cell_Sweep_table.csv`)
- Whether the stimulus times are provided in the database (check the box if yes)
- Stimulus duration (in seconds)

Once all the information is provided, the user can click the **Add Database** button. If an error has been made in the information, the user can re-enter the details for the database; the JSON file will be automatically corrected.  
Once ready, the user can click the **Save JSON** button to save the JSON configuration file.

---

### (2) Running the analysis

To run the analysis, the user can go to the second panel **Run analysis**.  

The pipeline analyzes the data of multiple cells simultaneously using parallel processing. The user can choose the number of CPU cores to use; by default, the analysis will use half of the computer’s CPU cores.  

The user can then enter the path of the JSON configuration file (e.g., `/Path/to/save/JSON_file.json`), choose which analyses to perform (Spike, Sweep, Firing, or Metadata), and indicate whether to overwrite any existing analysis files.  

Once all information has been entered, the analysis can be launched by clicking the **Run analysis** button. Progress can be monitored via the progress bar displayed in the app. During the analysis, the user should **not close the Terminal or the application page in the browser**.  

After the analysis, the user can specify a path to save summary files (e.g., `/path/to/saving/folder/`). Tables summarizing fit parameters, firing properties, and linear properties will be saved at this location.

---

### (3) Visual inspection

To visually inspect the analysis, the user can go (after completion of the analysis) to the third panel **Cell Visualization**. In this panel, the user can select the JSON configuration file. The list of available cell files is generated based on the cells indicated in the different databases’ population class tables.

---

## Example use of the TACO pipeline

We provide a Test_folder.zip containing few example cells originating from **two open-access databases**:

- Da Silva Lantyer et al., 2018  
- Harrison et al., 2015  

The following files are provided:

- Original cell files  
- Database-specific scripts  
- Population and cell-sweep tables  
- A ready-to-use `config_json_file_test.json`  

You can directly use this file in the Analysis part of the pipeline.  

**Note:** Before using the test data, do not forget to update the `config_json_file_test.json` with correct file and folder paths.
