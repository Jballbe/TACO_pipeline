#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 19:05:50 2025

@author: julienballbe
"""
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

#%% IO detailed plot
def plot_IO_detailed_fit_plot_choice(IO_plot_dict,stim_freq_table = None, plot_type = "matplotlib", do_fit = True):
    
    if do_fit == True:
        if plot_type == "plotly":
            fig = plot_IO_detailed_fit_plotly(IO_plot_dict, do_fit, stim_freq_table)
            
        elif plot_type == "matplotlib":
            fig = plot_IO_detailed_fit_matplotlib(IO_plot_dict, return_plot = True)[0]
            
        
    else:
        
        if plot_type == "plotly":
            fig = plot_stim_freq_table_plotly(stim_freq_table)
            
        elif plot_type == "matplotlib":
            fig = plot_stim_freq_table_matplotlib(stim_freq_table, return_plot = True)[0]
        
    return fig
    
    
def plot_IO_detailed_fit_matplotlib(IO_plot_dict, return_plot=False, saving_path=""):
    """
    Create a single figure with all fits plotted on one axis (PLOS ONE–compliant design):
    - Original data (QC passed/failed)
    - Polynomial fit
    - Final Hill Sigmoid fit
    - IO fit
    """
    
    # --- Default style adapted to PLOS ONE figure design ---
    default_style = {
        "figsize": (6, 6),          # ~15 cm wide — fits single-column width at 300 dpi
        "fontsize": 12,             # axis titles and legend
        "tick_font_size": 11,       # ticks slightly smaller
        "line_width": 2,            # standard scientific thickness
        "marker_size": 60,          # ensures readability in print
        "font_family": "Arial",     # PLOS recommends sans-serif, readable font
        "grid": True,
        "legend_frame": False,
    
        # Color-blind–friendly Okabe–Ito palette
        "passed_color": "#0072B2",      # Blue
        "failed_color": "#EE6677",     # Red–orange
        "trimmed_color": "#00694d",     # Green
        "poly_colors": None,
        "initial_color": "#56B4E9",     # Sky blue
        "final_color": "#D55E00",       # Vermilion
        "model_color": "#E69F00",       # Blue
        "gain_color": "#d52b00",        # Green
        "linear_color": "#000000",      # Black
        "threshold_color": "#000000",   # Black
        "saturation_color": "#D55E00",  # Gray
        "failure_color": "#D55E00",     # Red–orange
    }
    
    
    s = default_style
    
    plt.rcParams.update({
        "font.family": s["font_family"],
        "mathtext.fontset": "dejavusans",   # Utilise DejaVu Sans pour les maths
        "mathtext.default": "regular",       # Évite les polices mathématiques spéciales
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plt.rcParams['text.usetex'] = False
    
    
        
    # --- Create single axis ---
    fig, ax = plt.subplots(figsize=s["figsize"])
    
    # ---------------- PANEL (Single): IO Fit ----------------
    original_data_table = IO_plot_dict["1-Polynomial_fit"]["original_data_table"]
    passed = original_data_table[original_data_table['Passed_QC']]
    failed = original_data_table[~original_data_table['Passed_QC']]
    
    poly_dict = IO_plot_dict["1-Polynomial_fit"]
    trimmed_stimulus_frequency_table = poly_dict['trimmed_stimulus_frequency_table']
    trimmed_3rd_poly_table = poly_dict['trimmed_3rd_poly_table']
    color_shape_dict_poly = poly_dict['color_shape_dict']
    
    hill_dict = IO_plot_dict['2-Final_Hill_Sigmoid_Fit']
    Initial_fit_color = hill_dict['Initial_Fit_table']
    color_shape_dict_hill = hill_dict['color_shape_dict']
    
    io_dict = IO_plot_dict['3-IO_fit']
    model_table = io_dict['model_table']
    gain_table = io_dict['gain_table']
    Intercept = io_dict['intercept']
    Gain = io_dict['Gain']
    Threshold_table = io_dict['Threshold']
    stimulus_for_max_freq = io_dict["stimulus_for_maximum_frequency"]
    maximum_frequency = np.nanmax(original_data_table.loc[:,'Frequency_Hz'])
    if "Saturation" in io_dict:
        sat = io_dict["Saturation"]
        maximum_linear_fit = np.nanmax([maximum_frequency, sat["Frequency_Hz"].iloc[0]])
    else:
        maximum_linear_fit = maximum_frequency
    
    # --- Plot all elements ---
    ax.scatter(
        passed['Stim_amp_pA'], passed['Frequency_Hz'],
        c=s["passed_color"], s=s["marker_size"], label="QC passed", alpha=0.8
    )
    ax.scatter(
        failed['Stim_amp_pA'], failed['Frequency_Hz'],
        c=s["failed_color"], s=s["marker_size"], label="QC failed", alpha=0.8
    )
    ax.scatter(
        trimmed_stimulus_frequency_table['Stim_amp_pA'],
        trimmed_stimulus_frequency_table['Frequency_Hz'],
        c=s["trimmed_color"], s=s["marker_size"] , label="Trimmed data", alpha=0.8
    )
    
    for legend in trimmed_3rd_poly_table['Legend'].unique():
        label_legend = "Polynomial fit" if legend == "3rd_order_poly" else legend
        legend_data = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Legend'] == legend]
        ax.plot(
            legend_data['Stim_amp_pA'], legend_data['Frequency_Hz'],
            color=color_shape_dict_poly.get(legend, s["model_color"]),
            linewidth=s["line_width"], label=label_legend
        )
    
    ax.plot(
        Initial_fit_color['Stim_amp_pA'], Initial_fit_color['Frequency_Hz'],
        color=s["initial_color"], linewidth=s["line_width"], linestyle="--",
        label="Hill–Sigmoid (initial)"
    )
    ax.plot(
        model_table['Stim_amp_pA'], model_table['Frequency_Hz'],
        color=s["model_color"], linewidth=s["line_width"], label="IO fit"
    )
    ax.plot(
        gain_table['Stim_amp_pA'], gain_table['Frequency_Hz'],
        color=s["gain_color"], linewidth=s["line_width"] + 1.5, label="Linear IO portion"
    )
    
    xmin, xmax = ax.get_xlim()
    if "Saturation" in io_dict:
        sat = io_dict['Saturation']
        # ax.vlines(
        #     x=sat["Stim_amp_pA"].iloc[0],
        #     ymin=-5, ymax=sat["Frequency_Hz"].iloc[0],
        #     color=s["saturation_color"], linestyle="--", linewidth=s["line_width"], label="Saturation"
        # )
        ax.hlines(
            y=sat["Frequency_Hz"].iloc[0],
            xmin=xmin, xmax=sat["Stim_amp_pA"].iloc[0],
            color=s["saturation_color"], linestyle="--", linewidth=s["line_width"]
        )
    
    x_range = np.arange(original_data_table['Stim_amp_pA'].min(), stimulus_for_max_freq, 1)
    y_values = Intercept + Gain * x_range
    
    
    
    # Remove values above maximum_linear_fit AND below 0
    mask = (y_values <= maximum_linear_fit) & (y_values >= 0)
    x_range_masked = x_range[mask]
    y_values_masked = y_values[mask]
    
    ax.plot(
        x_range_masked, y_values_masked,
        color=s["linear_color"], linestyle=":", linewidth=s["line_width"], label="Linear fit"
    )
    
    ymin, ymax = ax.get_ylim()

    handles, labels = ax.get_legend_handles_labels()
    
    # --- Formatting ---
    ax.set_title("Input–Output Fit", fontsize=s["fontsize"], pad=10)
    ax.set_xlabel("Input current (pA)", fontsize=s["fontsize"])
    ax.set_ylabel("Firing frequency (Hz)", fontsize=s["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=s["tick_font_size"])
    ax.set_ylim(bottom=-5)
    ax.set_xlim([xmin, xmax])
    if s["grid"]:
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)


    # --- Threshold arrow below x-axis ---
    if Threshold_table is not None and not Threshold_table.empty:
        threshold_x = Threshold_table["Stim_amp_pA"].iloc[0]
        # Position sous l'axe des x
        threshold_y = -(ymax * 0.125)
        threshold_y = -5

        # Add arrow just below x-axis
        ax.annotate(
            '', 
            xy=(threshold_x, -5),  # arrow tip (x on axis, y=0)
            xytext=(threshold_x, -(ymax*0.125)),  # start of the arrow (below x-axis)
            arrowprops=dict(
                arrowstyle='-|>', 
                color=s["threshold_color"], 
                lw=1.8
            )
        )
        
        # Create proxy handle

        threshold_handle = Line2D(
            [0], [0], color=s["threshold_color"], 
            marker = "^",
            markersize=10, 
            linestyle='None'
        )

        
        
        
        
        handles.append(threshold_handle)
        labels.append("Threshold")
        
        # ax.plot(
        #     threshold_x, 
        #     threshold_y,
        #     marker='^',
        #     markersize=10,
        #     color=s["threshold_color"],
        #     markeredgewidth=1.5,
        #     markeredgecolor=s["threshold_color"],
        #     markerfacecolor=s["threshold_color"],
        #     clip_on=False,  # Important: permet d'afficher en dehors de la zone de plot
        #     zorder=10  # Assure que le marker est au-dessus
        # )
        
        # # Create proxy handle (identique au marker sur le plot)
        # threshold_handle = Line2D(
        #     [0], [0], 
        #     color=s["threshold_color"], 
        #     marker="^",
        #     markersize=10, 
        #     linestyle='None',
        #     markerfacecolor=s["threshold_color"],
        #     markeredgecolor=s["threshold_color"]
        # )
        
        # handles.append(threshold_handle)
        # labels.append("Threshold")
    
    
    if "Response_Failure" in io_dict:
        fail = io_dict['Response_Failure']
        
        failure_x = fail["Stim_amp_pA"].iloc[0]
    
    
        # Add arrow just below x-axis
        ax.annotate(
            '', 
            xy=(failure_x, -5),  # arrow tip (x on axis, y=0)
            xytext=(failure_x, -(ymax*0.125)),  # start of the arrow (below x-axis)
            arrowprops=dict(
                arrowstyle='-|>', 
                color=s["failure_color"], 
                lw=1.8
            )
        )
        failure_handle = Line2D(
            [0], [0], color=s["failure_color"], marker='^', markersize=14, linestyle='None'
        )
        handles.append(failure_handle)
        labels.append("Response failure")
        
    
    
    
    # --- Legend ---
    
    # Create a proxy arrow for the legend
    
    
    
    
    
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               fontsize=s["fontsize"]-1, ncol=4, frameon=s["legend_frame"])
    
    

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    

    
        
        
    # --- Saving ---
    if saving_path:
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi=300)
        # optional: also save as SVG for vector graphics
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    
    if return_plot:
        return fig, ax
    else:
        plt.show()
        
def plot_stim_freq_table_matplotlib(stim_freq_table, return_plot = False, saving_path = None):
    
    # --- Create figure ---
           
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # === QC passed ===
    scatter = ax.scatter(
        stim_freq_table["Stim_amp_pA"],
        stim_freq_table["Frequency_Hz"],
        s=s["marker_size"],
        alpha=0.8,
        picker=True  # Pour permettre l'interactivité si nécessaire
    )
    
    # --- Layout ---
    ax.set_title("Input–Output Fit", fontsize=s.get("fontsize", 12), pad=10)
    ax.set_xlabel("Input current (pA)", fontsize=s.get("fontsize", 12))
    ax.set_ylabel("Firing frequency (Hz)", fontsize=s.get("fontsize", 12))
    
    # Tick font size
    ax.tick_params(axis='both', which='major', labelsize=s["tick_font_size"])
    
    # Grid
    ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Grid behind data
    
    # Spines (axes lines)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    # Ticks outside
    ax.tick_params(
        axis='both',
        which='both',
        direction='out',
        top=True,      # Ticks on top
        right=True,    # Ticks on right
        bottom=True,
        left=True
    )
    
    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Legend (horizontal, below plot)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,  # Ajuster selon le nombre d'éléments
        frameon=False
    )
    
    # Margins
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.15)
    
    # --- Saving ---
    if saving_path:
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi=300)
        # optional: also save as SVG for vector graphics
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    
    if return_plot:
        return fig, ax
    else:
        plt.show()
        
    
def plot_IO_detailed_fit_plotly(IO_plot_dict, do_fit, stim_freq_table):

    # --- Default style ---
    default_style = {
        "fontsize": 14,
        "tick_font_size": 12,
        "line_width": 2,
        "marker_size": 10,
        "passed_color": "#0072B2",
        "failed_color": "#EE6677",
        "trimmed_color": "#00694d",
        "initial_color": "#56B4E9",
        "final_color": "#D55E00",
        "model_color": "#E69F00",
        "gain_color": "#d52b00",
        "linear_color": "#000000",
        "threshold_color": "#000000",
        "saturation_color": "#D55E00",
        "failure_color": "#D55E00"
    }
    
    s = default_style

    # --- Extract dictionaries ---
    original_data_table = IO_plot_dict["1-Polynomial_fit"]["original_data_table"]
    passed = original_data_table[original_data_table["Passed_QC"]]
    failed = original_data_table[~original_data_table["Passed_QC"]]

    poly_dict = IO_plot_dict["1-Polynomial_fit"]
    trimmed_data = poly_dict["trimmed_stimulus_frequency_table"]
    poly_table = poly_dict["trimmed_3rd_poly_table"]
    poly_color_dict = poly_dict["color_shape_dict"]

    hill_dict = IO_plot_dict["2-Final_Hill_Sigmoid_Fit"]
    initial_table = hill_dict["Initial_Fit_table"]

    io_dict = IO_plot_dict["3-IO_fit"]
    model_table = io_dict["model_table"]
    gain_table = io_dict["gain_table"]
    intercept = io_dict["intercept"]
    gain = io_dict["Gain"]
    threshold_table = io_dict["Threshold"]
    stim_max = io_dict["stimulus_for_maximum_frequency"]
    
    maximum_frequency = np.nanmax(original_data_table.loc[:,'Frequency_Hz'])
    if "Saturation" in io_dict:
        sat = io_dict["Saturation"]
        maximum_linear_fit = np.nanmax([maximum_frequency, sat["Frequency_Hz"].iloc[0]])
    else:
        maximum_linear_fit = maximum_frequency
    
# --- Create figure ---
    fig = go.Figure()

    # === Stim Freq table
    for df, color, name in [
        (passed, s["passed_color"], "QC passed"),
        (failed, s["failed_color"], "QC failed"),
        (trimmed_data, s["trimmed_color"], "Trimmed data")]:
    
        fig.add_trace(go.Scatter(
            x=df["Stim_amp_pA"],
            y=df["Frequency_Hz"],
            mode="markers",
            marker=dict(size=s["marker_size"], color=color, opacity=0.8),
            name=name,
            customdata=df[["Sweep"]],
            hovertemplate=(
                    "Sweep: %{customdata[0]}<br>"
                    "Stim: %{x:.2f} pA<br>"
                    "Freq: %{y:.2f} Hz<extra></extra>"
                )

        ))

    # === Polynomial fits ===
    for legend in poly_table["Legend"].unique():
        df_leg = poly_table[poly_table["Legend"] == legend]

        label = "Polynomial fit" if legend == "3rd_order_poly" else legend
        fig.add_trace(go.Scatter(
            x=df_leg["Stim_amp_pA"],
            y=df_leg["Frequency_Hz"],
            mode="lines",
            line=dict(color=poly_color_dict.get(legend, s["model_color"]),
                      width=s["line_width"]),
            name=label
        ))

    # === Hill sigmoid initial ===
    fig.add_trace(go.Scatter(
        x=initial_table["Stim_amp_pA"],
        y=initial_table["Frequency_Hz"],
        mode="lines",
        line=dict(color=s["initial_color"], width=s["line_width"], dash="dash"),
        name="Hill–Sigmoid (initial)"
    ))

    # === IO model ===
    fig.add_trace(go.Scatter(
        x=model_table["Stim_amp_pA"],
        y=model_table["Frequency_Hz"],
        mode="lines",
        line=dict(color=s["model_color"], width=s["line_width"]),
        name="IO fit"
    ))

    # === Linear IO portion ===
    fig.add_trace(go.Scatter(
        x=gain_table["Stim_amp_pA"],
        y=gain_table["Frequency_Hz"],
        mode="lines",
        line=dict(color=s["gain_color"], width=s["line_width"] + 1.5),
        name="Linear IO portion"
    ))

    # === Saturation line ===
    if "Saturation" in io_dict:
        sat = io_dict["Saturation"]
        fig.add_trace(go.Scatter(
            x=[-5, sat["Stim_amp_pA"].iloc[0]],
            y=[sat["Frequency_Hz"].iloc[0]] * 2,
            mode="lines",
            line=dict(color=s["saturation_color"], width=s["line_width"], dash="dash"),
            name="Saturation"
        ))

    # === Linear fit ===
    x_range = np.arange(original_data_table["Stim_amp_pA"].min(), stim_max, 1)
    y_vals = intercept + gain * x_range
    # Remove values above maximum_linear_fit AND below 0
    mask = (y_vals <= maximum_linear_fit) & (y_vals >= 0)
    fig.add_trace(go.Scatter(
        x=x_range[mask],
        y=y_vals[mask],
        mode="lines",
        line=dict(color=s["linear_color"], width=s["line_width"], dash="dot"),
        name="Linear fit"
    ))

    # ------------------------------------------------------------------
    # === Threshold arrow (Option A: inside plot at y = 0) ===
    # ------------------------------------------------------------------
    if threshold_table is not None and not threshold_table.empty:
        thr_x = threshold_table["Stim_amp_pA"].iloc[0]
        fig.add_trace(go.Scatter(
            x=[thr_x],
            y=[-2],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=s["threshold_color"]
            ),
            name="Threshold",
            showlegend=True
        ))


    # === Failure arrow ===
    if "Response_Failure" in io_dict:
        fail = io_dict["Response_Failure"]
        fx = fail["Stim_amp_pA"].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=[fx],
            y=[-2],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=s["failure_color"]
            ),
            name="Response Failure",
            showlegend=True
        ))

        

    # --- Layout ---
    fig.update_layout(
        # width=600,   # in pixels
        # height=600,  # in pixels
        plot_bgcolor="white",
        template = "none",
        title="Input–Output Fit",
        xaxis_title="Input current (pA)",
        yaxis_title="Firing frequency (Hz)",
        xaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        yaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=10, t=60, b=80)
    )
    
    
    #     # X axis
    # fig.update_layout(
    # plot_bgcolor="white",   # plotting area background
    
    # )
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    
    
        
    
    return fig

def plot_stim_freq_table_plotly(stim_freq_table):
    # --- Create figure ---
    fig = go.Figure()

    # === QC passed ===
    fig.add_trace(go.Scatter(
        x=stim_freq_table["Stim_amp_pA"],
        y=stim_freq_table["Frequency_Hz"],
        mode="markers",
        marker=dict(size=s["marker_size"], opacity=0.8),
        customdata=stim_freq_table[["Sweep"]],
        hovertemplate=(
                        "Sweep: %{customdata[0]}<br>"
                        "Stim: %{x:.2f} pA<br>"
                        "Freq: %{y:.2f} Hz<extra></extra>"
                    )

    ))
    # --- Layout ---
    fig.update_layout(
        # width=600,   # in pixels
        # height=600,  # in pixels
        plot_bgcolor="white",
        template = "none",
        title="Input–Output Fit",
        xaxis_title="Input current (pA)",
        yaxis_title="Firing frequency (Hz)",
        xaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        yaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=10, t=60, b=80)
    )
    
    
    
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    

    return fig

#%% Adaptation plot


def plot_Adaptation_fit_plot_choice(adaptation_plot_dict,stim_freq_table = None, plot_type = "matplotlib", do_fit = True):
    
    if do_fit == True:
        if plot_type == "matplotlib":
            adaptation_fig = plot_adaptation_TACO_paper_one_panel(adaptation_plot_dict, return_plot=True)[0]
        elif plot_type == 'plotly':
            adaptation_fig = plotly_adaptation_TACO_one_panel(adaptation_plot_dict)
            
    return adaptation_fig

def plot_adaptation_TACO_paper_one_panel(plot_dict, return_plot=False, style_dict = None, saving_path=""):
    # --- Extract data ---
    original_sim_table = plot_dict["original_sim_table"]
    sim_table = plot_dict["sim_table"]
    Na = plot_dict["Na"]
    C_ref = plot_dict["C_ref"]
    interval_frequency_table = plot_dict["interval_frequency_table"]
    median_table = plot_dict["median_table"]
    M = plot_dict["M"]
    C = plot_dict["C"]

    # --- Default style (PLOS ONE & color-blind friendly) ---
    default_style = {
        "figsize": (6, 5),
        "xlabel": "Spike interval",
        "ylabel": "Normalized feature",
        "fontsize": 10,
        "tick_font_size": 9,
        "line_color_original": "#56B4E9",  # Sky Blue
        "line_color_sim": "#009E73",       # Bluish Green
        "area_color": "#F0E442",           # Yellow
        "rect_color": "#CC79A7",           # Reddish Purple
        "scatter_cmap": "viridis",
        "median_marker": "s",
        "median_color": "#D55E00",         # Vermilion
        "max_median_size": 18,
        "dpi": 300,
    }
    if style_dict:
        default_style.update(style_dict)
    s = default_style

    # --- Create single figure ---
    fig, ax = plt.subplots(figsize=s["figsize"], dpi=s["dpi"])

    # --- Scatter: Original data ---
    sc = ax.scatter(
        interval_frequency_table["Spike_Interval"],
        interval_frequency_table["Normalized_feature"],
        c=interval_frequency_table["Stimulus_amp_pA"],
        cmap=s["scatter_cmap"],
        alpha=0.6,
        edgecolor="none",
        label="Original data",
    )

    # --- Median points ---
    sizes = (
        median_table["Count_weigths"] / median_table["Count_weigths"].max()
    ) * s["max_median_size"]
    ax.scatter(
        median_table["Spike_Interval"],
        median_table["Normalized_feature"],
        s=sizes,
        c=s["median_color"],
        marker=s["median_marker"],
        alpha=0.7,
        label="Median interval value",
    )

    # --- Simulation (exponential fit) ---
    ax.plot(
        sim_table["Spike_Interval"],
        sim_table["Normalized_feature"],
        color=s["line_color_sim"],
        linewidth=2,
        label="Exponential fit",
    )

    # --- Filled area (M) ---
    ax.fill_between(
        original_sim_table["Spike_Interval"],
        original_sim_table["Normalized_feature"],
        color=s["area_color"],
        alpha=0.6,
        label=f"M = {np.round(M, 2)}",
    )

    
    ax.fill_betweenx(
            y=[0, C_ref],
            x1=np.nanmin(sim_table["Spike_Interval"]),
            x2=(Na - 1) - 1,
            color=s["rect_color"],
            alpha=1,
            label=f"C = {np.round(C, 2)}",
        )
 
    # --- Formatting ---
    ax.set_xlabel(s["xlabel"], fontsize=s["fontsize"])
    ax.set_ylabel(s["ylabel"], fontsize=s["fontsize"])
    ax.tick_params(axis="both", which="major", labelsize=s["tick_font_size"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Legend (below plot) ---
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=s["fontsize"],
        ncol=2,
        frameon=False,
    )

    # --- Colorbar ---
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Input current (pA)", fontsize=s["fontsize"])
    cbar.ax.tick_params(labelsize=s["tick_font_size"])

    plt.tight_layout()
    
    if return_plot == True:
        return fig, ax

    if saving_path:
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi = 300)
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    plt.show()
    

def plotly_adaptation_TACO_one_panel(plot_dict, style_dict=None):
    # --- Extract data ---
    original_sim_table = plot_dict["original_sim_table"]
    sim_table = plot_dict["sim_table"]
    Na = plot_dict["Na"]
    C_ref = plot_dict["C_ref"]
    interval_frequency_table = plot_dict["interval_frequency_table"]
    median_table = plot_dict["median_table"]
    M = plot_dict["M"]
    C = plot_dict["C"]

    # --- Default style ---
    default_style = {
        "xlabel": "Spike interval",
        "ylabel": "Normalized feature",
        "line_color_sim": "#009E73",
        "area_color": "#F0E442",
        "rect_color": "#CC79A7",
        "median_color": "#D55E00",
        "max_median_size": 18,
        "scatter_cmap": "Viridis",  # Plotly name
    }
    if style_dict:
        default_style.update(style_dict)
    s = default_style

    # --- Create figure ---
    fig = go.Figure()

    # === Scatter: Original data ===
    fig.add_trace(go.Scatter(
        x=interval_frequency_table["Spike_Interval"],
        y=interval_frequency_table["Normalized_feature"],
        mode="markers",
        marker=dict(
            color=interval_frequency_table["Stimulus_amp_pA"],
            colorscale=s["scatter_cmap"],
            size=6,
            opacity=0.6,
            colorbar=dict(title="Input current (pA)")
        ),
        name="Original data"
    ))

    # === Median points ===
    sizes = (
        median_table["Count_weigths"] /
        median_table["Count_weigths"].max()
    ) * s["max_median_size"]

    fig.add_trace(go.Scatter(
        x=median_table["Spike_Interval"],
        y=median_table["Normalized_feature"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=s["median_color"],
            symbol="square",
            opacity=0.7,
            line=dict(width=0)
        ),
        name="Median interval value"
    ))

    # === Simulation curve ===
    fig.add_trace(go.Scatter(
        x=sim_table["Spike_Interval"],
        y=sim_table["Normalized_feature"],
        mode="lines",
        line=dict(color=s["line_color_sim"], width=2),
        name="Exponential fit"
    ))

    # === Filled area (M) ===
    fig.add_trace(go.Scatter(
        x=original_sim_table["Spike_Interval"],
        y=original_sim_table["Normalized_feature"],
        fill="tozeroy",
        fillcolor="rgba(240, 228, 66, 0.6)",  # hex #F0E442 with alpha 0.1
        line=dict(color="rgba(0,0,0,0)"),
        name=f"M = {np.round(M, 2)}"
    ))

    # === Rectangle for C ===
    rect_x0 = np.nanmin(sim_table["Spike_Interval"])
    rect_x1 = (Na - 1) - 1

    fig.add_shape(
        type="rect",
        x0=rect_x0,
        x1=rect_x1,
        y0=0,
        y1=C_ref,
        fillcolor=s["rect_color"],
        opacity=1,
        line=dict(width=0),
    )

    # Dummy trace to show C in legend as a thick line
    fig.add_trace(go.Scatter(
        x=[0, 1],               # invisible range
        y=[None, None],         # will not plot any actual points
        mode="lines",
        line=dict(
            color=s["rect_color"],
            width=8              # thickness of the legend line
        ),
        name=f"C = {np.round(C, 2)}",
        showlegend=True
    ))

    # === Formatting ===
    fig.update_layout(
        xaxis_title=s["xlabel"],
        yaxis_title=s["ylabel"],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=20, t=20, b=80),
    )

    return fig