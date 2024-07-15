import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
from lmfit import Model, Parameters, Minimizer, conf_interval, report_fit, report_ci
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# Define the Boltzmann sigmoid function
def boltzmann(x, A, B, C, D):
    return A + (B - A) / (1 + np.exp((C - x) / D))

# Find the inflection temperature of the Boltzmann fit
def find_inflection_temperature(params):
    return params['C'].value

# Split the DataFrame at the transition
def split_at_transition(df):
    transition_idx = df.index[df.eq("Channel 2").any(axis=1)].tolist()[0]
    df_before = df.iloc[:transition_idx, :]
    df_after = df.iloc[(transition_idx + 2):, :]
    return df_before, df_after

# Round float column names
def round_float_column_names(df):
    rounded_column_names = []
    for col_name, col_type in df.dtypes.items():
        if col_type == 'float64':
            rounded_column_names.append(round(float(col_name)))
        else:
            rounded_column_names.append(col_name)
    return rounded_column_names

# Show melt curves and ask the user to select a wavelength
def show_melt_curves_and_select_wavelength(smoothed_data, sample_name1):
    plt.figure(figsize=(10, 6))
    for col in smoothed_data.columns:
        plt.plot(smoothed_data.index, smoothed_data[col], label=f'{col}°C')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Ellipicity θ (millidegrees)')
    plt.title(f'Melt Curves for {sample_name1}')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)  # Non-blocking show

    wavelength = None
    while True:
        try:
            wavelength_input = simpledialog.askstring("Input", "Enter the wavelength to analyze:")
            if wavelength_input is None:
                plt.close()
                return None
            wavelength = float(wavelength_input)
            if wavelength in smoothed_data.index or str(wavelength) in smoothed_data.index:
                wavelength = float(wavelength) if float(wavelength) in smoothed_data.index else str(wavelength)
                plt.close()
                return wavelength
            else:
                messagebox.showerror("Input Error", f"Wavelength {wavelength} is not in the data. Please enter a valid wavelength.")
        except ValueError:
            messagebox.showerror("Input Error", "Invalid input. Please enter a numerical value for the wavelength.")
            
def residual(params, x, data):
    A = params['A']
    B = params['B']
    C = params['C']
    D = params['D']
    model = boltzmann(x, A, B, C, D)
    return model - data

# Function to fit the model and calculate confidence intervals
def fit_model_and_calculate_ci(ellipticity_data, temperatures):
    model = Model(boltzmann)
    params = model.make_params(A=min(ellipticity_data), B=max(ellipticity_data), C=np.mean(temperatures), D=10.0)
    result = model.fit(ellipticity_data, x=temperatures, params=params)
    
    residuals = ellipticity_data - result.best_fit
    ss_residuals = np.sum(residuals ** 2)
    ss_total = np.sum((ellipticity_data - np.mean(ellipticity_data)) ** 2)
    r_squared = 1 - (ss_residuals / ss_total)

    mini = Minimizer(residual, params, fcn_args=(temperatures, ellipticity_data))
    minimizer_result = mini.minimize()

    report_fit(minimizer_result)
    ci = conf_interval(mini, result=minimizer_result)
    report_ci(ci)
    
    ci_C = ci['C']
    print(f"ci_C: {ci_C}")  # Print out the structure of ci_C for debugging
    
    # Extract 95.45% confidence interval values
    ci_lower_bound = ci_C[1][1]  # 95.45% lower bound
    ci_upper_bound = ci_C[5][1]  # 95.45% upper bound
    
    ci_text = f'95% CI for Melting Temperature: [{ci_lower_bound:.2f}, {ci_upper_bound:.2f}]'
    melting_temperature = find_inflection_temperature(result.params)
    return melting_temperature, ci_text, (ci_lower_bound, ci_upper_bound), result.best_fit, r_squared

# Plot CD vs Wavelength and save to PDF
def plot_cd_vs_wavelength(output_dir, pdf_pages, graph_titles, *csv_files):
    num_files = len(csv_files)
    if num_files % 2 != 0:
        print("Please provide an even number of CSV files.")
        return

    melting_temperatures = []
    ci_intervals = []

    for i in range(0, len(csv_files), 2):
        csv1, csv2 = csv_files[i], csv_files[i + 1]
        sample_name1 = graph_titles[int(i / 2)]
        df_sampleCDHT = pd.read_csv(csv1, skiprows=19)
        df_blankCDHT = pd.read_csv(csv2, skiprows=19)
        df_sampleCD, df_sampleHT = split_at_transition(df_sampleCDHT)
        df_blankCD, df_blankHT = split_at_transition(df_blankCDHT)
        rounded_sampleCD_columns = round_float_column_names(df_sampleCD)
        rounded_sampleHT_columns = round_float_column_names(df_sampleHT)
        rounded_blankCD_columns = round_float_column_names(df_blankCD)
        rounded_blankHT_columns = round_float_column_names(df_blankHT)
        df_sampleCD.columns = rounded_sampleCD_columns
        df_sampleHT.columns = rounded_sampleHT_columns
        df_blankCD.columns = rounded_blankCD_columns
        df_blankHT.columns = rounded_blankHT_columns
        df_sampleCD.set_index(df_sampleCD.columns[0], inplace=True)
        df_sampleHT.set_index(df_sampleHT.columns[0], inplace=True)
        df_blankCD.set_index(df_blankCD.columns[0], inplace=True)
        df_blankHT.set_index(df_blankHT.columns[0], inplace=True)
        df_sampleCD = df_sampleCD.iloc[::-1]
        df_sampleHT = df_sampleHT.iloc[::-1]
        df_blankCD = df_blankCD.iloc[::-1]
        df_blankHT = df_blankHT.iloc[::-1]
        df_sampleCDblanked = df_sampleCD - df_blankCD

        # Filter out CD data associated with voltage exceeding 700
        voltage_threshold = 700
        filtered_data = []
        for col in df_sampleCDblanked.columns:
            voltage_col = df_sampleHT[col]
            mask = voltage_col <= voltage_threshold
            filtered_col = df_sampleCDblanked[col][mask].dropna()
            filtered_data.append(filtered_col)

        # Combine filtered columns into a DataFrame, aligning indexes
        df_sampleCDblanked_filtered = pd.concat(filtered_data, axis=1, join='inner')

        # Show the melt curves and get the selected wavelength from the user
        selected_wavelength = show_melt_curves_and_select_wavelength(df_sampleCDblanked_filtered, sample_name1)
        if selected_wavelength is None:
            return

        ellipticity_data = df_sampleCDblanked_filtered.loc[selected_wavelength]
        temperatures = np.array(ellipticity_data.index, dtype=float)

        # Create a new figure for the final analysis
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        axes[0].plot(df_sampleCDblanked_filtered.index, df_sampleCDblanked_filtered)
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Ellipicity θ (millidegrees)')
        axes[0].set_title(f'CD Melt Spectra of {sample_name1}')
        axes[0].set_xticks(df_sampleCDblanked_filtered.index[::100])
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].axvline(x=selected_wavelength, color='r', linestyle='--')

        axes[1].plot(df_sampleHT.index, df_sampleHT)
        axes[1].set_ylabel('HT (Voltage)', color='green')
        axes[1].set_title(f'HT (Voltage) Check')
        axes[1].tick_params(axis='y', labelcolor='green')
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_xticks(df_sampleHT.index[::100])

        axes[2].plot(temperatures, ellipticity_data, marker='o')
        axes[2].set_xlabel('Temperature (°C)')
        axes[2].set_ylabel('Ellipicity θ (millidegrees)')
        axes[2].set_title(f'Temperature vs. Ellipticity at {selected_wavelength} nm')

        melting_temperature, ci_text, ci_interval, best_fit, r_squared = fit_model_and_calculate_ci(ellipticity_data, temperatures)
        melting_temperatures.append(melting_temperature)
        ci_intervals.append(ci_interval)

        axes[2].plot(temperatures, best_fit, 'r-', label='Fitted curve')
        fit_summary_text = f'R-squared: {r_squared:.4f}\nMelting Temperature: {melting_temperature:.2f} °C\n{ci_text}'
        axes[2].text(0.05, 0.95, fit_summary_text, transform=axes[2].transAxes, fontsize=10, verticalalignment='top')
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.6, wspace=0.2)

        pdf_pages.savefig(fig)
        plt.close(fig)

    # Plot the melting temperatures and their confidence intervals only once
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(melting_temperatures))
    y = melting_temperatures
    ci_lower = [ci[0] for ci in ci_intervals]
    ci_upper = [ci[1] for ci in ci_intervals]

    # Plot the melting temperatures as points
    ax.plot(x, y, 'o', label='Melting Temperature')

    # Draw the error bars manually
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [ci_lower[i], ci_upper[i]], color='red', linestyle='-', linewidth=2)
        ax.plot([x[i] - 0.1, x[i] + 0.1], [ci_lower[i], ci_lower[i]], color='red', linestyle='-', linewidth=2)
        ax.plot([x[i] - 0.1, x[i] + 0.1], [ci_upper[i], ci_upper[i]], color='red', linestyle='-', linewidth=2)

    ax.set_xlabel('Sample')
    ax.set_ylabel('Melting Temperature (°C)')
    ax.set_title('Melting Temperatures with 95% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(graph_titles[:len(x)], rotation=45, ha='right')
    ax.margins(x=0.05)  # Adjust margins to reduce whitespace
    plt.tight_layout(pad=2)  # Adjust padding to reduce overall figure whitespace

    pdf_pages.savefig(fig)
    plt.close(fig)

# Function to add sample and blank file pair
def add_sample_blank_pair():
    sample_file = filedialog.askopenfilename(title="Select Sample CSV File", filetypes=[("CSV files", "*.csv")])
    if not sample_file:
        return
    blank_file = filedialog.askopenfilename(title="Select Blank CSV File", filetypes=[("CSV files", "*.csv")])
    if not blank_file:
        return
    title = title_entry.get().strip()
    if not title:
        messagebox.showerror("Input Error", "Please enter a title for the graphs.")
        return

    graph_title = graph_title_entry.get().strip()
    if not graph_title:
        messagebox.showerror("Input Error", "Please enter a title for the graph.")
        return

    sample_blank_pairs.append((sample_file, blank_file))
    titles.append(graph_title)

    samples_listbox.insert(tk.END, f"Sample: {os.path.basename(sample_file)} | Blank: {os.path.basename(blank_file)} | Title: {title} | Graph Title: {graph_title}")
    title_entry.delete(0, tk.END)
    graph_title_entry.delete(0, tk.END)

# Function to generate PDF
def generate_pdf():
    if not sample_blank_pairs:
        messagebox.showerror("Input Error", "Please add at least one sample and blank pair.")
        return
    output_filename = output_filename_entry.get().strip()
    if not output_filename:
        messagebox.showerror("Input Error", "Please enter an output filename.")
        return
    output_path = os.path.join(output_dir, f"{output_filename}.pdf")

    with PdfPages(output_path) as pdf_pages:
        plot_cd_vs_wavelength(output_dir, pdf_pages, titles, *(file for pair in sample_blank_pairs for file in pair))

    if os.path.exists(output_path) and os.stat(output_path).st_size == 0:
        os.remove(output_path)
        messagebox.showinfo("Info", "No plots were generated, the PDF file was not created.")
    else:
        messagebox.showinfo("Success", f"Plots and tables have been saved to {output_path}")

# GUI setup
root = tk.Tk()
root.title("CD Single Temperature Analysis")
root.geometry("1000x450")  # Set the window size to 800x600 pixels

frame = tk.Frame(root, bg="lightgray")
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Label and Entry for sample and blank pair title
title_label = tk.Label(frame, text="Sample and Blank Pair Title:", bg="lightgray")
title_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
title_entry = tk.Entry(frame, width=50)
title_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)

# Label and Entry for graph title
graph_title_label = tk.Label(frame, text="Graph Title:", bg="lightgray")
graph_title_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
graph_title_entry = tk.Entry(frame, width=50)
graph_title_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.EW)

# Button to add sample and blank pair
add_pair_button = tk.Button(frame, text="Add Sample and Blank Pair", command=add_sample_blank_pair)
add_pair_button.grid(row=0, column=2, rowspan=2, padx=10, pady=10)

# Listbox to display added sample and blank pairs
samples_listbox = tk.Listbox(frame, width=100, height=10)
samples_listbox.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky=tk.EW)

# Label and Entry for output PDF filename
output_filename_label = tk.Label(frame, text="Output PDF Filename:", bg="lightgray")
output_filename_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
output_filename_entry = tk.Entry(frame, width=50)
output_filename_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

# Button to generate PDF
generate_pdf_button = tk.Button(frame, text="Generate PDF", command=generate_pdf)
generate_pdf_button.grid(row=3, column=2, padx=10, pady=10)

# Configure column 1 to expand
frame.grid_columnconfigure(1, weight=1)

# Initialize lists for sample and blank pairs, and titles
sample_blank_pairs = []
titles = []

# Set output directory
output_dir = r'Y:\CD\Analysis-Development\Test_Outputs'
os.makedirs(output_dir, exist_ok=True)

# Run the GUI loop
root.mainloop()







