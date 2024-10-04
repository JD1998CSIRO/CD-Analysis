import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import re
import numpy as np
from lmfit import Model, Parameters, Minimizer, conf_interval, report_fit, report_ci
import tkinter as tk
from tkinter import Tk, filedialog, Entry, Label, Button, Listbox, messagebox, simpledialog
from scipy.signal import savgol_filter
from scipy.integrate import simps
from scipy import stats
from scipy.stats import ttest_ind
import seaborn as sns




# Function to handle CD Single
def cd_single():
    
    # Function to read CSV file and extract numeric and additional data
    def read_csv_until_string(file_path, start_string="XYDATA", use_cols=[0, 1, 2, 3], window_length=20, polyorder=3):
        numeric_data = {'Wavelength': [], 'CD': [], 'Voltage': [], 'Absorbance': []}
        additional_data = []
        read_numeric_data = False

        row_names_to_keep = [
            "Sample name", "Creation date", "Start", "End", "Data interval",
            "Instrument name", "Model name", "Serial No.", "Temperature",
            "Measurement date", "Data pitch", "CD scale", "FL scale", "D.I.T.",
            "Bandwidth", "Start mode", "Scanning speed", "Accumulations"
        ]
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if not read_numeric_data:
                        if start_string in line:
                            read_numeric_data = True
                        else:
                            cols = line.strip().split(',')
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
                    else:
                        if start_string in line:
                            continue

                        cols = line.strip().split(',')
                        try:
                            wavelength = float(cols[use_cols[0]].strip())
                            cd = float(cols[use_cols[1]].strip())
                            voltage = float(cols[use_cols[2]].strip())
                            absorbance = float(cols[use_cols[3]].strip())
                            numeric_data['Wavelength'].append(wavelength)
                            numeric_data['CD'].append(cd)
                            numeric_data['Voltage'].append(voltage)
                            numeric_data['Absorbance'].append(absorbance)
                        except (ValueError, IndexError):
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        df_numeric = pd.DataFrame(numeric_data)
        df_numeric = df_numeric.sort_values('Wavelength')  # Sort by wavelength
        df_additional = pd.DataFrame(additional_data, columns=['Property', 'Value'])
        return df_numeric, df_additional

    # Function to normalize CD data by area under curve
    def normalize_cd(df):
        auc = np.trapz(df['CD'], df['Wavelength'])  # Calculate the area under the curve using trapezoidal rule
        df.loc[:, 'Normalized CD'] = df['CD'] / auc  # Normalize CD values so that AUC equals 1
        return df

    # Function to calculate residuals relative to reference df
    def calculate_residuals(df, reference_df):
        df.loc[:, 'Relative Normalized CD'] = df['Normalized CD'] - reference_df['Normalized CD']
        return df

    # Corrected plot_overlays function
    def plot_overlays(output_dir, pdf_pages, graph_titles, reference_index, *csv_files):
        num_pairs = len(csv_files) // 2
        fig, axes = plt.subplots(2, 3, figsize=(25, 18))  # Adjusted figure size
        normalized_data = []
        first_sample_info = True  # Flag to track if additional info has been added

        color_map = plt.get_cmap('rainbow')  # Color map for shades of blue

        for i in range(num_pairs):
            csv1, csv2 = csv_files[i * 2], csv_files[i * 2 + 1]
            sample_name1 = os.path.splitext(os.path.basename(csv1))[0]
            sample_name2 = os.path.splitext(os.path.basename(csv2))[0]

            # Read sample and blank CSV files
            df_CD1, df_additional1 = read_csv_until_string(csv1)
            df_CD1.columns = ['Wavelength', 'CD', 'Voltage', 'Absorbance']
            df_CDblank1, df_additional_blank1 = read_csv_until_string(csv2)
            df_CDblank1.columns = ['Wavelength', 'CD', 'Voltage', 'Absorbance']

            # Subtract blank CD values from sample CD values
            df_CD_blank_subtracted1 = df_CD1.copy()
            df_CD_blank_subtracted1['CD'] = df_CD1['CD'] - df_CDblank1['CD']
            df_CD_blank_subtracted1['Absorbance'] = df_CD1['Absorbance'] - df_CDblank1['Absorbance']

            # Normalize CD data
            df_normalized = normalize_cd(df_CD_blank_subtracted1)
            normalized_data.append(df_normalized)

            # Prepare reference df and calculate residuals
            reference_df = normalized_data[reference_index]
            df_relative = calculate_residuals(df_normalized, reference_df)

            # Prepare additional info (only for the first sample)
            if first_sample_info:
                additional_info = f"Sample: {sample_name1}\n"
                additional_info += '\n'.join([f"{row[0]}: {row[1]}" for row in df_additional1.values])
                additional_info_list = [(additional_info, sample_name1)]  # List to hold additional info
                first_sample_info = False  # Additional info added, so turn off the flag
            else:
                additional_info_list = []  # No additional info for subsequent samples

            # Get title for the graph
            title = graph_titles[i] if i < len(graph_titles) else f'Sample {i+1}'

            # Determine color
            color = color_map(i / num_pairs)  # Get color from the colormap

            # Plot overlay of CD vs Wavelength
            if i == reference_index:
                axes[0, 0].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['CD'], label= f'{title} (Reference)', linewidth=0.5, linestyle='--', color='grey')
            else:
                axes[0, 0].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['CD'], label=title, linewidth=0.5, color=color)

            # Plot overlay of Normalized CD vs Wavelength
            if i == reference_index:
                axes[1, 0].plot(df_normalized['Wavelength'], df_normalized['Normalized CD'], label = f'{title} (Reference)', linewidth=0.5, linestyle='--', color='grey')
            else:
                axes[1, 0].plot(df_normalized['Wavelength'], df_normalized['Normalized CD'], label=title, linewidth=0.5, color=color)  

            # Plot overlay of HT vs Wavelength
            axes[0, 1].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['Voltage'], label=title, linewidth=0.5, color=color)
            # Add vertical line at wavelength where voltage = 700
            wavelength_at_700 = df_CD_blank_subtracted1[df_CD_blank_subtracted1['Voltage'] == 700]['Wavelength']
            if not wavelength_at_700.empty:
                axes[0, 1].axvline(x=wavelength_at_700.values[0], color='red', linestyle='--', linewidth=0.5)

            # Plot Absorbance
            axes[0, 2].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['Absorbance'], label=title, linewidth=0.5, color=color)

            # Plot Residuals relative to reference spectrum
            axes[1, 1].plot(df_relative['Wavelength'], df_relative['Relative Normalized CD'], label=title, linewidth=0.5, color=color)

            # Plot additional information on [1, 2] only for the first sample
            if additional_info_list:
                info, sample_name = additional_info_list[0]
                axes[1, 2].text(0.1, 0.9, f"Additional Information for Sample {sample_name}:\n\n{info}", fontsize=8, verticalalignment='top', wrap=True)
                axes[1, 2].axis('off')  # Turn off the axis for this subplot

        # Set titles, labels, and other properties for the axes
        axes[0, 0].set_title('Overlay of CD vs. Wavelength')
        axes[0, 0].set_xlabel('Wavelength')
        axes[0, 0].set_ylabel('CD millidegrees')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[0, 0].legend()

        axes[0, 2].set_title('Overlay of Absorbance vs. Wavelength')
        axes[0, 2].set_xlabel('Wavelength')
        axes[0, 2].set_ylabel('Abs.')
        axes[0, 2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[0, 2].legend()

        axes[0, 1].set_title('Overlay of HT vs. Wavelength')
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('HT (Voltage)')
        axes[0, 1].legend()

        axes[1, 0].set_title('Normalized CD')
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 0].set_ylabel('Normalized Ellipticity')
        axes[1, 0].legend()

        axes[1, 1].set_title('Residuals relative to Reference')
        axes[1, 1].set_xlabel('Wavelength (nm)')
        axes[1, 1].set_ylabel('Difference from Reference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 1].legend()

        # Use tight layout to adjust spacing between subplots
        plt.tight_layout()

        # Save the plots to the PDF
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
            messagebox.showerror("Input Error", "Please enter a title for the graph.")
            return
        sample_blank_pairs.append((sample_file, blank_file))
        titles.append(title)

        samples_listbox.insert(tk.END, f"Sample: {os.path.basename(sample_file)} | Blank: {os.path.basename(blank_file)} | Title: {title}")
        title_entry.delete(0, tk.END)

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
            reference_index = reference_index_entry.get().strip()
            if not reference_index.isdigit() or int(reference_index) < 0 or int(reference_index) >= len(sample_blank_pairs):
                messagebox.showerror("Input Error", "Please enter a valid reference index.")
                return
            reference_index = int(reference_index)
            plot_overlays(output_dir, pdf_pages, titles, reference_index, *(file for pair in sample_blank_pairs for file in pair))

        messagebox.showinfo("Success", f"Plots and tables have been saved to {output_path}")

    # GUI setup
    root = tk.Tk()
    root.title("CD Single Temperature Analysis")
    root.geometry("800x600")  # Set the window size to 800x600 pixels

    frame = tk.Frame(root, bg="lightgray")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Label and Entry for graph title
    title_label = tk.Label(frame, text="Graph Title:", bg="lightgray")
    title_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    title_entry = tk.Entry(frame, width=50)
    title_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to add sample and blank pair
    add_pair_button = tk.Button(frame, text="Add Sample and Blank Pair", command=add_sample_blank_pair)
    add_pair_button.grid(row=0, column=2, padx=10, pady=10)

    # Listbox to display added sample and blank pairs
    samples_listbox = tk.Listbox(frame, width=100, height=10)
    samples_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for output PDF filename
    output_filename_label = tk.Label(frame, text="Output PDF Filename:", bg="lightgray")
    output_filename_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    output_filename_entry = tk.Entry(frame, width=50)
    output_filename_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for reference index
    reference_index_label = tk.Label(frame, text="Reference Index:", bg="lightgray")
    reference_index_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
    reference_index_entry = tk.Entry(frame, width=50)
    reference_index_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to generate PDF
    generate_pdf_button = tk.Button(frame, text="Generate PDF", command=generate_pdf)
    generate_pdf_button.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

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
    
# Function to handle CD Single MW
def cd_single_mw():
    
    # Function to read CSV file and extract numeric and additional data
    def read_csv_until_string(file_path, start_string="XYDATA", use_cols=[0, 1, 2, 3], window_length=20, polyorder=3):
        numeric_data = {'Wavelength': [], 'CD': [], 'Voltage': [], 'Absorbance': []}
        additional_data = []
        read_numeric_data = False

        row_names_to_keep = [
            "Sample name", "Creation date", "Start", "End", "Data interval",
            "Instrument name", "Model name", "Serial No.", "Temperature",
            "Measurement date", "Data pitch", "CD scale", "FL scale", "D.I.T.",
            "Bandwidth", "Start mode", "Scanning speed", "Accumulations"
        ]
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if not read_numeric_data:
                        if start_string in line:
                            read_numeric_data = True
                        else:
                            cols = line.strip().split(',')
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
                    else:
                        if start_string in line:
                            continue

                        cols = line.strip().split(',')
                        try:
                            wavelength = float(cols[use_cols[0]].strip())
                            cd = float(cols[use_cols[1]].strip())
                            voltage = float(cols[use_cols[2]].strip())
                            absorbance = float(cols[use_cols[3]].strip())
                            numeric_data['Wavelength'].append(wavelength)
                            numeric_data['CD'].append(cd)
                            numeric_data['Voltage'].append(voltage)
                            numeric_data['Absorbance'].append(absorbance)
                        except (ValueError, IndexError):
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        df_numeric = pd.DataFrame(numeric_data)
        df_numeric = df_numeric.sort_values('Wavelength')  # Sort by wavelength
        df_additional = pd.DataFrame(additional_data, columns=['Property', 'Value'])
        return df_numeric, df_additional

    # Function to normalize millidegrees to MRE
    def normalize_cd(df, molar_conc, no_residues, path_length):
        conversion_factor = 1 / (path_length * molar_conc * no_residues)
        df['Normalized CD'] = (df['CD'] * conversion_factor) / 3298  # Convert millidegrees to MRE
        return df

    # Function to calculate residuals relative to reference df
    def calculate_residuals(df, reference_df):
        df.loc[:, 'Relative Normalized CD'] = df['Normalized CD'] - reference_df['Normalized CD']
        return df

    # Corrected plot_overlays function
    def plot_overlays(output_dir, pdf_pages, graph_titles, reference_index, *csv_files):
        num_pairs = len(csv_files) // 2
        fig, axes = plt.subplots(2, 3, figsize=(25, 18))  # Adjusted figure size
        normalized_data = []
        first_sample_info = True  # Flag to track if additional info has been added

        color_map = plt.get_cmap('rainbow')  # Color map for shades of blue

        for i in range(num_pairs):
            csv1, csv2 = csv_files[i * 2], csv_files[i * 2 + 1]
            sample_name1 = os.path.splitext(os.path.basename(csv1))[0]
            sample_name2 = os.path.splitext(os.path.basename(csv2))[0]

            # Read sample and blank CSV files
            df_CD1, df_additional1 = read_csv_until_string(csv1)
            df_CD1.columns = ['Wavelength', 'CD', 'Voltage', 'Absorbance']
            df_CDblank1, df_additional_blank1 = read_csv_until_string(csv2)
            df_CDblank1.columns = ['Wavelength', 'CD', 'Voltage', 'Absorbance']

            # Subtract blank CD values from sample CD values
            df_CD_blank_subtracted1 = df_CD1.copy()
            df_CD_blank_subtracted1['CD'] = df_CD1['CD'] - df_CDblank1['CD']
            df_CD_blank_subtracted1['Absorbance'] = df_CD1['Absorbance'] - df_CDblank1['Absorbance']

            # Normalize CD data
            molar_conc = float(molar_conc_entry.get().strip())
            no_residues = float(no_residues_entry.get().strip())
            path_length = float(path_length_entry.get().strip())
            df_normalized = normalize_cd(df_CD_blank_subtracted1, molar_conc, no_residues, path_length)
            normalized_data.append(df_normalized)

            # Prepare reference df and calculate residuals
            reference_df = normalized_data[reference_index]
            df_relative = calculate_residuals(df_normalized, reference_df)

            # Prepare additional info (only for the first sample)
            if first_sample_info:
                additional_info = f"Sample: {sample_name1}\n"
                additional_info += '\n'.join([f"{row[0]}: {row[1]}" for row in df_additional1.values])
                additional_info_list = [(additional_info, sample_name1)]  # List to hold additional info
                first_sample_info = False  # Additional info added, so turn off the flag
            else:
                additional_info_list = []  # No additional info for subsequent samples

            # Get title for the graph
            title = graph_titles[i] if i < len(graph_titles) else f'Sample {i+1}'

            # Determine color
            color = color_map(i / num_pairs)  # Get color from the colormap

            # Plot overlay of CD vs Wavelength
            if i == reference_index:
                axes[0, 0].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['CD'], label= f'{title} (Reference)', linewidth=0.5, linestyle='--', color='grey')
            else:
                axes[0, 0].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['CD'], label=title, linewidth=0.5, color=color)

            # Plot overlay of Normalized CD vs Wavelength
            if i == reference_index:
                axes[1, 0].plot(df_normalized['Wavelength'], df_normalized['Normalized CD'], label = f'{title} (Reference)', linewidth=0.5, linestyle='--', color='grey')
            else:
                axes[1, 0].plot(df_normalized['Wavelength'], df_normalized['Normalized CD'], label=title, linewidth=0.5, color=color)  

            # Plot HT vs Wavelength
            axes[0, 1].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['Voltage'], label=title, linewidth=0.5, color=color)
            # Add vertical line at wavelength where voltage = 700
            wavelength_at_700 = df_CD_blank_subtracted1[df_CD_blank_subtracted1['Voltage'] == 700]['Wavelength']
            if not wavelength_at_700.empty:
                axes[0, 1].axvline(x=wavelength_at_700.values[0], color='red', linestyle='--', linewidth=0.5)

            # Plot Absorbance
            axes[0, 2].plot(df_CD_blank_subtracted1['Wavelength'], df_CD_blank_subtracted1['Absorbance'], label=title, linewidth=0.5, color=color)

            # Plot Residuals relative to reference spectrum
            axes[1, 1].plot(df_relative['Wavelength'], df_relative['Relative Normalized CD'], label=title, linewidth=0.5, color=color)

            # Plot additional information on [1, 2] only for the first sample
            if additional_info_list:
                info, sample_name = additional_info_list[0]
                axes[1, 2].text(0.1, 0.9, f"Additional Information for Sample {sample_name}:\n\n{info}", fontsize=8, verticalalignment='top', wrap=True)
                axes[1, 2].axis('off')  # Turn off the axis for this subplot

        # Set titles, labels, and other properties for the axes
        axes[0, 0].set_title('Overlay of CD vs. Wavelength')
        axes[0, 0].set_xlabel('Wavelength')
        axes[0, 0].set_ylabel('CD millidegrees')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[0, 0].legend()

        axes[0, 2].set_title('Overlay of Absorbance vs. Wavelength')
        axes[0, 2].set_xlabel('Wavelength')
        axes[0, 2].set_ylabel('Abs.')
        axes[0, 2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[0, 2].legend()

        axes[0, 1].set_title('Overlay of HT vs. Wavelength')
        axes[0, 1].set_xlabel('Wavelength (nm)')
        axes[0, 1].set_ylabel('HT (Voltage)')
        axes[0, 1].legend()

        axes[1, 0].set_title('Normalized CD')
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 0].set_ylabel('Normalized Ellipticity')
        axes[1, 0].legend()

        axes[1, 1].set_title('Residuals relative to Reference')
        axes[1, 1].set_xlabel('Wavelength (nm)')
        axes[1, 1].set_ylabel('Difference from Reference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[1, 1].legend()

        # Use tight layout to adjust spacing between subplots
        plt.tight_layout()

        # Save the plots to the PDF
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
            messagebox.showerror("Input Error", "Please enter a title for the graph.")
            return
        sample_blank_pairs.append((sample_file, blank_file))
        titles.append(title)

        samples_listbox.insert(tk.END, f"Sample: {os.path.basename(sample_file)} | Blank: {os.path.basename(blank_file)} | Title: {title}")
        title_entry.delete(0, tk.END)

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
            reference_index = reference_index_entry.get().strip()
            if not reference_index.isdigit() or int(reference_index) < 0 or int(reference_index) >= len(sample_blank_pairs):
                messagebox.showerror("Input Error", "Please enter a valid reference index.")
                return
            reference_index = int(reference_index)
            plot_overlays(output_dir, pdf_pages, titles, reference_index, *(file for pair in sample_blank_pairs for file in pair))

        messagebox.showinfo("Success", f"Plots and tables have been saved to {output_path}")

    # GUI setup
    root = tk.Tk()
    root.title("CD Single Temperature Analysis")
    root.geometry("800x700")  # Set the window size to 800x700 pixels

    frame = tk.Frame(root, bg="lightgray")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Label and Entry for graph title
    title_label = tk.Label(frame, text="Graph Title:", bg="lightgray")
    title_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    title_entry = tk.Entry(frame, width=50)
    title_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to add sample and blank pair
    add_pair_button = tk.Button(frame, text="Add Sample and Blank Pair", command=add_sample_blank_pair)
    add_pair_button.grid(row=0, column=2, padx=10, pady=10)

    # Listbox to display added sample and blank pairs
    samples_listbox = tk.Listbox(frame, width=100, height=10)
    samples_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for output PDF filename
    output_filename_label = tk.Label(frame, text="Output PDF Filename:", bg="lightgray")
    output_filename_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    output_filename_entry = tk.Entry(frame, width=50)
    output_filename_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for reference index
    reference_index_label = tk.Label(frame, text="Reference Index:", bg="lightgray")
    reference_index_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
    reference_index_entry = tk.Entry(frame, width=50)
    reference_index_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for molar concentration
    molar_conc_label = tk.Label(frame, text="Molar Concentration (M):", bg="lightgray")
    molar_conc_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
    molar_conc_entry = tk.Entry(frame, width=50)
    molar_conc_entry.grid(row=4, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for number of residues
    no_residues_label = tk.Label(frame, text="Number of Residues:", bg="lightgray")
    no_residues_label.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W)
    no_residues_entry = tk.Entry(frame, width=50)
    no_residues_entry.grid(row=5, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for path length
    path_length_label = tk.Label(frame, text="Path Length (cm):", bg="lightgray")
    path_length_label.grid(row=6, column=0, padx=10, pady=10, sticky=tk.W)
    path_length_entry = tk.Entry(frame, width=50)
    path_length_entry.grid(row=6, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to generate PDF
    generate_pdf_button = tk.Button(frame, text="Generate PDF", command=generate_pdf)
    generate_pdf_button.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

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
    
# Function to handle CD Single MW Sequence
def cd_single_mw_sequence():
    
    # Function to read CSV file and extract numeric and additional data
    def read_csv_until_string(file_path, start_string="XYDATA", use_cols=[0, 1, 2]):
        numeric_data = {'Wavelength': [], 'CD': [], 'Voltage': []}
        additional_data = []
        read_numeric_data = False

        # Row names to keep in the additional data
        row_names_to_keep = [
            "Sample name", "Creation date", "Start", "End", "Data interval",
            "Instrument name", "Model name", "Serial No.", "Temperature",
            "Measurement date", "Data pitch", "CD scale", "FL scale", "D.I.T.",
            "Bandwidth", "Start mode", "Scanning speed", "Accumulations"
        ]
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if not read_numeric_data:
                        if start_string in line:
                            read_numeric_data = True
                        else:
                            cols = line.strip().split(',')
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
                    else:
                        # Skip the "XYDATA" line itself, start reading the next line
                        if start_string in line:
                            continue

                        cols = line.strip().split(',')
                        try:
                            # Extract numeric data
                            wavelength = float(cols[use_cols[0]].strip())
                            cd = float(cols[use_cols[1]].strip())
                            voltage = float(cols[use_cols[2]].strip())
                            numeric_data['Wavelength'].append(wavelength)
                            numeric_data['CD'].append(cd)
                            numeric_data['Voltage'].append(voltage)
                        except (ValueError, IndexError):
                            # Handle rows that don't contain numeric data
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        # Convert lists to DataFrames
        df_numeric = pd.DataFrame(numeric_data)
        df_additional = pd.DataFrame(additional_data, columns=['Property', 'Value'])
        return df_numeric, df_additional

    # Function to calculate molar extinction coefficient from peptide sequence
    def calculate_molar_extinction_coefficient(peptide_sequence):
        # Calculate the molar extinction coefficient using the formula
        molar_extinction_coefficient = (
            peptide_sequence.count('W') * 5500 +
            peptide_sequence.count('Y') * 1490 +
            peptide_sequence.count('C') * 125
        )
        
        return molar_extinction_coefficient

    # Function to calculate molar concentration from absorbance and peptide sequence
    def calculate_molar_concentration(absorbance, peptide_sequence, nanodrop_path_length):
        molar_extinction_coefficient = calculate_molar_extinction_coefficient(peptide_sequence)
        return absorbance / (molar_extinction_coefficient * nanodrop_path_length)

    # Function to plot CD vs Wavelength
    def plot_cd_vs_wavelength(output_dir, pdf_pages, graph_titles, output_filename_label, *csv_files):
        num_pairs = len(csv_files) // 6
        additional_data_tables = []

        for i in range(num_pairs):
            csv1, csv2, absorbance, peptide_sequence, nanodrop_path_length, cd_path_length = csv_files[i * 6], csv_files[i * 6 + 1], float(csv_files[i * 6 + 2]), csv_files[i * 6 + 3], float(csv_files[i * 6 + 4]), float(csv_files[i * 6 + 5])
            molar_conc = calculate_molar_concentration(absorbance, peptide_sequence, nanodrop_path_length)
            no_residues = len(peptide_sequence)
            sample_name1 = os.path.splitext(os.path.basename(csv1))[0]
            sample_name1 = re.sub(r'\d+', '', sample_name1)

            # Read sample and blank CSV files
            df_CD1, df_additional1 = read_csv_until_string(csv1)
            df_CD1.columns = ['Wavelength', 'CD', 'Voltage']
            df_CDblank1, df_additional_blank1 = read_csv_until_string(csv2)
            df_CDblank1.columns = ['Wavelength', 'CD', 'Voltage']

            # Subtract blank CD values from sample CD values
            df_CD_blank_subtracted1 = df_CD1.copy()
            df_CD_blank_subtracted1['CD'] = df_CD1['CD'] - df_CDblank1['CD']

            # Convert Millidegrees to Molar Ellipticity
            conversion_factor = 1 / (cd_path_length * molar_conc * no_residues)
            df_CD_blank_subtracted1['CD'] = df_CD_blank_subtracted1['CD'] * conversion_factor

            # Calculate mean residue ellipticity
            df_CD_blank_subtracted1['MRE'] = df_CD_blank_subtracted1['CD'] / 3298

            # Filter out CD data associated with voltage exceeding 700
            voltage_threshold = 700
            mask = df_CD_blank_subtracted1['Voltage'] <= voltage_threshold
            df_CD_blank_subtracted_filtered1 = df_CD_blank_subtracted1[mask]

            # Get title for the graph
            title = graph_titles[i] if i < len(graph_titles) else f'Sample {i+1}'

            # Plot CD vs Wavelength
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_CD_blank_subtracted_filtered1['Wavelength'], df_CD_blank_subtracted_filtered1['CD'], color='blue', marker='o', markersize=0, linestyle='-')
            ax.set_title(f'CD vs. Wavelength ({title})')
            ax.set_xlabel('Wavelength')
            ax.set_ylabel('Molar Ellipticity')
            ax.grid(True)
            ax.axhline(y=0, color='black', linestyle='--')
            pdf_pages.savefig(fig)
            plt.close(fig)

            # Plot HT vs Wavelength
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_CD_blank_subtracted_filtered1['Wavelength'], df_CD_blank_subtracted_filtered1['Voltage'], color='blue', marker='o', markersize=0, linestyle='-')
            ax.set_title(f'HT vs. Wavelength ({title})')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('HT (Voltage)')
            ax.grid(True)
            ax.axhline(y=700, color='black', linestyle='--')
            pdf_pages.savefig(fig)
            plt.close(fig)

            # Store additional data tables for later
            additional_data_tables.append((df_additional1, os.path.basename(csv1)))

            # Export processed data to CSV with the same filename as the PDF
            export_file_path = os.path.join(output_dir, f"{output_filename_label}_{sample_name1}_processed.csv")
            df_CD_blank_subtracted_filtered1.to_csv(export_file_path, columns=['Wavelength', 'Voltage', 'CD', 'MRE'], index=False)

        # Plot overlay of CD vs Wavelength
        overlay_fig, overlay_ax = plt.subplots(figsize=(10, 6))
        for i in range(num_pairs):
            csv1, csv2, absorbance, peptide_sequence, nanodrop_path_length, cd_path_length = csv_files[i * 6], csv_files[i * 6 + 1], float(csv_files[i * 6 + 2]), csv_files[i * 6 + 3], float(csv_files[i * 6 + 4]), float(csv_files[i * 6 + 5])
            molar_conc = calculate_molar_concentration(absorbance, peptide_sequence, nanodrop_path_length)
            no_residues = len(peptide_sequence)
            sample_name1 = os.path.splitext(os.path.basename(csv1))[0]
            sample_name1 = re.sub(r'\d+', '', sample_name1)

            # Read sample and blank CSV files
            df_CD1, _ = read_csv_until_string(csv1)
            df_CD1.columns = ['Wavelength', 'CD', 'Voltage']
            df_CDblank1, _ = read_csv_until_string(csv2)
            df_CDblank1.columns = ['Wavelength', 'CD', 'Voltage']
            
            # Subtract blank CD values from sample CD values
            df_CD_blank_subtracted1 = df_CD1.copy()
            df_CD_blank_subtracted1['CD'] = df_CD1['CD'] - df_CDblank1['CD']

            # Convert Millidegrees to Molar Ellipticity
            conversion_factor = 1 / (cd_path_length * molar_conc * no_residues)
            df_CD_blank_subtracted1['CD'] = df_CD_blank_subtracted1['CD'] * conversion_factor

            # Filter out CD data associated with voltage exceeding 700
            voltage_threshold = 700
            mask = df_CD_blank_subtracted1['Voltage'] <= voltage_threshold
            df_CD_blank_subtracted_filtered1 = df_CD_blank_subtracted1[mask]

            # Get title for the graph
            title = graph_titles[i] if i < len(graph_titles) else f'Sample {i+1}'

            # Plot overlay of CD vs Wavelength
            overlay_ax.plot(df_CD_blank_subtracted_filtered1['Wavelength'], df_CD_blank_subtracted_filtered1['CD'], label=title)

        overlay_ax.set_title('Overlay of CD vs. Wavelength')
        overlay_ax.set_xlabel('Wavelength')
        overlay_ax.set_ylabel('Molar Ellipticity')
        overlay_ax.grid(True)
        overlay_ax.axhline(y=0, color='black', linestyle='--')
        overlay_ax.legend()
        pdf_pages.savefig(overlay_fig)
        plt.close(overlay_fig)

        # Add all additional data tables at the end
        for df_additional, filename in additional_data_tables:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            table_data = df_additional
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.2)
            ax.set_title(f'Additional Data for {filename}')
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
        try:
            absorbance = float(absorbance_entry.get().strip())
            peptide_sequence = peptide_sequence_entry.get().strip()
            nanodrop_path_length = float(nanodrop_path_length_entry.get().strip())
            cd_path_length = float(cd_path_length_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for Absorbance and Path Lengths, and a valid peptide sequence.")
            return

        sample_blank_pairs.append((sample_file, blank_file, absorbance, peptide_sequence, nanodrop_path_length, cd_path_length))
        titles.append(title)

        samples_listbox.insert(tk.END, f"Sample: {os.path.basename(sample_file)} | Blank: {os.path.basename(blank_file)} | Title: {title}")
        title_entry.delete(0, tk.END)
        absorbance_entry.delete(0, tk.END)
        peptide_sequence_entry.delete(0, tk.END)
        nanodrop_path_length_entry.delete(0, tk.END)
        cd_path_length_entry.delete(0, tk.END)

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
            if sample_blank_pairs:
                plot_cd_vs_wavelength(output_dir, pdf_pages, titles, output_filename, *(file for pair in sample_blank_pairs for file in pair))
            else:
                messagebox.showerror("Input Error", "No valid sample and blank pairs provided.")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            messagebox.showinfo("Success", f"Plots and tables have been saved to {output_path}")
        else:
            messagebox.showerror("Error", f"Failed to save plots and tables to {output_path}")

    # GUI setup
    root = tk.Tk()
    root.title("CD Single Temperature Analysis")
    root.geometry("800x600")  # Set the window size to 800x600 pixels

    frame = tk.Frame(root, bg="lightgray")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Label and Entry for graph title
    title_label = tk.Label(frame, text="Graph Title:", bg="lightgray")
    title_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    title_entry = tk.Entry(frame, width=50)
    title_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for absorbance
    absorbance_label = tk.Label(frame, text="Absorbance (A):", bg="lightgray")
    absorbance_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
    absorbance_entry = tk.Entry(frame, width=50)
    absorbance_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for peptide sequence
    peptide_sequence_label = tk.Label(frame, text="Peptide Sequence:", bg="lightgray")
    peptide_sequence_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    peptide_sequence_entry = tk.Entry(frame, width=50)
    peptide_sequence_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for Nanodrop path length
    nanodrop_path_length_label = tk.Label(frame, text="Nanodrop Path Length (cm):", bg="lightgray")
    nanodrop_path_length_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
    nanodrop_path_length_entry = tk.Entry(frame, width=50)
    nanodrop_path_length_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for CD path length
    cd_path_length_label = tk.Label(frame, text="CD Path Length (mm):", bg="lightgray")
    cd_path_length_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
    cd_path_length_entry = tk.Entry(frame, width=50)
    cd_path_length_entry.grid(row=4, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to add sample and blank pair
    add_pair_button = tk.Button(frame, text="Add Sample and Blank Pair", command=add_sample_blank_pair)
    add_pair_button.grid(row=5, column=2, padx=10, pady=10)

    # Listbox to display added sample and blank pairs
    samples_listbox = tk.Listbox(frame, width=100, height=10)
    samples_listbox.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for output PDF filename
    output_filename_label = tk.Label(frame, text="Output PDF Filename:", bg="lightgray")
    output_filename_label.grid(row=7, column=0, padx=10, pady=10, sticky=tk.W)
    output_filename_entry = tk.Entry(frame, width=50)
    output_filename_entry.grid(row=7, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to generate PDF
    generate_pdf_button = tk.Button(frame, text="Generate PDF", command=generate_pdf)
    generate_pdf_button.grid(row=7, column=2, padx=10, pady=10)

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

def cd_single_wsd_auc():
    
   # Function to read CSV file and extract numeric and additional data
    def read_csv_until_string(file_path, start_string="XYDATA", use_cols=[0, 1, 2, 3], window_length=100, polyorder=3):
        numeric_data = {'Wavelength': [], 'CD': [], 'Voltage': [], 'Absorbance': []}
        additional_data = []
        read_numeric_data = False

        row_names_to_keep = [
            "Sample name", "Creation date", "Start", "End", "Data interval",
            "Instrument name", "Model name", "Serial No.", "Temperature",
            "Measurement date", "Data pitch", "CD scale", "FL scale", "D.I.T.",
            "Bandwidth", "Start mode", "Scanning speed", "Accumulations"
        ]
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if not read_numeric_data:
                        if start_string in line:
                            read_numeric_data = True
                        else:
                            cols = line.strip().split(',')
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
                    else:
                        if start_string in line:
                            continue

                        cols = line.strip().split(',')
                        try:
                            wavelength = float(cols[use_cols[0]].strip())
                            cd = float(cols[use_cols[1]].strip())
                            voltage = float(cols[use_cols[2]].strip())
                            absorbance = float(cols[use_cols[3]].strip())                          
                            numeric_data['Wavelength'].append(wavelength)
                            numeric_data['CD'].append(cd)
                            numeric_data['Voltage'].append(voltage)
                            numeric_data['Absorbance'].append(absorbance)
                            
                            
                        except (ValueError, IndexError):
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        df_numeric = pd.DataFrame(numeric_data)
        df_additional = pd.DataFrame(additional_data, columns=['Property', 'Value'])
        
        # Apply Savgol Smoothing
        df_numeric['CD'] = savgol_filter(df_numeric['CD'], window_length=window_length, polyorder=polyorder)        

        return df_numeric, df_additional
        
    def process_cd_data(sample_file, blank_file):
        """
        Processes CD data by loading, subtracting blanks, filtering, and normalizing the data,
        and fills missing values by averaging the two neighboring values.

        Parameters:
        - sample_file: Path to the sample CSV file
        - blank_file: Path to the blank CSV file
        - absorbance_threshold: The minimum absorbance value to use for normalization

        Returns:
        - normalized_df: DataFrame with normalized and blank-subtracted CD data
        """
        # Load sample and blank data
        df_CD1, _ = read_csv_until_string(sample_file)
        df_CDblank1, _ = read_csv_until_string(blank_file)

        # Subtract blank from sample data
        df_CD_blank_subtracted1 = df_CD1.copy()
        df_CD_blank_subtracted1['CD'] = df_CD1['CD'] - df_CDblank1['CD']
        df_CD_blank_subtracted1['Absorbance'] = df_CD1['Absorbance'] - df_CDblank1['Absorbance']
        
        # Normalize the data by auc
        normalized_df = normalize_by_area(df_CD_blank_subtracted1)       
        
        # Ensure data is sorted by wavelength
        normalized_df = normalized_df.sort_values('Wavelength').reset_index(drop=True)

        return normalized_df
            
    def normalize_by_area(df, target_auc=1.0):
        """
        Normalize the CD data by the area under the curve (AUC) between the spectrum and the x-axis over a specified wavelength range.
        The normalization ensures that the area is equal to a target value (e.g., 1).

        Parameters:
        - df: DataFrame containing 'Wavelength' and 'CD' columns.
        - target_auc: The desired area under the curve for normalization (default is 1.0).

        Returns:
        - df_normalized: DataFrame with an additional column 'Normalized CD' containing the normalized CD data.
        """
        # Ensure the data is sorted by Wavelength
        df = df.sort_values('Wavelength')
               
        # Calculate the area under the curve using the trapezoidal rule
        auc = np.trapz(np.abs(df['CD']), df['Wavelength'])
        
        # Handle cases where AUC is very small or zero
        if auc < 1e-10:
            raise ValueError("Area under the curve is too small or zero, cannot normalize.")
        
        # Normalize the CD values by the target area
        df['Normalized CD'] = df['CD'] / auc * target_auc

        return df
        
        
    def calculate_wsd(df, avg_spectrum, reference_spectra_std):
        """
        Calculate the Weighted Spectral Distance (WSD) relative to the average spectrum.

        Parameters:
        - df: DataFrame containing the smoothed area normalised CD data of the current spectrum.
        - avg_spectrum: DataFrame containing the average smoothed area normalised CD spectrum.
        - reference_spectra_std: DataFrame containing the standard deviation of the reference group spectra.

        Returns:
        - wsd_value: The calculated WSD value.
        """
        # Merge the current spectrum with the average spectrum and standard deviation on 'Wavelength'
        df_merged = pd.merge(df[['Wavelength', 'Normalized CD']], avg_spectrum, on='Wavelength', suffixes=('', '_avg'))
        df_merged = pd.merge(df_merged, reference_spectra_std, on='Wavelength', suffixes=('', '_std'))

        # Calculate the squared difference between the current spectrum and the average spectrum
        squared_diff = (df_merged['Normalized CD'] - df_merged['Normalized CD_avg']) ** 2

        # Calculate WSD, weighting each squared difference by the point-specific standard deviation
        # Avoid division by zero by checking if the standard deviation is non-zero at each point
        weighted_squared_diff = squared_diff / df_merged['Normalized CD_std']
        
        # Sum the weighted differences and take the square root
        wsd_value = np.sqrt(np.sum(weighted_squared_diff) / len(df_merged)) if len(df_merged) > 0 else np.nan

        return wsd_value


    def calculate_wsd_values(replicate_groups, reference_index):
        """
        Calculate WSD values for each spectrum relative to the average spectrum of the reference group.
        Store these values in a dictionary and return it.

        Parameters:
        - replicate_groups: List of tuples where each tuple contains a list of (sample_file, blank_file) and a group title.
        - reference_index: Index of the reference group in the replicate_groups list.

        Returns:
        - wsd_values_by_group: Dictionary where keys are group titles and values are lists of WSD values for each spectrum.
        """
        wsd_values_by_group = {}

        # Handle Reference Replicate Group First
        reference_group_files, reference_title = replicate_groups[reference_index]
        reference_normalized_spectra = []

        for sample_file, blank_file in reference_group_files:
            normalized_df = process_cd_data(sample_file, blank_file)
            reference_normalized_spectra.append(normalized_df[['Wavelength', 'Normalized CD']])

        if reference_normalized_spectra:
            reference_normalized_df_list = pd.concat(reference_normalized_spectra)
            reference_normalized_df_list = reference_normalized_df_list.sort_values('Wavelength').reset_index(drop=True)
            
            # Calculate the average spectrum
            avg_normalized_spectrum = reference_normalized_df_list.groupby('Wavelength').agg({'Normalized CD': 'mean'}).reset_index()

            # Calculate the standard deviation across the spectra in the reference group at each wavelength
            reference_spectra_std = reference_normalized_df_list.groupby('Wavelength').agg({'Normalized CD': 'std'}).reset_index()
            reference_spectra_std.rename(columns={'Normalized CD': 'Normalized CD_std'}, inplace=True)
            
            # Calculate WSD values for the reference group
            wsd_values_by_group[f"{reference_title} (Reference)"] = []
            for sample_file, blank_file in reference_group_files:
                normalized_df = process_cd_data(sample_file, blank_file)
                wsd_value = calculate_wsd(normalized_df, avg_normalized_spectrum, reference_spectra_std)
                wsd_values_by_group[f"{reference_title} (Reference)"].append(wsd_value)

        # Handle Other Replicate Groups
        for i, (replicate_files, title) in enumerate(replicate_groups):
            if title == reference_title:
                continue  # Skip the reference group

            # Calculate WSD values for each spectrum relative to the reference group's average spectrum
            wsd_values_by_group[title] = []
            for sample_file, blank_file in replicate_files:
                normalized_df = process_cd_data(sample_file, blank_file)
                wsd_value = calculate_wsd(normalized_df, avg_normalized_spectrum, reference_spectra_std)
                wsd_values_by_group[title].append(wsd_value)                  

        return wsd_values_by_group
        
        
    def perform_and_plot_t_tests(wsd_values_by_group, reference_title, ax):
        """
        Perform t-tests comparing each group's WSD values to the reference group's WSD values,
        and plot the results on the provided axis.

        Parameters:
        - wsd_values_by_group: Dictionary with group titles as keys and lists of WSD values as values.
        - reference_title: Title of the reference group.
        - ax: The axis on which to plot the results.
        """
        reference_wsd = wsd_values_by_group.get(reference_title, [])
        if not reference_wsd:
            return

        reference_wsd_mean = np.mean(reference_wsd)
        reference_wsd_std = np.std(reference_wsd)

        # Plot vertical lines for reference WSD
        ax.axvline(reference_wsd_mean, color='red', linestyle='--', label='Reference Mean')
        ax.axvline(reference_wsd_mean + 2 * reference_wsd_std, color='black', linestyle=':', label='+2 Std Dev')
        ax.axvline(reference_wsd_mean - 2 * reference_wsd_std, color='black', linestyle=':', label='-2 Std Dev')

        # Perform t-tests and plot means with whiskers
        for group_title, wsd_values in wsd_values_by_group.items():
            if group_title == reference_title:
                continue  # Skip the reference group itself

            # Calculate mean and std for the current group
            mean_wsd = np.mean(wsd_values)
            std_wsd = np.std(wsd_values)            

            # Calculate y-position for this group in the scatter plot
            y_position = list(wsd_values_by_group.keys()).index(group_title)

            # Plot mean and error bars (whiskers) for the group
            ax.errorbar(mean_wsd, y_position, 
                        xerr=2 * std_wsd, 
                        fmt='o',  # Circle marker
                        color='blue', 
                        label=group_title + ' Mean ± 2 Std Dev' if y_position == 0 else "",  # Only label first instance
                        capsize=5)

            # Perform t-tests
            t_stat, p_value = ttest_ind(reference_wsd, wsd_values, equal_var=False)
            

            # Adjusted x coordinate to shift the text to the right of the plot
            x_offset = ax.get_xlim()[1] * 0.1  # 10% of the x-axis range as offset
            
            # Plot t-test results
            ax.text(ax.get_xlim()[1] + x_offset, y_position, 
                    f"Group: {group_title}\nT-stat: {t_stat:.2f}\nP-value: {p_value:.6f}",
                    ha='left', va='center', fontsize=8, color='black')

    

    def calculate_residuals_gfactor(df, reference_df):
        """
        Calculate residuals between the g factor of the current spectrum and the average g factor spectrum.

        Parameters:
        - df: DataFrame containing the g factor data of the current spectrum.
        - reference_df: DataFrame containing the average g factor data from the reference spectra.

        Returns:
        - df: DataFrame with an additional column 'Residual' showing the difference between the current spectrum and the average spectrum.
        """
        # Merge the current spectrum with the average spectrum on 'Wavelength'
        df_merged = pd.merge(df[['Wavelength', 'Normalized CD']], reference_df[['Wavelength', 'Normalized CD']], on='Wavelength', suffixes=('', '_avg'))

        # Calculate the residuals
        df_merged['Residual'] = df_merged['Normalized CD'] - df_merged['Normalized CD_avg']

        return df_merged
        

                                    
    def plot_overlays(output_dir, pdf_pages, replicate_groups, reference_index):
        
        # Handle Reference Replicate Group First
        reference_group_files, reference_title = replicate_groups[reference_index]
        reference_normalized_spectra = []
        reference_blank_spectra = []
        
        # Define color map and styles
        colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(replicate_groups)))  # Shades of blue
        color_map = {f"{reference_title} (Reference)": 'black'}  # Initialize color map with reference color
        linestyle_map = {'Reference': 'dashdot', 'Normal': '-'}

        fig, axes = plt.subplots(3, 3, figsize=(46, 18))  # Updated to 3x3 for the additional plot

        # Initialize dictionaries to store average Normalized CD, WSD values, and average residuals
        avg_spectra_by_group = {}
        wsd_values_by_group = {}
        avg_residuals_by_group = {}

        # Initialize a dictionary to track if the label was already added for each group
        label_added1 = {}; label_added2 = {}; label_added3 = {}; label_added4 = {}; label_added5 = {}; label_added6 = {}

        # Calculate WSD values
        wsd_values_by_group = calculate_wsd_values(replicate_groups, reference_index)       

        for sample_file, blank_file in reference_group_files:
            normalized_df = process_cd_data(sample_file, blank_file)
            reference_normalized_spectra.append(normalized_df[['Wavelength', 'Absorbance', 'CD', 'Normalized CD', 'Voltage']])

            # Store the blank CD data
            blank_df, _ = read_csv_until_string(blank_file)
            reference_blank_spectra.append(blank_df[['Wavelength', 'CD']])

        if reference_normalized_spectra:
            reference_normalized_df_list = pd.concat(reference_normalized_spectra)
            reference_normalized_df_list = reference_normalized_df_list.sort_values('Wavelength').reset_index(drop=True)
            avg_normalized_spectrum = reference_normalized_df_list.groupby('Wavelength').agg({'Normalized CD': 'mean'}).reset_index()
            std_normalized_spectrum = reference_normalized_df_list.groupby('Wavelength').agg({'Normalized CD': 'std'}).reset_index()

            # Plot individual CD spectra for the reference replicates on [0, 0]
            for df in reference_normalized_spectra:
                axes[0, 0].plot(df['Wavelength'], df['CD'], color='gray', linestyle=':', alpha=0.7, linewidth=0.2, label=f"{reference_title} (Reference)" if reference_title not in label_added1 else "")
                label_added1[reference_title] = True  # Mark label as added

            # Plot individual Normalized CD spectra for the reference replicates on [1, 0]
            for df in reference_normalized_spectra:
                axes[1, 0].plot(df['Wavelength'], df['Normalized CD'], color='gray', linestyle=':', alpha=0.2, linewidth=0.2, label=f"{reference_title} (Reference)" if reference_title not in label_added2 else "")
                label_added2[reference_title] = True

            # Plot individual Absorbance spectra for the reference replicates on [0, 2]
            for df in reference_normalized_spectra:
                axes[0, 2].plot(df['Wavelength'], df['Absorbance'], color='gray', linestyle=':', alpha=0.2, linewidth=0.2, label=f"{reference_title} (Reference)" if reference_title not in label_added3 else "")
                label_added3[reference_title] = True

            # Plot reference blank CD on [2, 2]
            for blank_df in reference_blank_spectra:
                axes[2, 2].plot(blank_df['Wavelength'], blank_df['CD'], color='gray', linestyle='--', linewidth=0.2, alpha=0.7, label=f"{reference_title} (Reference)" if reference_title not in label_added4 else "")
                label_added4[reference_title] = True
                
            # Plot Voltage on [0, 1]
            for df in reference_normalized_spectra:
                axes[0, 1].plot(df['Wavelength'], df['Voltage'], color='gray', linestyle='--', linewidth=0.2, alpha=0.7, label=f"{reference_title} (Reference)" if reference_title not in label_added5 else "")
                label_added5[reference_title] = True

            # Plot average Normalized CD spectrum for reference group with shaded region for ±3 std dev on [2, 0]
            axes[2, 0].plot(avg_normalized_spectrum['Wavelength'], avg_normalized_spectrum['Normalized CD'], color='black', linestyle='--', label=f"{reference_title} (Reference)")
            axes[2, 0].fill_between(
                avg_normalized_spectrum['Wavelength'],
                avg_normalized_spectrum['Normalized CD'] - 3 * std_normalized_spectrum['Normalized CD'],
                avg_normalized_spectrum['Normalized CD'] + 3 * std_normalized_spectrum['Normalized CD'],
                color='red', alpha=0.2, label='±3 Std Dev')

            # Plot + or - 3 Standard Deviations around Residuals plot
            axes[2, 1].fill_between(
                avg_normalized_spectrum['Wavelength'],
                -3 * std_normalized_spectrum['Normalized CD'],  # Lower bound for shading
                3 * std_normalized_spectrum['Normalized CD'],   # Upper bound for shading
                color='red', alpha=0.2, label='±3 Std Dev')

        # Handle Other Replicate Groups
        for i, (replicate_files, title) in enumerate(replicate_groups):
            if title == reference_title:
                continue  # Skip the reference group

            color = colors[i]  # Use color map to get a color
            linestyle = linestyle_map.get(title, '-')
            color_map[title] = color
            spectra_list = []
            reference_blank_spectra = []
            
            # Initialize a dictionary to track if the label was already added for each group
            label_added1 = {}; label_added2 = {}; label_added3 = {}; label_added4 = {}; label_added5 = {}; label_added6 = {}
                       

            for sample_file, blank_file in replicate_files:
                normalized_df = process_cd_data(sample_file, blank_file)
                normalized_df['Title'] = title

                # Store the blank CD data
                blank_df, _ = read_csv_until_string(blank_file)
                reference_blank_spectra.append(blank_df[['Wavelength', 'CD']])

                # Plot CD data
                axes[0, 0].plot(normalized_df['Wavelength'], normalized_df['CD'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added1 else "")
                label_added1[title] = True

                # Plot Voltage 
                axes[0, 1].plot(normalized_df['Wavelength'], normalized_df['Voltage'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added2 else "")
                label_added2[title] = True
                
                # Plot Normalized CD spectra
                axes[1, 0].plot(normalized_df['Wavelength'], normalized_df['Normalized CD'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added3 else "")
                label_added3[title] = True
                
                # Calculate and plot Normalized CD Residuals                 
                residuals_df = calculate_residuals_gfactor(normalized_df, avg_normalized_spectrum)
                axes[1, 1].plot(residuals_df['Wavelength'], residuals_df['Residual'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added4 else "")
                label_added4[title] = True
                
                # Plot absorbance
                axes[0, 2].plot(normalized_df['Wavelength'], normalized_df['Absorbance'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added5 else "")
                label_added5[title] = True
                
                # Append to spectra list
                spectra_list.append(normalized_df[['Wavelength', 'Normalized CD']])

            # Calculate and plot the average Normalized CD spectra for each replicate group
            if spectra_list:
                concatenated_df = pd.concat(spectra_list)
                concatenated_df = concatenated_df.sort_values('Wavelength').reset_index(drop=True)
                avg_spectra = concatenated_df.groupby('Wavelength').agg({'Normalized CD': 'mean'}).reset_index()
                avg_spectra_by_group[title] = avg_spectra
                axes[2, 0].plot(avg_spectra['Wavelength'], avg_spectra['Normalized CD'], color=color, linestyle=linestyle, linewidth=0.2, label=title)

                # Calculate average residuals for each replicate group relative to the reference group
                merged_df = avg_spectra.merge(avg_normalized_spectrum, on='Wavelength', suffixes=('', '_ref'))
                merged_df['Residual'] = merged_df['Normalized CD'] - merged_df['Normalized CD_ref']
                avg_residuals_by_group[title] = merged_df[['Wavelength', 'Residual']]
            
                # Plot reference blank CD on [2, 2]
                for blank_df in reference_blank_spectra:
                    axes[2, 2].plot(blank_df['Wavelength'], blank_df['CD'], color='black', linestyle='--', linewidth=0.2, alpha=0.7, label=title if title not in label_added6 else "")
                    label_added6[title] = True

        # Plot average residuals
        for i, (group_title, residuals_df) in enumerate(avg_residuals_by_group.items()):
            y = len(avg_residuals_by_group) - i - 1
            axes[2, 1].plot(residuals_df['Wavelength'], residuals_df['Residual'], label=group_title, linewidth=0.2, color=color_map.get(group_title, 'blue'))
            
                       

        # Plot WSD values as a scatter plot with WSD on x-axis and groups on y-axis
        wsd_df = pd.DataFrame([(group, wsd) for group, wsd_values in wsd_values_by_group.items() for wsd in wsd_values], columns=['Group', 'WSD'])
        sns.scatterplot(x='WSD', y='Group', data=wsd_df, ax=axes[1, 2], hue='Group', palette=color_map, s=100)
        axes[1, 2].set_title('WSD Values')
        axes[1, 2].set_xlabel('WSD Value')
        axes[1, 2].set_ylabel('Group')
             
        # Perform and plot t-tests
        perform_and_plot_t_tests(wsd_values_by_group, f"{reference_title} (Reference)", axes[1, 2])


        # Set titles and axis labels for each subplot
        axes[0, 0].set_title('CD Data')
        axes[0, 0].set_xlabel('Wavelength')
        axes[0, 0].set_ylabel('CD (millidegrees)')

        axes[0, 1].set_title('HT Voltage')
        axes[0, 1].set_xlabel('Wavelength')
        axes[0, 1].set_ylabel('Voltage')

        axes[0, 2].set_title('Absorbance')
        axes[0, 2].set_xlabel('Wavelength')
        axes[0, 2].set_ylabel('Absorbance')

        axes[1, 0].set_title('Normalised CD')
        axes[1, 0].set_xlabel('Wavelength')   

        axes[1, 1].set_title('Normalised CD Residuals')
        axes[1, 1].set_xlabel('Wavelength')
        axes[1, 1].set_ylabel('Residual')

        axes[2, 0].set_title('Normalised CD (Average)')
        axes[2, 0].set_xlabel('Wavelength')

        axes[2, 1].set_title('Residuals (Average Spectra)')
        axes[2, 1].set_xlabel('Wavelength')
        axes[2, 1].set_ylabel('Residual')

        axes[2, 2].set_title('Blank CD Spectra')
        axes[2, 2].set_xlabel('Wavelength')
        axes[2, 2].set_ylabel('CD (millidegrees)')
        
        # Adjust line width for each plot and remove grid
        axes[0, 0].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for CD Data
        axes[0, 1].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Voltage
        axes[0, 2].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Absorbance
        axes[1, 0].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Normalized CD Spectra
        axes[1, 1].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Normalized CD Residuals
        axes[2, 0].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Avg Normalized CD Spectrum
        axes[2, 1].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Average Residuals

        # Move legend outside the plot area
        for ax in axes.flat:
            # Move legend outside the plot to the right
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        # Adjust the layout to prevent overlapping
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust right margin for space for legend


        # Save the figure to PDF      
        pdf_pages.savefig(fig)
        plt.close(fig)
        
    # Function to add replicate group
    def add_replicate_group():
        replicate_title = replicate_title_entry.get().strip()
        if not replicate_title:
            messagebox.showerror("Input Error", "Please enter a title for the replicate series.")
            return

        replicate_files = []
        while True:
            sample_file = filedialog.askopenfilename(title="Select Sample CSV File", filetypes=[("CSV files", "*.csv")])
            if not sample_file:
                break
            blank_file = filedialog.askopenfilename(title="Select Blank CSV File", filetypes=[("CSV files", "*.csv")])
            if not blank_file:
                break
            replicate_files.append((sample_file, blank_file))
        
        if not replicate_files:
            messagebox.showerror("Input Error", "Please add at least one sample and blank pair.")
            return
        
        replicate_groups.append((replicate_files, replicate_title))

        samples_listbox.delete(0, tk.END)
        for replicate_files, title in replicate_groups:
            samples_listbox.insert(tk.END, f"Replicate Series: {title} ({len(replicate_files)} pairs)")

        replicate_title_entry.delete(0, tk.END)

    # Function to generate PDF
    def generate_pdf():
        if not replicate_groups:
            messagebox.showerror("Input Error", "Please add at least one replicate group.")
            return
        output_filename = output_filename_entry.get().strip()
        if not output_filename:
            messagebox.showerror("Input Error", "Please enter an output filename.")
            return
        output_path = os.path.join(output_dir, f"{output_filename}.pdf")

        with PdfPages(output_path) as pdf_pages:
            reference_index = reference_index_entry.get().strip()
            if not reference_index.isdigit() or int(reference_index) < 0 or int(reference_index) >= len(replicate_groups):
                messagebox.showerror("Input Error", "Please enter a valid reference index.")
                return
            reference_index = int(reference_index)
            plot_overlays(output_dir, pdf_pages, replicate_groups, reference_index)

        messagebox.showinfo("Success", f"Plots and tables have been saved to {output_path}")

    # GUI setup
    root = tk.Tk()
    root.title("CD Single Temperature Analysis")
    root.geometry("800x600")

    frame = tk.Frame(root, bg="lightgray")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Label and Entry for replicate series title
    replicate_title_label = tk.Label(frame, text="Replicate Series Title:", bg="lightgray")
    replicate_title_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    replicate_title_entry = tk.Entry(frame, width=50)
    replicate_title_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to add replicate group
    add_replicate_group_button = tk.Button(frame, text="Add Replicate Group", command=add_replicate_group)
    add_replicate_group_button.grid(row=0, column=2, padx=10, pady=10)

    # Listbox to display added replicate groups
    samples_listbox = tk.Listbox(frame, width=100, height=10)
    samples_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for output PDF filename
    output_filename_label = tk.Label(frame, text="Output PDF Filename:", bg="lightgray")
    output_filename_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    output_filename_entry = tk.Entry(frame, width=50)
    output_filename_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for reference index
    reference_index_label = tk.Label(frame, text="Reference Index:", bg="lightgray")
    reference_index_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
    reference_index_entry = tk.Entry(frame, width=50)
    reference_index_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to generate PDF
    generate_pdf_button = tk.Button(frame, text="Generate PDF", command=generate_pdf)
    generate_pdf_button.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    # Configure column 1 to expand
    frame.grid_columnconfigure(1, weight=1)

    # Initialize lists for replicate groups
    replicate_groups = []

    # Set output directory
    output_dir = r'Y:\CD\Analysis-Development\Test_Outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Run the GUI loop
    root.mainloop()
    
def cd_single_wsd_gfactor():
    
    # Function to read CSV file and extract numeric and additional data
    def read_csv_until_string(file_path, start_string="XYDATA", use_cols=[0, 1, 2, 3], window_length=100, polyorder=3):
        numeric_data = {'Wavelength': [], 'CD': [], 'Voltage': [], 'Absorbance': []}
        additional_data = []
        read_numeric_data = False

        row_names_to_keep = [
            "Sample name", "Creation date", "Start", "End", "Data interval",
            "Instrument name", "Model name", "Serial No.", "Temperature",
            "Measurement date", "Data pitch", "CD scale", "FL scale", "D.I.T.",
            "Bandwidth", "Start mode", "Scanning speed", "Accumulations"
        ]
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if not read_numeric_data:
                        if start_string in line:
                            read_numeric_data = True
                        else:
                            cols = line.strip().split(',')
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
                    else:
                        if start_string in line:
                            continue

                        cols = line.strip().split(',')
                        try:
                            wavelength = float(cols[use_cols[0]].strip())
                            cd = float(cols[use_cols[1]].strip())
                            voltage = float(cols[use_cols[2]].strip())
                            absorbance = float(cols[use_cols[3]].strip())                          
                            numeric_data['Wavelength'].append(wavelength)
                            numeric_data['CD'].append(cd)
                            numeric_data['Voltage'].append(voltage)
                            numeric_data['Absorbance'].append(absorbance)
                            
                            
                        except (ValueError, IndexError):
                            if cols[0] in row_names_to_keep:
                                additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        df_numeric = pd.DataFrame(numeric_data)
        df_additional = pd.DataFrame(additional_data, columns=['Property', 'Value'])
        
        # Apply Savgol Smoothing
        df_numeric['CD'] = savgol_filter(df_numeric['CD'], window_length=window_length, polyorder=polyorder)        

        return df_numeric, df_additional
        
    def process_cd_data(sample_file, blank_file, absorbance_threshold=0.005):
        """
        Processes CD data by loading, subtracting blanks, filtering, and normalizing the data,
        and fills missing values by averaging the two neighboring values.

        Parameters:
        - sample_file: Path to the sample CSV file
        - blank_file: Path to the blank CSV file
        - absorbance_threshold: The minimum absorbance value to use for normalization

        Returns:
        - normalized_df: DataFrame with normalized and blank-subtracted CD data
        """
        # Load sample and blank data
        df_CD1, _ = read_csv_until_string(sample_file)
        df_CDblank1, _ = read_csv_until_string(blank_file)

        # Subtract blank from sample data
        df_CD_blank_subtracted1 = df_CD1.copy()
        df_CD_blank_subtracted1['CD'] = df_CD1['CD'] - df_CDblank1['CD']
        df_CD_blank_subtracted1['Absorbance'] = df_CD1['Absorbance'] - df_CDblank1['Absorbance']

        # Convert CD millidegrees to delta A (A left - A right)
        df_CD_blank_subtracted1['CD'] = df_CD_blank_subtracted1['CD'] / (1000 * 32.98)
        
        # Replace absorbance values below the threshold with the threshold value
        df_CD_blank_subtracted1['Absorbance'] = df_CD_blank_subtracted1['Absorbance'].apply(lambda x: max(x, absorbance_threshold) if pd.notna(x) else absorbance_threshold)      
        
        # Normalize the data by G factor
        normalized_df = normalize_by_gfactor(df_CD_blank_subtracted1)       
        
        # Ensure data is sorted by wavelength
        normalized_df = normalized_df.sort_values('Wavelength').reset_index(drop=True)

        return normalized_df
            
    def normalize_by_gfactor(df):
        """
        Normalize the CD data by calculating the gfactor spectrum, with handling for low absorbance values.

        Parameters:
        - df: DataFrame containing 'Wavelength', 'CD', and 'Absorbance' columns.
        - absorbance_threshold: The minimum absorbance value below which normalization will use the threshold value.

        Returns:
        - df_normalized: DataFrame with an additional column 'G factor' containing the G factor spectrum
        """

        # Calculate G factor using the updated absorbance values
        df['G factor'] = df['CD'] / df['Absorbance']

        return df
        
    def calculate_wsd(df, avg_spectrum, reference_spectra_std):
        """
        Calculate the Weighted Spectral Distance (WSD) relative to the average spectrum.

        Parameters:
        - df: DataFrame containing the smoothed area normalised CD data of the current spectrum.
        - avg_spectrum: DataFrame containing the average smoothed area normalised CD spectrum.
        - reference_spectra_std: DataFrame containing the standard deviation of the reference group spectra.

        Returns:
        - wsd_value: The calculated WSD value.
        """
        # Merge the current spectrum with the average spectrum and standard deviation on 'Wavelength'
        df_merged = pd.merge(df[['Wavelength', 'G factor']], avg_spectrum, on='Wavelength', suffixes=('', '_avg'))
        df_merged = pd.merge(df_merged, reference_spectra_std, on='Wavelength', suffixes=('', '_std'))

        # Calculate the squared difference between the current spectrum and the average spectrum
        squared_diff = (df_merged['G factor'] - df_merged['G factor_avg']) ** 2

        # Calculate WSD, weighting each squared difference by the point-specific standard deviation
        # Avoid division by zero by checking if the standard deviation is non-zero at each point
        weighted_squared_diff = squared_diff / (df_merged['G factor_std'])
        
        # Sum the weighted differences and take the square root
        wsd_value = np.sqrt(np.sum(weighted_squared_diff) / len(df_merged)) if len(df_merged) > 0 else np.nan

        return wsd_value

    def calculate_wsd_values(replicate_groups, reference_index):
        """
        Calculate WSD values for each spectrum relative to the average spectrum of the reference group.
        Store these values in a dictionary and return it.

        Parameters:
        - replicate_groups: List of tuples where each tuple contains a list of (sample_file, blank_file) and a group title.
        - reference_index: Index of the reference group in the replicate_groups list.

        Returns:
        - wsd_values_by_group: Dictionary where keys are group titles and values are lists of WSD values for each spectrum.
        """
        wsd_values_by_group = {}

        # Handle Reference Replicate Group First
        reference_group_files, reference_title = replicate_groups[reference_index]
        reference_normalized_spectra = []

        for sample_file, blank_file in reference_group_files:
            normalized_df = process_cd_data(sample_file, blank_file)
            reference_normalized_spectra.append(normalized_df[['Wavelength', 'G factor']])

        if reference_normalized_spectra:
            reference_normalized_df_list = pd.concat(reference_normalized_spectra)
            reference_normalized_df_list = reference_normalized_df_list.sort_values('Wavelength').reset_index(drop=True)
            
            # Calculate the average spectrum
            avg_normalized_spectrum = reference_normalized_df_list.groupby('Wavelength').agg({'G factor': 'mean'}).reset_index()

            # Calculate the standard deviation across the spectra in the reference group at each wavelength
            reference_spectra_std = reference_normalized_df_list.groupby('Wavelength').agg({'G factor': 'std'}).reset_index()
            reference_spectra_std.rename(columns={'G factor': 'G factor_std'}, inplace=True)
            
            # Calculate WSD values for the reference group
            wsd_values_by_group[f"{reference_title} (Reference)"] = []
            for sample_file, blank_file in reference_group_files:
                normalized_df = process_cd_data(sample_file, blank_file)
                wsd_value = calculate_wsd(normalized_df, avg_normalized_spectrum, reference_spectra_std)
                wsd_values_by_group[f"{reference_title} (Reference)"].append(wsd_value)

        # Handle Other Replicate Groups
        for i, (replicate_files, title) in enumerate(replicate_groups):
            if title == reference_title:
                continue  # Skip the reference group

            # Calculate WSD values for each spectrum relative to the reference group's average spectrum
            wsd_values_by_group[title] = []
            for sample_file, blank_file in replicate_files:
                normalized_df = process_cd_data(sample_file, blank_file)
                wsd_value = calculate_wsd(normalized_df, avg_normalized_spectrum, reference_spectra_std)
                wsd_values_by_group[title].append(wsd_value)                      

        return wsd_values_by_group
        
    def perform_and_plot_t_tests(wsd_values_by_group, reference_title, ax):
        """
        Perform t-tests comparing each group's WSD values to the reference group's WSD values,
        and plot the results on the provided axis.

        Parameters:
        - wsd_values_by_group: Dictionary with group titles as keys and lists of WSD values as values.
        - reference_title: Title of the reference group.
        - ax: The axis on which to plot the results.
        """
        reference_wsd = wsd_values_by_group.get(reference_title, [])
        if not reference_wsd:
            return

        reference_wsd_mean = np.mean(reference_wsd)
        reference_wsd_std = np.std(reference_wsd)

        # Plot vertical lines for reference WSD
        ax.axvline(reference_wsd_mean, color='red', linestyle='--', label='Reference Mean')
        ax.axvline(reference_wsd_mean + 2 * reference_wsd_std, color='black', linestyle=':', label='+2 Std Dev')
        ax.axvline(reference_wsd_mean - 2 * reference_wsd_std, color='black', linestyle=':', label='-2 Std Dev')

        # Perform t-tests and plot means with whiskers
        for group_title, wsd_values in wsd_values_by_group.items():
            if group_title == reference_title:
                continue  # Skip the reference group itself

            # Calculate mean and std for the current group
            mean_wsd = np.mean(wsd_values)
            std_wsd = np.std(wsd_values)            

            # Calculate y-position for this group in the scatter plot
            y_position = list(wsd_values_by_group.keys()).index(group_title)

            # Plot mean and error bars (whiskers) for the group
            ax.errorbar(mean_wsd, y_position, 
                        xerr=2 * std_wsd, 
                        fmt='o',  # Circle marker
                        color='blue', 
                        label=group_title + ' Mean ± 2 Std Dev' if y_position == 0 else "",  # Only label first instance
                        capsize=5)

            # Perform t-tests
            t_stat, p_value = ttest_ind(reference_wsd, wsd_values, equal_var=False)
            

            # Adjusted x coordinate to shift the text to the right of the plot
            x_offset = ax.get_xlim()[1] * 0.1  # 10% of the x-axis range as offset
            
            # Plot t-test results
            ax.text(ax.get_xlim()[1] + x_offset, y_position, 
                    f"Group: {group_title}\nT-stat: {t_stat:.2f}\nP-value: {p_value:.6f}",
                    ha='left', va='center', fontsize=8, color='black')

    def calculate_residuals_gfactor(df, reference_df):
        """
        Calculate residuals between the g factor of the current spectrum and the average g factor spectrum.

        Parameters:
        - df: DataFrame containing the g factor data of the current spectrum.
        - reference_df: DataFrame containing the average g factor data from the reference spectra.

        Returns:
        - df: DataFrame with an additional column 'Residual' showing the difference between the current spectrum and the average spectrum.
        """
        # Merge the current spectrum with the average spectrum on 'Wavelength'
        df_merged = pd.merge(df[['Wavelength', 'G factor']], reference_df[['Wavelength', 'G factor']], on='Wavelength', suffixes=('', '_avg'))

        # Calculate the residuals
        df_merged['Residual'] = df_merged['G factor'] - df_merged['G factor_avg']

        return df_merged
        

                                    
    def plot_overlays(output_dir, pdf_pages, replicate_groups, reference_index):
        
        # Handle Reference Replicate Group First
        reference_group_files, reference_title = replicate_groups[reference_index]
        reference_normalized_spectra = []
        reference_blank_spectra = []
        
        # Define color map and styles
        colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(replicate_groups)))  # Shades of blue
        color_map = {f"{reference_title} (Reference)": 'black'}  # Initialize color map with reference color
        linestyle_map = {f"{reference_title} (Reference)": 'dashdot', 'Normal': '-'}

        fig, axes = plt.subplots(3, 3, figsize=(46, 18))  # Updated to 3x3 for the additional plot

        # Initialize dictionaries to store average G spectra, WSD values, and average residuals
        avg_spectra_by_group = {}
        wsd_values_by_group = {}
        avg_residuals_by_group = {}
        
        # Initialize a dictionary to track if the label was already added for each group
        label_added1 = {}; label_added2 = {}; label_added3 = {}; label_added4 = {}; label_added5 = {}; label_added6 = {}

        # Calculate WSD values
        wsd_values_by_group = calculate_wsd_values(replicate_groups, reference_index)

        

        for sample_file, blank_file in reference_group_files:
            normalized_df = process_cd_data(sample_file, blank_file)
            reference_normalized_spectra.append(normalized_df[['Wavelength', 'G factor', 'Absorbance', 'CD', 'Voltage']])
            
            # Store the blank CD data
            blank_df, _ = read_csv_until_string(blank_file)
            reference_blank_spectra.append(blank_df[['Wavelength', 'CD']])

        if reference_normalized_spectra:
            reference_normalized_df_list = pd.concat(reference_normalized_spectra)
            reference_normalized_df_list = reference_normalized_df_list.sort_values('Wavelength').reset_index(drop=True)
            avg_normalized_spectrum = reference_normalized_df_list.groupby('Wavelength').agg({'G factor': 'mean'}).reset_index()
            std_normalized_spectrum = reference_normalized_df_list.groupby('Wavelength').agg({'G factor': 'std'}).reset_index()

            # Plot individual CD spectra for the reference replicates on [0, 0]
            for df in reference_normalized_spectra:
                axes[0, 0].plot(df['Wavelength'], df['CD'], color='gray', linestyle=':', alpha=0.7, linewidth=0.2, label=f"{reference_title} (Reference)" if reference_title not in label_added1 else "")
                label_added1[reference_title] = True  # Mark label as added
                
            # Plot individual G spectra for the reference replicates on [1, 0]
            for df in reference_normalized_spectra:
                axes[1, 0].plot(df['Wavelength'], df['G factor'], color='gray', linestyle=':', alpha=0.7, linewidth=0.2, label=f"{reference_title} (Reference)" if reference_title not in label_added2 else "")
                label_added2[reference_title] = True  # Mark label as added
                
            # Plot Absorbance for the reference replicates on [1, 0]
            for df in reference_normalized_spectra:
                axes[0, 2].plot(df['Wavelength'], df['Absorbance'], color='gray', linestyle=':', alpha=0.7, linewidth=0.2, label=f"{reference_title} (Reference)" if reference_title not in label_added3 else "")
                label_added3[reference_title] = True  # Mark label as added
                
            # Plot reference blank CD on [2, 2]
            for blank_df in reference_blank_spectra:
                axes[2, 2].plot(blank_df['Wavelength'], blank_df['CD'], color='black', linestyle='--', linewidth=0.2, alpha=0.7, label=f"{reference_title} (Reference)" if reference_title not in label_added4 else "")
                label_added4[reference_title] = True  # Mark label as added
                
            # Plot individual Voltage spectra for the reference replicates on [0, 1]
            for df in reference_normalized_spectra:
                axes[0, 1].plot(df['Wavelength'], df['Voltage'], color='gray', linestyle=':', alpha=0.7, linewidth=0.2, label=f"{reference_title} (Reference)" if reference_title not in label_added5 else "")
                label_added5[reference_title] = True  # Mark label as added

            # Plot average G spectrum for reference group with shaded region for ±2 std dev on [2, 0]
            axes[2, 0].plot(avg_normalized_spectrum['Wavelength'], avg_normalized_spectrum['G factor'], color='black', linestyle='--', label=f"{reference_title} (Reference)")
            axes[2, 0].fill_between(
                avg_normalized_spectrum['Wavelength'],
                avg_normalized_spectrum['G factor'] - 3 * std_normalized_spectrum['G factor'],
                avg_normalized_spectrum['G factor'] + 3 * std_normalized_spectrum['G factor'],
                color='red', alpha=0.2, label='±3 Std Dev')
                
            # Plot + or - 2 Standard Deviations around Residuals plot
                
            axes[2, 1].fill_between(
                avg_normalized_spectrum['Wavelength'],
                -3 * std_normalized_spectrum['G factor'],  # Lower bound for shading
                3 * std_normalized_spectrum['G factor'],   # Upper bound for shading
                color='red', alpha=0.2, label='±3 Std Dev')
                
          

        # Handle Other Replicate Groups
        for i, (replicate_files, title) in enumerate(replicate_groups):
            if title == reference_title:              
                continue  # Skip the reference group

            color = colors[i]  # Use color map to get a color
            linestyle = linestyle_map.get(title, '-')
            color_map[title] = color
            spectra_list = []
            blank_spectra_list = []

            for sample_file, blank_file in replicate_files:
                normalized_df = process_cd_data(sample_file, blank_file)
                normalized_df['Title'] = title
                
                # Store blank CD data
                blank_df, _ = read_csv_until_string(blank_file)
                blank_spectra_list.append(blank_df[['Wavelength', 'CD']])

                # Plot CD data
                axes[0, 0].plot(normalized_df['Wavelength'], normalized_df['CD'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added1 else "")
                label_added1[title] = True

                # Plot Voltage 
                axes[0, 1].plot(normalized_df['Wavelength'], normalized_df['Voltage'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added2 else "")
                label_added2[title] = True

                # Plot G spectra
                axes[1, 0].plot(normalized_df['Wavelength'], normalized_df['G factor'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added3 else "")
                label_added3[title] = True     

                # Calculate and plot G factor Residuals                 
                residuals_df = calculate_residuals_gfactor(normalized_df, avg_normalized_spectrum)
                axes[1, 1].plot(residuals_df['Wavelength'], residuals_df['Residual'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added4 else "")
                label_added4[title] = True            

                # Plot absorbance
                axes[0, 2].plot(normalized_df['Wavelength'], normalized_df['Absorbance'], color=color, linestyle=linestyle, linewidth=0.2, label=title if title not in label_added5 else "")
                label_added5[title] = True

                # Append to spectra list
                spectra_list.append(normalized_df[['Wavelength', 'G factor']])
            
            # Plot blank CD for this group on [2, 2]
            for blank_df in blank_spectra_list:
                axes[2, 2].plot(blank_df['Wavelength'], blank_df['CD'], color=color, linestyle=linestyle, linewidth=0.5, alpha=0.7, label=title if title not in label_added5 else "")
                label_added6[title] = True

            # Calculate and plot the average G spectra for each replicate group
            if spectra_list:
                concatenated_df = pd.concat(spectra_list)
                concatenated_df = concatenated_df.sort_values('Wavelength').reset_index(drop=True)
                avg_spectra = concatenated_df.groupby('Wavelength').agg({'G factor': 'mean'}).reset_index()
                avg_spectra_by_group[title] = avg_spectra
                axes[2, 0].plot(avg_spectra['Wavelength'], avg_spectra['G factor'], color=color, linestyle=linestyle, linewidth=0.2, label=title)

                # Calculate average residuals for each replicate group relative to the reference group
                merged_df = avg_spectra.merge(avg_normalized_spectrum, on='Wavelength', suffixes=('', '_ref'))
                merged_df['Residual'] = merged_df['G factor'] - merged_df['G factor_ref']
                avg_residuals_by_group[title] = merged_df[['Wavelength', 'Residual']]

        # Plot average residuals
        for i, (group_title, residuals_df) in enumerate(avg_residuals_by_group.items()):
            y = len(avg_residuals_by_group) - i - 1
            axes[2, 1].plot(residuals_df['Wavelength'], residuals_df['Residual'], label=group_title, linewidth=0.2, color=color_map.get(group_title, 'blue'))
      
        # Plot WSD values as a scatter plot with WSD on x-axis and groups on y-axis
        wsd_df = pd.DataFrame([(group, wsd) for group, wsd_values in wsd_values_by_group.items() for wsd in wsd_values], columns=['Group', 'WSD'])
        sns.scatterplot(x='WSD', y='Group', data=wsd_df, ax=axes[1, 2], hue='Group', palette=color_map, s=100)
        axes[1, 2].set_title('WSD Values')
        axes[1, 2].set_xlabel('WSD Value')
        axes[1, 2].set_ylabel('Group')
             
        # Perform and plot t-tests
        perform_and_plot_t_tests(wsd_values_by_group, f"{reference_title} (Reference)", axes[1, 2])

        # Set titles and axis labels for each subplot
        axes[0, 0].set_title('CD Data')
        axes[0, 0].set_xlabel('Wavelength')
        axes[0, 0].set_ylabel('CD (degrees)')

        axes[0, 1].set_title('HT Voltage')
        axes[0, 1].set_xlabel('Wavelength')
        axes[0, 1].set_ylabel('Voltage')

        axes[0, 2].set_title('Absorbance')
        axes[0, 2].set_xlabel('Wavelength')
        axes[0, 2].set_ylabel('Absorbance')

        axes[1, 0].set_title('G Spectra')
        axes[1, 0].set_xlabel('Wavelength')
        axes[1, 0].set_ylabel('G Factor')

        axes[1, 1].set_title('G Spectra Residuals')
        axes[1, 1].set_xlabel('Wavelength')
        axes[1, 1].set_ylabel('Residual')

        axes[2, 0].set_title('Average G Spectrum')
        axes[2, 0].set_xlabel('Wavelength')
        axes[2, 0].set_ylabel('G Factor')

        axes[2, 1].set_title('Average Residuals')
        axes[2, 1].set_xlabel('Wavelength')
        axes[2, 1].set_ylabel('Residual')

        axes[2, 2].set_title('Blank CD Spectra')
        axes[2, 2].set_xlabel('Wavelength')
        axes[2, 2].set_ylabel('CD (millidegrees)')
        
        # Adjust line width for each plot and remove grid
        axes[0, 0].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for CD Data
        axes[0, 1].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Voltage
        axes[0, 2].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Absorbance
        axes[1, 0].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for G Spectra
        axes[1, 1].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for G Factor Residuals
        axes[2, 0].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Avg G Spectrum
        axes[2, 1].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Average Residuals
        axes[2, 2].axhline(0, color='black', linestyle='-')  # Solid black line at y=0 for Blank CD

        # Adjust line width for each plot and move legend outside the plot area
        for ax in axes.flat:
            # Move legend outside the plot to the right
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        # Adjust the layout to prevent overlapping
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust right margin for space for legend


        # Save the figure to PDF      
        pdf_pages.savefig(fig)
        plt.close(fig)
        
    # Function to add replicate group
    def add_replicate_group():
        replicate_title = replicate_title_entry.get().strip()
        if not replicate_title:
            messagebox.showerror("Input Error", "Please enter a title for the replicate series.")
            return

        replicate_files = []
        while True:
            sample_file = filedialog.askopenfilename(title="Select Sample CSV File", filetypes=[("CSV files", "*.csv")])
            if not sample_file:
                break
            blank_file = filedialog.askopenfilename(title="Select Blank CSV File", filetypes=[("CSV files", "*.csv")])
            if not blank_file:
                break
            replicate_files.append((sample_file, blank_file))
        
        if not replicate_files:
            messagebox.showerror("Input Error", "Please add at least one sample and blank pair.")
            return
        
        replicate_groups.append((replicate_files, replicate_title))

        samples_listbox.delete(0, tk.END)
        for replicate_files, title in replicate_groups:
            samples_listbox.insert(tk.END, f"Replicate Series: {title} ({len(replicate_files)} pairs)")

        replicate_title_entry.delete(0, tk.END)

    # Function to generate PDF
    def generate_pdf():
        if not replicate_groups:
            messagebox.showerror("Input Error", "Please add at least one replicate group.")
            return
        output_filename = output_filename_entry.get().strip()
        if not output_filename:
            messagebox.showerror("Input Error", "Please enter an output filename.")
            return
        output_path = os.path.join(output_dir, f"{output_filename}.pdf")

        with PdfPages(output_path) as pdf_pages:
            reference_index = reference_index_entry.get().strip()
            if not reference_index.isdigit() or int(reference_index) < 0 or int(reference_index) >= len(replicate_groups):
                messagebox.showerror("Input Error", "Please enter a valid reference index.")
                return
            reference_index = int(reference_index)
            plot_overlays(output_dir, pdf_pages, replicate_groups, reference_index)

        messagebox.showinfo("Success", f"Plots and tables have been saved to {output_path}")

    # GUI setup
    root = tk.Tk()
    root.title("CD Single Temperature Analysis")
    root.geometry("800x600")

    frame = tk.Frame(root, bg="lightgray")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Label and Entry for replicate series title
    replicate_title_label = tk.Label(frame, text="Replicate Series Title:", bg="lightgray")
    replicate_title_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    replicate_title_entry = tk.Entry(frame, width=50)
    replicate_title_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to add replicate group
    add_replicate_group_button = tk.Button(frame, text="Add Replicate Group", command=add_replicate_group)
    add_replicate_group_button.grid(row=0, column=2, padx=10, pady=10)

    # Listbox to display added replicate groups
    samples_listbox = tk.Listbox(frame, width=100, height=10)
    samples_listbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for output PDF filename
    output_filename_label = tk.Label(frame, text="Output PDF Filename:", bg="lightgray")
    output_filename_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    output_filename_entry = tk.Entry(frame, width=50)
    output_filename_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.EW)

    # Label and Entry for reference index
    reference_index_label = tk.Label(frame, text="Reference Index:", bg="lightgray")
    reference_index_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
    reference_index_entry = tk.Entry(frame, width=50)
    reference_index_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

    # Button to generate PDF
    generate_pdf_button = tk.Button(frame, text="Generate PDF", command=generate_pdf)
    generate_pdf_button.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    # Configure column 1 to expand
    frame.grid_columnconfigure(1, weight=1)

    # Initialize lists for replicate groups
    replicate_groups = []

    # Set output directory
    output_dir = r'Y:\CD\Analysis-Development\Test_Outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Run the GUI loop
    root.mainloop()
    
# Function to handle CD Melt
def cd_melt():
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


# Create GUI for selecting analysis type
def create_gui():
    root = tk.Tk()
    root.title("Unified Analysis Program")
    root.geometry("400x300")  # Adjust as needed

    def on_cd_single():
        root.withdraw()
        cd_single()
    
    def on_cd_single_mw():
        root.withdraw()
        cd_single_mw()
    
    def on_cd_single_mw_sequence():
        root.withdraw()
        cd_single_mw_sequence()

    def on_cd_single_wsd_auc():
        root.withdraw()
        cd_single_wsd_auc()
        
    def on_cd_single_wsd_gfactor():
        root.withdraw()
        cd_single_wsd_gfactor()
        
    def on_cd_melt():
        root.withdraw()
        cd_melt()
    
   
    # Create buttons for each analysis type
    tk.Button(root, text="Wavelengh Scan Millidegrees", command=on_cd_single).pack(pady=10)
    tk.Button(root, text="Wavelengh Scan MRE", command=on_cd_single_mw).pack(pady=10)
    tk.Button(root, text="Wavelengh Scan MRE (Sequence Input)", command=on_cd_single_mw_sequence).pack(pady=10)
    tk.Button(root, text="Wavelengh Scan WSD comparison (AUC)", command=on_cd_single_wsd_auc).pack(pady=10)
    tk.Button(root, text="Wavelengh Scan WSD comparison (G Factor)", command=on_cd_single_wsd_gfactor).pack(pady=10)
    tk.Button(root, text="Thermal Melt", command=on_cd_melt).pack(pady=10)
    
    root.mainloop()

# Entry point for the unified program
if __name__ == "__main__":
    create_gui()
