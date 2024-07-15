import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

# Define rows and columns to pull
skip_rows = list(range(20))
use_cols = [0, 1, 2]

# Function to read CSV file and extract numeric and additional data
def read_csv_until_string(file_path, skip_rows=20, use_cols=[0, 1, 2]):
    numeric_data = {'Wavelength': [], 'CD': [], 'Voltage': []}
    additional_data = []
    read_numeric_data = True

    # Row names to keep in the additional data
    row_names_to_keep = [
        "Sample name", "Creation date", "Start", "End", "Data interval",
        "Instrument name", "Model name", "Serial No.", "Temperature",
        "Measurement date", "Data pitch", "CD scale", "FL scale", "D.I.T.",
        "Bandwidth", "Start mode", "Scanning speed", "Accumulations"
    ]
    
    try:
        with open(file_path, 'r') as file:
            for _ in range(skip_rows):
                next(file)
            for line in file:
                cols = line.strip().split(',')
                if read_numeric_data:
                    try:
                        # Extract numeric data
                        wavelength = float(cols[use_cols[0]].strip())
                        cd = float(cols[use_cols[1]].strip())
                        voltage = float(cols[use_cols[2]].strip())
                        numeric_data['Wavelength'].append(wavelength)
                        numeric_data['CD'].append(cd)
                        numeric_data['Voltage'].append(voltage)
                    except (ValueError, IndexError):
                        # Switch to reading additional data if numeric data is not found
                        read_numeric_data = False
                        if cols[0] in row_names_to_keep:
                            additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
                else:
                    if cols[0] in row_names_to_keep:
                        additional_data.append([cols[0], cols[1] if len(cols) > 1 else ''])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    # Convert lists to DataFrames
    df_numeric = pd.DataFrame(numeric_data)
    df_additional = pd.DataFrame(additional_data, columns=['Property', 'Value'])
    return df_numeric, df_additional

# Function to plot CD vs Wavelength
def plot_cd_vs_wavelength(output_dir, pdf_pages, graph_titles, output_filename_label, *csv_files):
    num_pairs = len(csv_files) // 5

    if num_pairs == 1:
        fig, axes = plt.subplots(2, figsize=(10, 12))
    else:
        fig, axes = plt.subplots(2, num_pairs + 1, figsize=(20, 12))

    for i in range(num_pairs):
        csv1, csv2, molar_conc, no_residues, path_length = csv_files[i * 5], csv_files[i * 5 + 1], float(csv_files[i * 5 + 2]), float(csv_files[i * 5 + 3]), float(csv_files[i * 5 + 4])
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
        conversion_factor = 1 / (path_length * molar_conc * no_residues)
        df_CD_blank_subtracted1['CD'] = df_CD_blank_subtracted1['CD'] * conversion_factor
        
        # Calculate mean residue ellipticity
        df_CD_blank_subtracted1['MRE'] = df_CD_blank_subtracted1['CD'] / 3298

        # Filter out CD data associated with voltage exceeding 700
        voltage_threshold = 700
        mask = df_CD_blank_subtracted1['Voltage'] <= voltage_threshold
        df_CD_blank_subtracted_filtered1 = df_CD_blank_subtracted1[mask]

        # Get title for the graph
        title = graph_titles[i] if i < len(graph_titles) else f'Sample {i+1}'
        
        # Create vertically stacked subplots
        if num_pairs == 1:
            ax0, ax1 = axes
        else:
            ax0, ax1 = axes[0][i], axes[1][i]

        # Plot CD vs Wavelength
        ax0.plot(df_CD_blank_subtracted_filtered1['Wavelength'], df_CD_blank_subtracted_filtered1['CD'], color='blue', marker='o', markersize=0, linestyle='-')
        ax0.set_title(f'CD vs. Wavelength ({title})')
        ax0.set_xlabel('Wavelength')
        ax0.set_ylabel('Molar Ellipticity')
        ax0.grid(True)
        ax0.axhline(y=0, color='black', linestyle='--')

        # Plot HT vs Wavelength
        ax1.plot(df_CD_blank_subtracted_filtered1['Wavelength'], df_CD_blank_subtracted_filtered1['Voltage'], color='blue', marker='o', markersize=0, linestyle='-')
        ax1.set_title(f'HT vs. Wavelength ({title})')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('HT (Voltage)')
        ax1.grid(True)
        ax1.axhline(y=700, color='black', linestyle='--')

        # Export processed data to CSV with the same filename as the PDF
        export_file_path = os.path.join(output_dir, f"{output_filename_label}_processed.csv")
        df_CD_blank_subtracted_filtered1.to_csv(export_file_path, columns=['Wavelength', 'Voltage', 'CD', 'MRE'], index=False)

    if num_pairs > 1:
        for i in range(num_pairs):
            csv1, csv2, molar_conc, no_residues, path_length = csv_files[i * 5], csv_files[i * 5 + 1], float(csv_files[i * 5 + 2]), float(csv_files[i * 5 + 3]), float(csv_files[i * 5 + 4])
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
            conversion_factor = 1 / (path_length * molar_conc * no_residues)
            df_CD_blank_subtracted1['CD'] = df_CD_blank_subtracted1['CD'] * conversion_factor

            # Calculate mean residue ellipticity
            df_CD_blank_subtracted1['MRE'] = df_CD_blank_subtracted1['CD'] / 3298

            # Filter out CD data associated with voltage exceeding 700
            voltage_threshold = 700
            mask = df_CD_blank_subtracted1['Voltage'] <= voltage_threshold
            df_CD_blank_subtracted_filtered1 = df_CD_blank_subtracted1[mask]

            # Get title for the graph
            title = graph_titles[i] if i < len(graph_titles) else f'Sample {i+1}'

            # Plot overlay of CD vs Wavelength
            axes[0][num_pairs].plot(df_CD_blank_subtracted_filtered1['Wavelength'], df_CD_blank_subtracted_filtered1['CD'], label=title)

        axes[0][num_pairs].set_title('Overlay of CD vs. Wavelength')
        axes[0][num_pairs].set_xlabel('Wavelength')
        axes[0][num_pairs].set_ylabel('Molar Ellipticity')
        axes[0][num_pairs].grid(True)
        axes[0][num_pairs].axhline(y=0, color='black', linestyle='--')
        axes[0][num_pairs].legend()

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.2)

    # Save the figure to the PDF file
    pdf_pages.savefig(fig)
    plt.close(fig)

    for i in range(num_pairs):
        _, df_additional1 = read_csv_until_string(csv_files[i * 5])

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        table_data = df_additional1
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

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
        molar_conc = float(molar_conc_entry.get().strip())
        no_residues = float(no_residues_entry.get().strip())
        path_length = float(path_length_entry.get().strip())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for Molar Concentration, Number of Residues, and Path Length.")
        return

    sample_blank_pairs.append((sample_file, blank_file, molar_conc, no_residues, path_length))
    titles.append(title)

    samples_listbox.insert(tk.END, f"Sample: {os.path.basename(sample_file)} | Blank: {os.path.basename(blank_file)} | Title: {title}")
    title_entry.delete(0, tk.END)
    molar_conc_entry.delete(0, tk.END)
    no_residues_entry.delete(0, tk.END)
    path_length_entry.delete(0, tk.END)

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

# Label and Entry for molar concentration
molar_conc_label = tk.Label(frame, text="Molar Concentration:", bg="lightgray")
molar_conc_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
molar_conc_entry = tk.Entry(frame, width=50)
molar_conc_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.EW)

# Label and Entry for number of residues
no_residues_label = tk.Label(frame, text="Number of Residues:", bg="lightgray")
no_residues_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
no_residues_entry = tk.Entry(frame, width=50)
no_residues_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.EW)

# Label and Entry for path length
path_length_label = tk.Label(frame, text="Path Length (mm):", bg="lightgray")
path_length_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
path_length_entry = tk.Entry(frame, width=50)
path_length_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

# Button to add sample and blank pair
add_pair_button = tk.Button(frame, text="Add Sample and Blank Pair", command=add_sample_blank_pair)
add_pair_button.grid(row=4, column=2, padx=10, pady=10)

# Listbox to display added sample and blank pairs
samples_listbox = tk.Listbox(frame, width=100, height=10)
samples_listbox.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky=tk.EW)

# Label and Entry for output PDF filename
output_filename_label = tk.Label(frame, text="Output PDF Filename:", bg="lightgray")
output_filename_label.grid(row=6, column=0, padx=10, pady=10, sticky=tk.W)
output_filename_entry = tk.Entry(frame, width=50)
output_filename_entry.grid(row=6, column=1, padx=10, pady=10, sticky=tk.EW)

# Button to generate PDF
generate_pdf_button = tk.Button(frame, text="Generate PDF", command=generate_pdf)
generate_pdf_button.grid(row=6, column=2, padx=10, pady=10)

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
