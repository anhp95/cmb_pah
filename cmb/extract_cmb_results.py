import pandas as pd
import re
import os
from pathlib import Path

def extract_cmb_results(file_path):
    """
    Extract all CMB result sections from output file
    
    Parameters:
    file_path (str): Path to the CMB result text file
    
    Returns:
    tuple: (fitting_stats_df, source_contrib_with_singular_df, species_concentrations_df, source_name_table_df)
    """
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Extract fitting statistics
    fitting_stats = {}
    
    # R SQUARE
    r_square_match = re.search(r'R SQUARE\s+([\d.]+)', content)
    if r_square_match:
        fitting_stats['R_SQUARE'] = float(r_square_match.group(1))
    
    # % MASS
    mass_match = re.search(r'% MASS\s+([\d.]+)', content)
    if mass_match:
        fitting_stats['PERCENT_MASS'] = float(mass_match.group(1))
    
    # CHI SQUARE
    chi_square_match = re.search(r'CHI SQUARE\s+([\d.]+)', content)
    if chi_square_match:
        fitting_stats['CHI_SQUARE'] = float(chi_square_match.group(1))
    
    # DEGREES FREEDOM
    df_match = re.search(r'DEGREES FREEDOM\s+(\d+)', content)
    if df_match:
        fitting_stats['DEGREES_FREEDOM'] = int(df_match.group(1))
    
    # FIT MEASURE
    fit_measure_match = re.search(r'FIT MEASURE\s+([\d.]+)', content)
    if fit_measure_match:
        fitting_stats['FIT_MEASURE'] = float(fit_measure_match.group(1))
    
    fitting_stats_df = pd.DataFrame([fitting_stats])
    
    # 2. Extract source contribution estimates
    source_contrib_pattern = r'SOURCE CONTRIBUTION ESTIMATES:(.*?)----------------------------------------------------'
    source_match = re.search(source_contrib_pattern, content, re.DOTALL)
    
    source_contrib_data = []
    if source_match:
        source_section = source_match.group(1)
        # Find lines with source data (YES/NO followed by source code)
        source_lines = re.findall(r'(YES|NO)\s+(\w+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', source_section)
        
        for line in source_lines:
            source_contrib_data.append({
                'STATUS': line[0],
                'EST_CODE': line[1], 
                'NAME': line[2],
                'SCE_ng_m3': float(line[3]),
                'Std_Err': float(line[4]),
                'Tstat': float(line[5])
            })
    
    # Extract singular values
    singular_pattern = r'1 / Singular Value\s*-+\s*([\d.\s]+)\s*-+'
    singular_match = re.search(singular_pattern, content, re.DOTALL)
    
    singular_values = []
    if singular_match:
        values_text = singular_match.group(1).strip()
        # Split by whitespace and convert to float
        values = [float(x) for x in values_text.split() if x.strip()]
        singular_values = values
    
    # Combine source contributions with singular values
    source_contrib_with_singular_data = []
    for i, source in enumerate(source_contrib_data):
        combined_row = source.copy()
        if i < len(singular_values):
            combined_row['Singular_Value_Inverse'] = singular_values[i]
        else:
            combined_row['Singular_Value_Inverse'] = None
        source_contrib_with_singular_data.append(combined_row)
    
    source_contrib_with_singular_df = pd.DataFrame(source_contrib_with_singular_data)
    
    # 3. Extract species concentrations
    species_pattern = r'SPECIES CONCENTRATIONS:.*?SPECIES\s+FIT\s+MEASURED\s+CALCULATED\s+CALCULATED\s+RESIDUAL.*?MEASURED\s+UNCERTAINTY\s*-+(.*?)-+'
    species_match = re.search(species_pattern, content, re.DOTALL)
    
    species_data = []
    if species_match:
        species_section = species_match.group(1)
        # Parse each species line
        species_lines = species_section.strip().split('\n')
        for line in species_lines:
            line = line.strip()
            if line and not line.startswith('-'):
                # Parse complex format: TMAC   TMAU       1.00670+- 0.27960   0.65894+- 0.06655   0.65+- 0.19    -1.2
                parts = line.split()
                if len(parts) >= 6:
                    species_code = parts[0]
                    species_uncertainty = parts[1]
                    
                    # Extract fit value (remove +- and uncertainty)
                    fit_str = parts[2] if len(parts) > 2 else ''
                    fit_match = re.search(r'([\d.]+)', fit_str)
                    fit_value = float(fit_match.group(1)) if fit_match else None
                    
                    # Extract calculated value
                    calc_str = parts[3] if len(parts) > 3 else ''
                    calc_match = re.search(r'([\d.]+)', calc_str)
                    calc_value = float(calc_match.group(1)) if calc_match else None
                    
                    # Extract calculated/measured ratio
                    ratio_str = parts[4] if len(parts) > 4 else ''
                    ratio_match = re.search(r'([\d.]+)', ratio_str)
                    ratio_value = float(ratio_match.group(1)) if ratio_match else None
                    
                    # Extract residual
                    residual_str = parts[5] if len(parts) > 5 else ''
                    residual_match = re.search(r'([-\d.]+)', residual_str)
                    residual_value = float(residual_match.group(1)) if residual_match else None
                    
                    species_data.append({
                        'SPECIES_CODE': species_code,
                        'SPECIES_UNCERTAINTY': species_uncertainty,
                        'FIT': fit_value,
                        'CALCULATED': calc_value,
                        'CALC_MEAS_RATIO': ratio_value,
                        'RESIDUAL': residual_value
                    })
    
    species_concentrations_df = pd.DataFrame(species_data)
    
    # 4. Extract source name table (final table)
    source_name_pattern = r'SOURCE NAME\s+(.*?)$'
    source_name_match = re.search(source_name_pattern, content, re.DOTALL | re.MULTILINE)
    
    source_name_data = []
    if source_name_match:
        source_name_section = source_name_match.group(1)
        lines = source_name_section.strip().split('\n')
        
        # Get source names from header (skip first line which might be SPECIES  CALCULATED  MEASURED)
        source_names = []
        header_line = None
        for line in lines:
            if 'SPECIES' in line and 'CALCULATED' in line and 'MEASURED' in line:
                header_line = line
                # Extract source names after MEASURED
                parts = line.split()
                measured_idx = next((i for i, part in enumerate(parts) if 'MEASURED' in part), -1)
                if measured_idx >= 0:
                    source_names = parts[measured_idx + 1:]
                break
        
        # Parse data lines
        for line in lines:
            line = line.strip()
            if line and not line.startswith('SPECIES') and line != header_line:
                parts = line.split()
                if len(parts) >= 3:  # At least SPECIES, CALCULATED, MEASURED
                    species = parts[0]
                    calculated = float(parts[1]) if parts[1].replace('.', '').replace('-', '').isdigit() else None
                    measured = float(parts[2]) if parts[2].replace('.', '').replace('-', '').isdigit() else None
                    
                    row_data = {
                        'SPECIES': species,
                        'CALCULATED': calculated, 
                        'MEASURED': measured
                    }
                    
                    # Add source values
                    for i, source_name in enumerate(source_names):
                        if i + 3 < len(parts):  # +3 to account for SPECIES, CALCULATED, MEASURED
                            value_str = parts[i + 3]
                            try:
                                row_data[source_name] = float(value_str)
                            except:
                                row_data[source_name] = None
                        else:
                            row_data[source_name] = None
                    
                    source_name_data.append(row_data)
    
    source_name_table_df = pd.DataFrame(source_name_data)
    
    return fitting_stats_df, source_contrib_with_singular_df, species_concentrations_df, source_name_table_df

def process_cmb_directory(directory_path):
    """
    Process all CMB result files in a directory
    
    Parameters:
    directory_path (str): Path to directory containing CMB result files
    
    Returns:
    tuple: (combined_fitting_df, combined_source_contrib_df, combined_species_df, combined_source_name_df)
    """
    
    all_fitting = []
    all_source_contrib = []
    all_species = []
    all_source_name = []
    
    directory = Path(directory_path)
    
    # Process all .txt files in directory
    for file_path in directory.glob('*.txt'):
        try:
            fitting_df, source_contrib_df, species_df, source_name_df = extract_cmb_results(str(file_path))
            
            # Add file identifier
            file_id = file_path.stem
            fitting_df['File_ID'] = file_id
            source_contrib_df['File_ID'] = file_id
            species_df['File_ID'] = file_id
            source_name_df['File_ID'] = file_id
            
            all_fitting.append(fitting_df)
            all_source_contrib.append(source_contrib_df)
            all_species.append(species_df)
            all_source_name.append(source_name_df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Combine all results
    combined_fitting_df = pd.concat(all_fitting, ignore_index=True) if all_fitting else pd.DataFrame()
    combined_source_contrib_df = pd.concat(all_source_contrib, ignore_index=True) if all_source_contrib else pd.DataFrame()
    combined_species_df = pd.concat(all_species, ignore_index=True) if all_species else pd.DataFrame()
    combined_source_name_df = pd.concat(all_source_name, ignore_index=True) if all_source_name else pd.DataFrame()
    
    return combined_fitting_df, combined_source_contrib_df, combined_species_df, combined_source_name_df

# Example usage
if __name__ == "__main__":
    # Process single file
    file_path = "d:/project/phuong_tnmt/cmb/cmb_result/pah/india_sp/HPBG.txt"
    
    fitting_stats, source_contrib_singular, species_concentrations, source_name_table = extract_cmb_results(file_path)
    
    print("1. FITTING STATISTICS:")
    print(fitting_stats)
    print("\n2. SOURCE CONTRIBUTION ESTIMATES + SINGULAR VALUES:")
    print(source_contrib_singular)
    print("\n3. SPECIES CONCENTRATIONS:")
    print(species_concentrations)
    print("\n4. SOURCE NAME TABLE:")
    print(source_name_table)
    
    # Process entire directory
    directory_path = "d:/project/phuong_tnmt/cmb/cmb_result/pah/india_sp/"
    
    all_fitting, all_source_contrib, all_species, all_source_name = process_cmb_directory(directory_path)
    
    print("\nCOMBINED RESULTS:")
    print(f"Total files processed: {len(all_fitting)}")
    print("\n1. Combined Fitting Statistics:")
    print(all_fitting.head())
    print("\n2. Combined Source Contributions + Singular Values:")
    print(all_source_contrib.head())
    print("\n3. Combined Species Concentrations:")
    print(all_species.head())
    print("\n4. Combined Source Name Tables:")
    print(all_source_name.head())
    
    # Save to CSV files
    all_fitting.to_csv("cmb_fitting_statistics.csv", index=False)
    all_source_contrib.to_csv("cmb_source_contributions_with_singular.csv", index=False)
    all_species.to_csv("cmb_species_concentrations.csv", index=False)
    all_source_name.to_csv("cmb_source_name_tables.csv", index=False)
    
    print("\nAll 4 tables saved to CSV files!")
    print("- cmb_fitting_statistics.csv")
    print("- cmb_source_contributions_with_singular.csv")
    print("- cmb_species_concentrations.csv") 
    print("- cmb_source_name_tables.csv")