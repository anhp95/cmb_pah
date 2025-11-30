# %%
import pandas as pd
import re


def extract_cmb_from_file(file_path):
    """Extract all sections from CMB output file into 4 DataFrames"""

    with open(file_path, "r") as f:
        text = f.read()

    # 1. FITTING STATISTICS
    fitting_stats = {}
    r_square = re.search(r"R SQUARE\s+([\d.]+)", text)
    percent_mass = re.search(r"% MASS\s+([\d.]+)", text)
    chi_square = re.search(r"CHI SQUARE\s+([\d.]+)", text)
    degrees_freedom = re.search(r"DEGREES FREEDOM\s+(\d+)", text)
    fit_measure = re.search(r"FIT MEASURE\s+([\d.]+)", text)

    fitting_stats = {
        "R_SQUARE": float(r_square.group(1)) if r_square else None,
        "PERCENT_MASS": float(percent_mass.group(1)) if percent_mass else None,
        "CHI_SQUARE": float(chi_square.group(1)) if chi_square else None,
        "DEGREES_FREEDOM": int(degrees_freedom.group(1)) if degrees_freedom else None,
        "FIT_MEASURE": float(fit_measure.group(1)) if fit_measure else None,
    }
    fitting_df = pd.DataFrame([fitting_stats])

    # 2. SOURCE CONTRIBUTION ESTIMATES + SINGULAR VALUES
    source_pattern = r"(YES|NO)\s+(\w+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    source_matches = re.findall(source_pattern, text)

    singular_pattern = r"1 / Singular Value\s*-+\s*([\d.\s]+)\s*-+"
    singular_match = re.search(singular_pattern, text, re.DOTALL)
    singular_values = []
    if singular_match:
        values_text = singular_match.group(1).strip()
        singular_values = [float(x) for x in values_text.split() if x.strip()]

    source_data = []
    for i, match in enumerate(source_matches):
        row = {
            "STATUS": match[0],
            "EST_CODE": match[1],
            "NAME": match[2],
            "SCE_ng_m3": float(match[3]),
            "Std_Err": float(match[4]),
            "Tstat": float(match[5]),
            "Singular_Value_Inverse": (
                singular_values[i] if i < len(singular_values) else None
            ),
        }
        source_data.append(row)
    source_df = pd.DataFrame(source_data)

    # 3. SPECIES CONCENTRATIONS
    # Find the species concentration section more broadly
    species_start = text.find("SPECIES CONCENTRATIONS:")
    species_end = text.find("SOURCE NAME", species_start)

    species_data = []
    if species_start != -1 and species_end != -1:
        species_section = text[species_start:species_end]
        species_lines = species_section.split("\n")

        for line in species_lines:
            line = line.strip()
            # Look for lines that start with species codes like TMAC, FLUC, etc.
            if re.match(r"^[A-Z]{3,4}C?\s+[A-Z]{3,4}U?", line):
                # Parse the complex format: TMAC   TMAU       1.26330+- 0.04180   0.96385+- 0.06217   0.76+- 0.06    -4.0
                # or: FLUC   FLUU   *   0.08330+- 0.00330   0.08370+- 0.01273   1.00+- 0.16     0.0
                parts = line.split()
                if len(parts) >= 6:
                    species_code = parts[0]
                    uncertainty_code = parts[1]

                    # Check if there's a "*" indicating species was used in fit
                    if parts[2] == "*":
                        # Format: FLUC   FLUU   *   0.08330+- 0.00330   0.08370+- 0.01273   1.00+- 0.16     0.0
                        fit_val = parts[3]
                        calc_val = parts[4]
                        ratio_val = parts[5]
                        residual_val = parts[6] if len(parts) > 6 else None
                        used_in_fit = True
                    else:
                        # Format: TMAC   TMAU       1.26330+- 0.04180   0.96385+- 0.06217   0.76+- 0.06    -4.0
                        fit_val = parts[2]
                        calc_val = parts[3]
                        ratio_val = parts[4]
                        residual_val = parts[5]
                        used_in_fit = False

                    # Extract numeric values before +-
                    fit_match = re.search(r"([\d.]+)", fit_val) if fit_val else None
                    calc_match = re.search(r"([\d.]+)", calc_val) if calc_val else None
                    ratio_match = (
                        re.search(r"([\d.]+)", ratio_val) if ratio_val else None
                    )
                    residual_match = (
                        re.search(r"([-\d.]+)", residual_val) if residual_val else None
                    )

                    species_data.append(
                        {
                            "SPECIES_CODE": species_code,
                            "UNCERTAINTY_CODE": uncertainty_code,
                            "USED_IN_FIT": used_in_fit,
                            "FIT": float(fit_match.group(1)) if fit_match else None,
                            "CALCULATED": (
                                float(calc_match.group(1)) if calc_match else None
                            ),
                            "CALC_MEAS_RATIO": (
                                float(ratio_match.group(1)) if ratio_match else None
                            ),
                            "RESIDUAL": (
                                float(residual_match.group(1))
                                if residual_match
                                else None
                            ),
                        }
                    )

    species_df = pd.DataFrame(species_data)

    # 4. SOURCE NAME TABLE
    source_name_pattern = r"SOURCE NAME\s+(.*?)$"
    source_name_match = re.search(source_name_pattern, text, re.DOTALL)

    source_name_data = []
    if source_name_match:
        source_name_section = source_name_match.group(1)
        lines = source_name_section.strip().split("\n")

        # Extract header to get source names
        header_line = None
        source_names = []
        for line in lines:
            if "SPECIES" in line and "CALCULATED" in line and "MEASURED" in line:
                header_line = line
                parts = line.split()
                # Get source names after MEASURED
                measured_idx = next(
                    (i for i, part in enumerate(parts) if "MEASURED" in part), -1
                )
                if measured_idx >= 0:
                    source_names = parts[measured_idx + 1 :]
                break

        # Parse data lines
        for line in lines:
            line = line.strip()
            if line and not line.startswith("SPECIES") and line != header_line:
                parts = line.split()
                if len(parts) >= 3:
                    species = parts[0]
                    calculated = (
                        float(parts[1])
                        if parts[1].replace(".", "").replace("-", "").isdigit()
                        else None
                    )
                    measured = (
                        float(parts[2])
                        if parts[2].replace(".", "").replace("-", "").isdigit()
                        else None
                    )

                    row_data = {
                        "SPECIES": species,
                        "CALCULATED": calculated,
                        "MEASURED": measured,
                    }

                    # Add source values
                    for i, source_name in enumerate(source_names):
                        if i + 3 < len(parts):
                            try:
                                row_data[source_name] = float(parts[i + 3])
                            except:
                                row_data[source_name] = None
                        else:
                            row_data[source_name] = None

                    source_name_data.append(row_data)

    source_name_df = pd.DataFrame(source_name_data)

    return fitting_df, source_df, species_df, source_name_df


# Extract data from HPBG.txt file
file_path = r"d:\project\phuong_tnmt\cmb\cmb_result\pah\india_sp\HPBG.txt"


fitting_table, source_table, species_table, source_name_table = extract_cmb_from_file(
    file_path
)

print("1. FITTING STATISTICS:")
print(fitting_table)
print("\n" + "=" * 80 + "\n")

print("2. SOURCE CONTRIBUTIONS + SINGULAR VALUES:")
print(source_table)
print("\n" + "=" * 80 + "\n")

print("3. SPECIES CONCENTRATIONS:")
print(species_table)
print("\n" + "=" * 80 + "\n")

print("4. SOURCE NAME TABLE:")
print(source_name_table)

# Save to CSV files
#     fitting_table.to_csv("HPBG_fitting_statistics.csv", index=False)
#     source_table.to_csv("HPBG_source_contributions.csv", index=False)
#     species_table.to_csv("HPBG_species_concentrations.csv", index=False)
#     source_name_table.to_csv("HPBG_source_name_table.csv", index=False)

#     print("\n" + "=" * 80)
#     print("Tables saved to CSV files:")
#     print("- HPBG_fitting_statistics.csv")
#     print("- HPBG_source_contributions.csv")
#     print("- HPBG_species_concentrations.csv")
#     print("- HPBG_source_name_table.csv")

# except FileNotFoundError:
#     print(f"File not found: {file_path}")
# except Exception as e:
#     print(f"Error processing file: {e}")
# %%
