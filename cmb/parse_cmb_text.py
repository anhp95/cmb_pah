# %%
import pandas as pd
import re


def parse_fitting_statistics(text):
    """Parse FITTING STATISTICS section into a DataFrame"""

    # Extract values using regex
    r_square = re.search(r"R SQUARE\s+([\d.]+)", text)
    percent_mass = re.search(r"% MASS\s+([\d.]+)", text)
    chi_square = re.search(r"CHI SQUARE\s+([\d.]+)", text)
    degrees_freedom = re.search(r"DEGREES FREEDOM\s+(\d+)", text)
    fit_measure = re.search(r"FIT MEASURE\s+([\d.]+)", text)

    # Create dictionary
    data = {
        "R_SQUARE": float(r_square.group(1)) if r_square else None,
        "PERCENT_MASS": float(percent_mass.group(1)) if percent_mass else None,
        "CHI_SQUARE": float(chi_square.group(1)) if chi_square else None,
        "DEGREES_FREEDOM": int(degrees_freedom.group(1)) if degrees_freedom else None,
        "FIT_MEASURE": float(fit_measure.group(1)) if fit_measure else None,
    }

    return pd.DataFrame([data])


def parse_source_contributions_with_singular(text):
    """Parse SOURCE CONTRIBUTION section with singular values into a DataFrame"""

    # Extract source contribution data
    source_pattern = r"(YES|NO)\s+(\w+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    source_matches = re.findall(source_pattern, text)

    # Extract singular values
    singular_pattern = r"1 / Singular Value\s*-+\s*([\d.\s]+)\s*-+"
    singular_match = re.search(singular_pattern, text, re.DOTALL)

    singular_values = []
    if singular_match:
        values_text = singular_match.group(1).strip()
        singular_values = [float(x) for x in values_text.split() if x.strip()]

    # Combine data
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

    return pd.DataFrame(source_data)


# Example usage with your text
if __name__ == "__main__":

    # 1. Fitting Statistics Text
    fitting_text = """FITTING STATISTICS:
 
R SQUARE      0.93                       % MASS      76.3                             
CHI SQUARE      3.17              DEGREES FREEDOM         6                             
FIT MEASURE      0.75"""

    # 2. Source Contributions Text
    source_text = """SOURCE                                                                          
EST CODE   NAME     SCE(ng/m³)    Std Err      Tstat                            
----------------------------------------------------                            
YES TRPET  TRANSPET    0.25175    0.05972    4.21549                            
YES WASTE  WASTE       0.37062    0.14234    2.60372                            
YES NRDDIE NONRDDIE    0.34148    0.10607    3.21957                            
 
----------------------------------------------------
                      0.96385
                                                                                
MEASURED CONCENTRATION FOR SIZE: FINE                                           
       1.3+-     0.0                                                            
                                                                                
                   Eligible Space Collinearity Display                          
================================================================================
ELIGIBLE SPACE DIM. =   3 FOR MAX. UNC. =  0.25266  (20.% OF TOTAL MEAS. MASS)  
                                                                                
1 / Singular Value                                                              
--------------------------------------------------------------------------------
 0.03482   0.05800   0.17465                                                    
--------------------------------------------------------------------------------"""

    # Parse and display tables
    print("1. FITTING STATISTICS TABLE:")
    fitting_df = parse_fitting_statistics(fitting_text)
    print(fitting_df)
    print("\n" + "=" * 80 + "\n")

    print("2. SOURCE CONTRIBUTIONS WITH SINGULAR VALUES TABLE:")
    source_df = parse_source_contributions_with_singular(source_text)
    print(source_df)

    # Save to CSV if needed
    # fitting_df.to_csv("fitting_statistics.csv", index=False)
    # source_df.to_csv("source_contributions_with_singular.csv", index=False)

    # print("\n" + "="*80)
    # print("Tables saved to CSV files:")
    # print("- fitting_statistics.csv")
    # print("- source_contributions_with_singular.csv")
# %%
