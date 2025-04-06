import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

@st.cache_data
def load_generic_data(csv_path, value_name):
    df = pd.read_csv(csv_path, skiprows=4)
    df_long = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="Year",
        value_name=value_name
    ).dropna(subset=[value_name])
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long.sort_values("Year", inplace=True)
    return df_long

def main():
    st.title("World Bank Data Explorer")
    
    # File paths and corresponding value names
    files = {
        "Birth Rate": "./csv-data/birth_rate_world_bank/API_SP.DYN.CBRT.IN_DS2_en_csv_v2_13718.csv",
        "Real GDP (USD)": "./csv-data/real_gdp_usd_world_bank/API_NY.ADJ.NNTY.PC.CD_DS2_en_csv_v2_14320.csv",
        "Female Labor Force Participation (%)": "./csv-data/female_labor_force_participation_world_bank/API_SL.TLF.CACT.FE.ZS_DS2_en_csv_v2_13361.csv"
    }
    
    # Load datasets generically
    datasets = {name: load_generic_data(path, name) for name, path in files.items()}
    
    # Combine countries for filtering
    countries = sorted(set().union(*[set(df["Country Name"].unique()) for df in datasets.values()]))
    selected_countries = st.multiselect("Select countries:", countries, default=["Sweden", "Switzerland"])
    
    if selected_countries:
        # Filter each dataset by selected countries
        filtered = {name: df[df["Country Name"].isin(selected_countries)] for name, df in datasets.items()}
        col_keys = list(filtered.keys())
        
        # --- Regular Time Series Chart with Toggle Options ---
        st.subheader("Time Series Chart")
        left_dataset = st.selectbox("Select dataset for left axis:", col_keys, index=0, key="ts_left")
        right_dataset = st.selectbox("Select dataset for right axis:", col_keys, index=1, key="ts_right")
        
        if left_dataset == right_dataset:
            st.warning("Please select two distinct datasets for the time series chart.")
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            for country in selected_countries:
                country_data = filtered[left_dataset][filtered[left_dataset]["Country Name"] == country]
                ax1.plot(country_data["Year"], country_data[left_dataset],
                         marker='o', label=f"{country} {left_dataset}")
            ax1.set_xlabel("Year")
            ax1.set_ylabel(left_dataset, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            for country in selected_countries:
                country_data = filtered[right_dataset][filtered[right_dataset]["Country Name"] == country]
                ax2.plot(country_data["Year"], country_data[right_dataset],
                         linestyle='--', marker='x', label=f"{country} {right_dataset}")
            ax2.set_ylabel(right_dataset, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            st.pyplot(fig)
        
        # --- Display Summary Statistics for Each Dataset ---
        for name, df in filtered.items():
            st.subheader(f"{name} Summary Statistics")
            st.dataframe(df.groupby("Country Name")[name].describe())
        
        # --- Correlation Chart (as before) ---
        st.subheader("Correlation Between Two Datasets")
        x_dataset = st.selectbox("Select dataset for X-axis:", col_keys, index=0, key="corr_x")
        y_dataset = st.selectbox("Select dataset for Y-axis:", col_keys, index=1, key="corr_y")
        
        if x_dataset == y_dataset:
            st.warning("Please select two distinct datasets for correlation.")
        else:
            merged = pd.merge(
                filtered[x_dataset],
                filtered[y_dataset],
                on=["Country Name", "Country Code", "Year"],
                suffixes=(f"_{x_dataset}", f"_{y_dataset}")
            )
            if merged.empty:
                st.write("No overlapping data for the selected datasets.")
            else:
                fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                ax_corr.scatter(merged[x_dataset], merged[y_dataset])
                z = np.polyfit(merged[x_dataset], merged[y_dataset], 1)
                p_line = np.poly1d(z)
                ax_corr.plot(merged[x_dataset], p_line(merged[x_dataset]), linestyle='--', color='red')
                r, p_value = pearsonr(merged[x_dataset], merged[y_dataset])
                st.write(f"Pearson's correlation coefficient: {r:.2f}")
                st.write(f"P-value: {p_value:.3f}")
                ax_corr.set_xlabel(x_dataset)
                ax_corr.set_ylabel(y_dataset)
                st.pyplot(fig_corr)

if __name__ == "__main__":
    main()
