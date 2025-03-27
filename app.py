import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path, skiprows=4)
    df_long = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="Year",
        value_name="Birth Rate"
    ).dropna(subset=["Birth Rate"])
    # Convert year to numeric for proper sorting
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long.sort_values("Year", inplace=True)
    return df_long

def main():
    st.title("World Bank Birth Rate Explorer")

    # 1. Load data
    csv_path = "./csv-data/birth_rate_world_bank/API_SP.DYN.CBRT.IN_DS2_en_csv_v2_13718.csv"
    df_long = load_data(csv_path)

    # 2. Country selection
    countries = sorted(df_long["Country Name"].unique())
    selected_countries = st.multiselect("Select countries:", countries, default=["Sweden", "Switzerland"])

    # 3. Filter data for selected countries
    if selected_countries:
        df_filtered = df_long[df_long["Country Name"].isin(selected_countries)]

        # 4. Plot with matplotlib
        plt.figure()
        for country in selected_countries:
            df_country = df_filtered[df_filtered["Country Name"] == country]
            plt.plot(df_country["Year"], df_country["Birth Rate"], label=country)

        plt.legend()
        plt.xlabel("Year")
        plt.ylabel("Birth Rate (per 1,000 people)")
        st.pyplot(plt.gcf())

        # 5. Summary statistics table
        st.subheader("Summary Statistics")
        summary_stats = df_filtered.groupby("Country Name")["Birth Rate"].describe()
        st.dataframe(summary_stats)

if __name__ == "__main__":
    main()
