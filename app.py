# === saare imports ===
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# === set the width ===
st.set_page_config(layout="wide", page_title="Minimal Data Explorer", initial_sidebar_state="expanded")

# === css section ai se banaya hai
st.markdown("""
<style>

/* =======================
   GLOBAL LAYOUT & THEME
======================= */
.stApp {
    background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
    color: #f5f5f5;
    font-family: 'Poppins', sans-serif;
    animation: fadeIn 1.2s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-15px);}
    to {opacity: 1; transform: translateY(0);}
}

.block-container {
    margin-top: 5vh;
    padding: 2rem 3rem;
    border-radius: 16px;
}

/* =======================
   TITLES & HEADERS
======================= */
h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    color: #4CAF50;
    letter-spacing: 0.5px;
    font-size: inherit; /* Default font size */
}

h1 {
    text-align: center;
    color: #00E676;
    margin-bottom: 0.5em;
}

h2, h3 {
    color: #00BFFF;
    border-bottom: 2px solid rgba(255,255,255,0.1);
    padding-bottom: 0.3rem;
}

/* Section separators */
hr, .stMarkdown > hr {
    border: 0;
    height: 2px;
    background: linear-gradient(90deg, #00BFFF, #4CAF50);
    border-radius: 2px;
    margin: 1.5rem 0;
}

/* =======================
   UPLOAD BOX
======================= */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed rgba(0,255,255,0.4);
    border-radius: 15px;
    padding: 2rem;
    transition: all 0.3s ease-in-out;
}
[data-testid="stFileUploader"]:hover {
    border: 2px dashed rgba(0,255,255,0.8);
    transform: scale(1.01);
}

/* Browse button inside uploader */
.stFileUploader > div > button {
    background: linear-gradient(90deg, #00BFFF, #00E676);
    border: none;
    color: black;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stFileUploader > div > button:hover {
    background: linear-gradient(90deg, #4CAF50, #00BFFF);
    transform: translateY(-2px);
}

/* =======================
   BUTTONS
======================= */
.stButton>button {
    background: linear-gradient(90deg, #00BFFF, #00E676);
    color: black;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #4CAF50, #00BFFF);
    transform: translateY(-2px);
}

/* =======================
   TABLES / DATAFRAMES
======================= */
.stDataFrame {
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
}

thead tr {
    background-color: #1E2A38 !important;
    color: #00E676 !important;
}

tbody tr:nth-child(even) {
    background-color: #151A1F !important;
}

tbody tr:hover {
    background-color: #22303F !important;
}

/* =======================
   METRIC & INFO BOXES
======================= */
.stMetric {
    background-color: #1B1F24;
    border: 1px solid #2E3238;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

/* Info/Warning messages */
.stAlert {
    background: rgba(255,255,255,0.05);
    border-left: 5px solid #00E6FF !important;
    border-radius: 10px;
    padding: 1rem;
}

/* =======================
   SELECTBOX, SLIDER, INPUTS
======================= */
.stSelectbox, .stSlider, .stTextInput {
    color: white !important;
}

.stSlider > div > div > div {
    background: linear-gradient(90deg, #00BFFF, #00E676);
}

.stSelectbox>div>div {
    background-color: #1E2A38 !important;
    border-radius: 6px;
}

/* =======================
   SIDEBAR
======================= */
[data-testid="stSidebar"] {
    background-color: #1B1F24 !important;
    border-right: 1px solid #333;
    padding-top: 1.5rem;
    backdrop-filter: blur(10px);
}
[data-testid="stSidebar"] * {
    color: #EAEAEA !important;
    font-weight: 500;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #00E676 !important;
}

/* =======================
   FOOTER / SMALL TEXT
======================= */
footer, .stCaption, .stMarkdown small {
    color: #AAAAAA !important;
    text-align: center;
    font-size: 0.9em;
    padding-top: 1rem;
}

/* =======================
   TOOLTIP / HOVER EFFECTS
======================= */
div[data-testid="stTooltip"] {
    background-color: #1F2833 !important;
    color: #FAFAFA !important;
}

</style>
""", unsafe_allow_html=True)

# === Stage 1 hai ye file uploading ka ===
if 'stage' not in st.session_state:
    st.session_state.stage = 'file_uploading'

# === saaf safai of dataset ===
def clean_data(df):
    try:
        df = df.dropna(axis=1, how='all')
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna("Unknown")
        df = df.drop_duplicates()
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"❌ Error cleaning data: {e}")
        return df

# === to handle the bade datasets ===
@st.cache_data
def load_data(file):
    def detect_delimiter(f):
        try:
            sample = f.read(2048).decode('utf-8', errors='ignore')
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except Exception:
            f.seek(0)
            return ','

    try:
        df = pd.read_csv(file, comment='#')
        st.success("✅ File loaded successfully.")
        return df
    except pd.errors.ParserError:
        st.warning("⚠️ Parsing issue detected. Trying to auto-detect the delimiter...")
        try:
            delimiter = detect_delimiter(file)
            df = pd.read_csv(file, sep=delimiter, comment='#')
            st.info(f"Detected delimiter: '{delimiter}'")
            return df
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}")
            return pd.DataFrame()
    except UnicodeDecodeError:
        st.warning("⚠️ Encoding issue detected. Trying fallback encoding ('latin1')...")
        try:
            df = pd.read_csv(file, encoding='latin1', comment='#')
            return df
        except Exception as e:
            st.error(f"❌ Encoding problem: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error while loading file: {e}")
        return pd.DataFrame()

if st.session_state.stage == 'file_uploading':
    st.title("Minimal Data Explorer 📊")
    st.header('Upload A CSV File To Start Exploring! 😊')

    import pandas as pd

    # --- sample datasets ---
    sample_datasets = {
        "Iris": "datasets/iris.csv",
        "Titanic": "datasets/train.csv",
        "Heart": "datasets/heart.csv",
        "Insurance": "datasets/insurance.csv",
        "Housing": "datasets/housing.csv"
    }

    # --- file uploader ---
    uploaded_file = st.file_uploader("Or upload your own CSV file:", type=["csv"])

    # --- sample dataset selector ---
    st.subheader("Or try a sample dataset:")
    selected_sample = st.selectbox("Choose a sample dataset:", [""] + list(sample_datasets.keys()))

    df = None  # initialize

    if selected_sample:
        df = pd.read_csv(sample_datasets[selected_sample])
        df = clean_data(df)  # if needed
        st.session_state.df = df

        st.sidebar.subheader("📁 Dataset Information")
        st.sidebar.write(f"**Dataset Name:** {selected_sample}")
        st.sidebar.write(f"**Rows:** {df.shape[0]}")
        st.sidebar.write(f"**Columns:** {df.shape[1]}")

        st.success(f"Loaded sample dataset: {selected_sample}")

    elif uploaded_file is not None:
        df = load_data(uploaded_file)
        if not df.empty:
            df = clean_data(df)
            st.session_state.df = df

            st.sidebar.subheader("📁 File Information")
            st.sidebar.write(f"**File Name:** {uploaded_file.name}")
            st.sidebar.write(f"**File Size:** {round(uploaded_file.size / 1024, 2)} KB")
            st.sidebar.write(f"**Rows:** {df.shape[0]}")
            st.sidebar.write(f"**Columns:** {df.shape[1]}")

    if df is not None:
        if st.button('📈 Start Exploring'):
            st.session_state.stage = 'show_data_options'
            st.rerun()
    else:
        st.info("Please upload a CSV file or select a sample dataset to start exploring the data.")


# === dusra stage hai; data exploration ka ===
elif st.session_state.stage == "show_data_options":
    st.header("🔴 Explore The Data & Choose One Analysis Option")


    # === data dekhlo apna ===
    st.write("### Data Preview:")
    df = st.session_state.df
    max_rows = 500 if len(df) > 500 else len(df)
    st.dataframe(df.head(max_rows))
    st.caption(f"Showing first {max_rows} rows (total {len(df)} rows).")
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Cols: {df.shape[1]}")
    st.write(f"Null Count:",df.isnull().sum())
    st.write("---")
    st.write("### Data Description:")
    try:
        st.dataframe(st.session_state.df.describe(include='all'))
    except Exception as e:
        st.warning(f"⚠️ Could not generate description: {e}")
    st.write("---")
    
    st.sidebar.title('📐Plot Settings')

    palette = st.sidebar.selectbox(
            "Select Heatmap Color Palette",
            options=['coolwarm', 'viridis', 'crest', 'magma', 'cividis', 'Blues', 'RdBu_r']
        )
    st.sidebar.header("Filter Data (Simplified)")
    # === apni marzi ka column ===
    all_columns = df.columns.tolist()
    selected_columns = st.sidebar.multiselect("Select columns to display:", all_columns, default=all_columns)

    # === apni marzi ka row ===
    num_rows = st.sidebar.slider("Number of rows to display:", min_value=10, max_value=len(df), value=min(100, len(df)))

    # === data filter karlo ===
    filtered_df = df[selected_columns].head(num_rows)
    st.subheader(f"Filtered Data: {filtered_df.shape[0]} rows x {filtered_df.shape[1]} columns")
    st.dataframe(filtered_df)
    st.session_state.filtered_df = filtered_df

    st.write("### Data Correlation")
    corr = filtered_df.corr(numeric_only = True).fillna(0)
    st.dataframe(corr)
    

    if st.button("Visualize Correlation"):
        st.warning('According to the number of rows & cols selected!')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr,cmap=palette, linewidths=0.5, ax=ax)
        ax.set_title(f"Correlation Matrix (Palette: {palette})")
        st.pyplot(fig)

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numerical_columns:
        st.subheader("Numerical Column Statistics")
        stats_df = df[numerical_columns].agg(['mean', 'median', 'std', 'min', 'max', 'skew', 'kurtosis']).T
        st.dataframe(stats_df)

    st.sidebar.title('> Next Step:')


    st.sidebar.header("Choose An Analysis Option:")
    selected_option = st.sidebar.selectbox("Select an option:", ['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])

    if st.sidebar.button('📈 Analyze'):
        st.session_state.selected_option = selected_option
        st.session_state.stage = 'show_results'
        st.rerun()

    if st.sidebar.button('🔙 Go Back to Upload'):
        st.session_state.stage = 'file_uploading'
        st.rerun()

elif st.session_state.stage == 'show_results':
    st.header("🔵 Data Analysis Section")

    df = st.session_state.df
    
    # Use st.button with a key to avoid conflicts
    if st.sidebar.button('🔙 Go Back to Options', key='go_back_results'):
        st.session_state.stage = 'show_data_options'
        st.rerun()

    selected_analysis = st.session_state.selected_option
    st.write(f"### You selected: **{selected_analysis}**")

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object','category']).columns.tolist()
    
    st.sidebar.header("🛠️ Data & Chart Options")

    # Univariate Analysis Section
    if selected_analysis == 'Univariate Analysis':
        if not numerical_columns and not categorical_columns:
            st.warning("The selected data has no valid columns for univariate analysis.")
        else:
            analysis_column = st.sidebar.selectbox('Select a column:', numerical_columns + categorical_columns)
            
            if analysis_column in numerical_columns:
                fig, ax = plt.subplots()
                charts = ['Line Chart', 'Area Chart',"Histogram", "KDE", "Histogram + KDE"]
                chart_type = st.sidebar.selectbox('Chart Type:', charts)
                
                if chart_type == 'Line Chart':
                    st.write(f"### Line Chart for {analysis_column}")
                    chart = alt.Chart(df.reset_index()).mark_line().encode(
                        x=alt.X('index', title='Observation Index'),
                        y=alt.Y(analysis_column, title=analysis_column)
                    )
                    st.altair_chart(chart, use_container_width=True)
                
                elif chart_type == 'Area Chart':
                    st.write(f"### Area Chart for {analysis_column}")
                    chart = alt.Chart(df.reset_index()).mark_area(line=True).encode(
                        x=alt.X('index', title='Observation Index'),
                        y=alt.Y(analysis_column, title=analysis_column)
                    )
                    st.altair_chart(chart, use_container_width=True)

                elif chart_type == 'Histogram':
                    sns.histplot(df[analysis_column], kde=False, ax=ax, color='teal')
                    ax.set_title(f"Distribution of {analysis_column}")
                    st.pyplot(fig)
                elif chart_type == "KDE":
                    sns.kdeplot(df[analysis_column], ax=ax, fill=True, color='teal')
                    ax.set_title(f"Distribution of {analysis_column}")
                    st.pyplot(fig)
                elif chart_type == "Histogram + KDE":
                    sns.histplot(df[analysis_column], kde=True, ax=ax, color='teal')
                    ax.set_title(f"Distribution of {analysis_column}")
                    st.pyplot(fig)
                


            elif analysis_column in categorical_columns:
                chart_type = st.sidebar.selectbox('Chart Type:', ['Bar Chart'])
                
                if chart_type == 'Bar Chart':
                    st.write(f"### Bar Chart for {analysis_column}")
                    frequency_data = df[analysis_column].value_counts().reset_index()
                    frequency_data.columns = [analysis_column, 'Frequency']
                    
                    chart = alt.Chart(frequency_data).mark_bar().encode(
                        x=alt.X(analysis_column, title=analysis_column),
                        y=alt.Y('Frequency', title='Frequency')
                    )
                    st.altair_chart(chart, use_container_width=True)
            
            else:
                st.warning("Please select a valid column.")

    elif selected_analysis == 'Bivariate Analysis':
      st.subheader("📊 Bivariate Analysis")

      all_columns = numerical_columns + categorical_columns

      if len(all_columns) < 2:
          st.warning("Please select a dataset with at least two columns for bivariate analysis.")
      else:
          # Use filtered dataset if available
          df_used = st.session_state.get("filtered_df", df)

          # Downsample large datasets for plotting
          if len(df_used) > 5000:
              df_plot = df_used.sample(5000, random_state=42)
              st.info("Using a random sample of 5000 rows for faster plotting.")
          else:
              df_plot = df_used

          x_axis = st.sidebar.selectbox('Select X-axis:', all_columns, key='bivariate_x')
          y_axis = st.sidebar.selectbox('Select Y-axis:', all_columns, key='bivariate_y')

          # ---------------- NUMERIC vs NUMERIC ----------------
          if x_axis in numerical_columns and y_axis in numerical_columns:
              chart_options = ['Scatter Plot', 'Line Chart', 'Regression Plot']
              chart_type = st.sidebar.selectbox('Chart Type:', chart_options, key='bivariate_num')
              hue = st.sidebar.selectbox('Color by (optional):', ['None'] + categorical_columns, index=0)

              if chart_type == 'Scatter Plot':
                  chart = alt.Chart(df_plot).mark_circle(size=60, opacity=0.7).encode(
                      x=x_axis,
                      y=y_axis,
                      color=hue if hue != 'None' else alt.value('#00BFFF'),
                      tooltip=[x_axis, y_axis] + ([hue] if hue != 'None' else [])
                  ).properties(
                      title=f"Scatter Plot: {x_axis} vs {y_axis}"
                  ).interactive()

              elif chart_type == 'Line Chart':
                  chart = alt.Chart(df_plot).mark_line().encode(
                      x=x_axis,
                      y=y_axis,
                      color=hue if hue != 'None' else alt.value('#FFA500'),
                      tooltip=[x_axis, y_axis] + ([hue] if hue != 'None' else [])
                  ).properties(
                      title=f"Line Chart: {x_axis} vs {y_axis}"
                  ).interactive()

              elif chart_type == 'Regression Plot':
                  import numpy as np
                  import pandas as pd
                  import statsmodels.api as sm

                  X = sm.add_constant(df_plot[x_axis])
                  model = sm.OLS(df_plot[y_axis], X).fit()
                  df_plot['Predicted'] = model.predict(X)

                  chart = alt.Chart(df_plot).mark_circle(size=60, opacity=0.5).encode(
                      x=x_axis,
                      y=y_axis,
                      color=alt.value("#1f77b4")
                  ) + alt.Chart(df_plot).mark_line(color='red').encode(
                      x=x_axis,
                      y='Predicted'
                  )
                  chart = chart.properties(title=f"Regression Plot: {y_axis} ~ {x_axis}")

              st.altair_chart(chart, use_container_width=True)

              # Show correlation
              corr_val = df_plot[[x_axis, y_axis]].corr().iloc[0, 1]
              st.metric(label="Correlation Coefficient", value=round(corr_val, 3))

          # ---------------- CATEGORY vs NUMERIC ----------------
          elif x_axis in categorical_columns and y_axis in numerical_columns:
              chart_options = ['Bar Chart', 'Box Plot', 'Violin Plot']
              chart_type = st.sidebar.selectbox('Chart Type:', chart_options, key='bivariate_cat')
              agg_func = st.sidebar.selectbox('Aggregation:', ['mean', 'median', 'sum', 'count'], index=0)
              hue = st.sidebar.selectbox('Color by (optional):', ['None'] + categorical_columns, index=0)

              # Aggregate data for better performance
              df_agg = df_plot.groupby(x_axis)[y_axis].agg(agg_func).reset_index()

              if chart_type == 'Bar Chart':
                  chart = alt.Chart(df_agg).mark_bar().encode(
                      x=x_axis,
                      y=y_axis,
                      color=hue if hue != 'None' else alt.value('#4C78A8'),
                      tooltip=[x_axis, y_axis]
                  ).properties(
                      title=f"Bar Chart ({agg_func}) of {y_axis} by {x_axis}"
                  ).interactive()

              elif chart_type == 'Box Plot':
                  chart = alt.Chart(df_plot).mark_boxplot(extent='min-max').encode(
                      x=x_axis,
                      y=y_axis,
                      color=hue if hue != 'None' else alt.value('#F39C12'),
                      tooltip=[x_axis, y_axis]
                  ).properties(
                      title=f"Box Plot of {y_axis} by {x_axis}"
                  ).interactive()

              elif chart_type == 'Violin Plot':
                  import seaborn as sns
                  import matplotlib.pyplot as plt
                  fig, ax = plt.subplots(figsize=(8, 4))
                  sns.violinplot(x=x_axis, y=y_axis, data=df_plot, ax=ax)
                  st.pyplot(fig)

              st.altair_chart(chart, use_container_width=True)

          # ---------------- CATEGORY vs CATEGORY ----------------
          elif x_axis in categorical_columns and y_axis in categorical_columns:
              st.subheader("📉 Category vs Category (Proportions)")

              df_cross = pd.crosstab(df_plot[x_axis], df_plot[y_axis], normalize='index') * 100
              chart = alt.Chart(df_cross.reset_index().melt(x_axis, var_name=y_axis, value_name='Percentage')).mark_bar().encode(
                  x=x_axis,
                  y='Percentage:Q',
                  color=y_axis
              ).properties(title=f"Percentage Distribution of {y_axis} by {x_axis}")
              st.altair_chart(chart, use_container_width=True)

          else:
              st.info("Please select valid X and Y columns.")


    elif selected_analysis == 'Multivariate Analysis':
      st.header("📈 Multivariate Analysis (3 or More Variables)")

      # Use filtered dataset if available
      df_used = st.session_state.get("filtered_df", df)

      # Downsample for speed
      if len(df_used) > 5000:
          df_plot = df_used.sample(5000, random_state=42)
          st.info("Using a random sample of 5000 rows for faster rendering.")
      else:
          df_plot = df_used

      # --------------------------- 1. Bubble Chart ---------------------------
      st.subheader("🟢 Bubble Chart (3–4 Variables)")

      if len(numerical_columns) >= 2 and categorical_columns:
          col1, col2, col3 = st.columns(3)
          with col1: x_bubble = st.selectbox('X-Axis (Numerical):', numerical_columns, key='x_bubble')
          with col2: y_bubble = st.selectbox('Y-Axis (Numerical):', numerical_columns, key='y_bubble')
          with col3: hue_bubble = st.selectbox('Color (Categorical):', categorical_columns, key='hue_bubble')

          size_bubble = st.sidebar.selectbox('Bubble Size (Optional):', ['None'] + numerical_columns, key='size_bubble')
          tooltip_list = [x_bubble, y_bubble, hue_bubble]
          if size_bubble != 'None':
              tooltip_list.append(size_bubble)

          bubble_chart = alt.Chart(df_plot).mark_circle(opacity=0.7).encode(
              x=alt.X(x_bubble, title=x_bubble),
              y=alt.Y(y_bubble, title=y_bubble),
              color=alt.Color(hue_bubble, legend=alt.Legend(title=hue_bubble)),
              size=size_bubble if size_bubble != 'None' else alt.value(100),
              tooltip=tooltip_list
          ).properties(
              title=f"Bubble Chart: {x_bubble} vs {y_bubble} (Color: {hue_bubble})"
          ).interactive()
          st.altair_chart(bubble_chart, use_container_width=True)
      else:
          st.warning("Needs ≥2 numerical and ≥1 categorical columns for Bubble Chart.")

      st.markdown("---")

      # --------------------------- 2. Categorical Heatmap ---------------------------
      st.subheader("🔥 Categorical Heatmap (2 Categories + 1 Numeric)")

      if len(categorical_columns) >= 2 and len(numerical_columns) >= 1:
          col1, col2, col3 = st.columns(3)
          with col1: x_heatmap = st.selectbox('X-Axis (Categorical):', categorical_columns, key='x_heatmap')
          with col2: y_heatmap = st.selectbox('Y-Axis (Categorical):', categorical_columns, key='y_heatmap')
          with col3: color_heatmap = st.selectbox('Color (Numerical):', numerical_columns, key='color_heatmap')

          agg_func = st.sidebar.selectbox("Aggregation for Heatmap:", ['mean', 'median', 'sum', 'count'], index=0)

          if x_heatmap == y_heatmap:
             st.error("❌ X-axis and Y-axis cannot be the same column for the heatmap.")
          else:
            heatmap_data = (
                df_plot.groupby([x_heatmap, y_heatmap])[color_heatmap]
                .agg(agg_func)
                .reset_index()
            )
            heatmap_data.rename(columns={color_heatmap: 'Value'}, inplace=True)

            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X(x_heatmap, title=x_heatmap),
                y=alt.Y(y_heatmap, title=y_heatmap),
                color=alt.Color('Value:Q', scale=alt.Scale(scheme='viridis'),
                                title=f'{agg_func.title()} {color_heatmap}'),
                tooltip=[x_heatmap, y_heatmap, alt.Tooltip('Value:Q', format='.2f')]
            ).properties(
                title=f"Heatmap: {agg_func.title()} {color_heatmap} by {x_heatmap} and {y_heatmap}"
            ).interactive()

            st.altair_chart(heatmap, use_container_width=True)

      else:
          st.warning("Needs ≥2 categorical and ≥1 numerical column for Heatmap.")

      st.markdown("---")

      # --------------------------- 3. Pairwise Relationship ---------------------------
      st.subheader("🔍 Pairwise Relationships (Pairplot)")

      if len(numerical_columns) >= 3:
        selected_pair_cols = st.multiselect(
            "Select Numerical Columns (max 5 for speed):",
            numerical_columns,
            default=numerical_columns[:3]
        )
        hue_pair = st.selectbox(
            "Color by (optional categorical):",
            ['None'] + categorical_columns,
            index=0
        )

        # Sample data for speed and avoid overloading the pairplot
        df_plot_used = df_plot.copy()
        if len(df_plot_used) > 3000:
            df_plot_used = df_plot_used.sample(3000, random_state=42)
            st.info("Using a random sample of 3000 rows for faster pairplot rendering.")

        if st.button("Generate Pairplot"):
            sns.set(style="ticks")

            # Limit number of columns to prevent overplotting
            if len(selected_pair_cols) > 5:
                st.warning("⚠️ Please select up to 5 columns to prevent lag.")
            elif len(selected_pair_cols) < 2:
                st.warning("⚠️ Please select at least 2 numerical columns.")
            else:
                try:
                    fig = sns.pairplot(
                        df_plot_used[selected_pair_cols + ([hue_pair] if hue_pair != 'None' else [])],
                        hue=hue_pair if hue_pair != 'None' else None,
                        diag_kind="kde",
                        plot_kws={'alpha': 0.6}
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"❌ Error generating pairplot: {e}")
        else:
            st.info("Requires at least 3 numerical columns for pairwise relationship analysis.")


      st.markdown("---")

      # --------------------------- 4. Parallel Coordinates Plot ---------------------------
      st.subheader("🧭 Parallel Coordinates (Multidimensional View)")

      if len(numerical_columns) >= 3:
          selected_parallel_cols = st.multiselect("Select columns for Parallel Coordinates:", numerical_columns, default=numerical_columns[:4])
          color_parallel = st.selectbox("Color by (categorical):", ['None'] + categorical_columns, index=0)

          if st.button("Generate Parallel Coordinates Plot"):
              from pandas.plotting import parallel_coordinates
              import matplotlib.pyplot as plt

              if color_parallel != 'None':
                  df_plot_sub = df_plot[selected_parallel_cols + [color_parallel]].dropna().copy()
                  fig, ax = plt.subplots(figsize=(9, 5))
                  parallel_coordinates(df_plot_sub, class_column=color_parallel, color=plt.cm.tab10.colors, alpha=0.6)
                  plt.title("Parallel Coordinates Plot")
                  st.pyplot(fig)
              else:
                  st.warning("Please select a categorical column for coloring.")
      else:
          st.info("Requires ≥3 numerical columns for Parallel Coordinates.")


else:
    st.info('Please upload a CSV file to begin.')
