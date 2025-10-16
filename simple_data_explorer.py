import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import csv
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Minimal Data Explorer", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* Global Background and Text Color */
    .stApp {
        background-color: #0E1117; 
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    /* Main Content Styling (Padding and Max Width) */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Header/Title Styling */
    h1 {
        font-size: 2.5em;
        font-weight: 700;
        color: #4CAF50; /* Primary accent color (Green) */
        margin-bottom: 0.5em;
    }
    /* Sub-headers */
    h3 {
        color: #00BFFF; /* Secondary accent color (Blue) */
        border-bottom: 2px solid #262730;
        padding-bottom: 0.5rem;
    }
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1F2833; 
        color: #FAFAFA;
        padding: 1rem;
    }
    /* Buttons/Info Boxes */
    .stButton>button, .stInfo {
        border-radius: 8px;
        border: 1px solid #4CAF50;
        color: #FAFAFA;
        background-color: #1F2833;
    }
    /* Dataframe look */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'stage' not in st.session_state:
    st.session_state.stage = 'file_uploading'

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

    uploaded_file = st.file_uploader("Choose a CSV file:", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if not df.empty:
            st.session_state.df = clean_data(df)

            st.sidebar.subheader("📁 File Information")
            st.sidebar.write(f"**File Name:** {uploaded_file.name}")
            st.sidebar.write(f"**File Size:** {round(uploaded_file.size / 1024, 2)} KB")
            st.sidebar.write(f"**Rows:** {df.shape[0]}")
            st.sidebar.write(f"**Columns:** {df.shape[1]}")

            if st.button('📈 Start Exploring'):
                st.session_state.stage = 'show_data_options'
                st.rerun()
    else:
        st.info("Please upload a CSV file to start exploring the data.")

elif st.session_state.stage == "show_data_options":
    st.header("🔴 Explore The Data & Choose One Analysis Option")

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
    st.write("### Data Correlation")
    corr = df.corr(numeric_only = True)
    st.dataframe(corr)
    st.sidebar.title('📐Plot Settings')

    palette = st.sidebar.selectbox(
            "Select Heatmap Color Palette",
            options=['coolwarm', 'viridis', 'crest', 'magma', 'cividis', 'Blues', 'RdBu_r']
        )

    if st.button("Visualize Correlation"):
        
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
    st.sidebar.header("Filter Data (Simplified)")

    # Select columns to display
    all_columns = df.columns.tolist()
    selected_columns = st.sidebar.multiselect("Select columns to display:", all_columns, default=all_columns)

    # Select number of rows to display
    num_rows = st.sidebar.slider("Number of rows to display:", min_value=10, max_value=min(1000, len(df)), value=min(100, len(df)))

    # Apply filtering
    filtered_df = df[selected_columns].head(num_rows)
    st.subheader(f"Filtered Data: {filtered_df.shape[0]} rows x {filtered_df.shape[1]} columns")
    st.dataframe(filtered_df)

    # Save filtered data to session_state
    st.session_state.filtered_df = filtered_df

    st.sidebar.title('> Next Step:')
    st.sidebar.header("Choose An Analysis Option:")
    selected_option = st.sidebar.selectbox("Select an option:", ['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])

    if st.sidebar.button('📈 Analyze'):
        st.session_state.selected_option = selected_option
        st.session_state.stage = 'show_results'
        st.rerun()

    if st.sidebar.button('🔙 Go Back'):
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
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
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
        all_columns = numerical_columns + categorical_columns
        
        if len(all_columns) < 2:
            st.warning("Please select a dataset with at least two columns for bivariate analysis.")
        else:
            x_axis = st.sidebar.selectbox('Select X-axis:', all_columns, key='bivariate_x')
            y_axis = st.sidebar.selectbox('Select Y-axis:', all_columns, key='bivariate_y')

            if x_axis in numerical_columns and y_axis in numerical_columns:
                charts = ['Scatter Chart', 'Line Chart']
                chart_type = st.sidebar.selectbox('Chart Type:', charts, key='bivariate_chart_num')
                hue = st.sidebar.selectbox('Select Hue (optional):', ['None'] + categorical_columns, index=0, key='bivariate_hue_num')

                if chart_type == 'Line Chart':
                    chart = alt.Chart(df).mark_line().encode(
                        x=alt.X(x_axis, title=x_axis),
                        y=alt.Y(y_axis, title=y_axis),
                        color=alt.Color(hue) if hue != 'None' else alt.value('#00BFFF'),
                        tooltip=[x_axis, y_axis, hue] if hue != 'None' else [x_axis, y_axis]
                    ).properties(
                        title=f'Line Chart: {x_axis} vs {y_axis}'
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                    
                elif chart_type == 'Scatter Chart':
                    if hue == 'None':
                        chart = alt.Chart(df).mark_circle().encode(
                            x=alt.X(x_axis, title=x_axis),
                            y=alt.Y(y_axis, title=y_axis)
                        ).properties(
                            title=f'Scatter Chart: {x_axis} vs {y_axis}'
                        ).interactive()
                    else:
                        chart = alt.Chart(df).mark_circle().encode(
                            x=alt.X(x_axis, title=x_axis),
                            y=alt.Y(y_axis, title=y_axis),
                            color=hue,
                            tooltip=[x_axis, y_axis, hue]
                        ).properties(
                            title=f'Scatter Chart: {x_axis} vs {y_axis}'
                        ).interactive()
                    st.altair_chart(chart, use_container_width=True)
            
            elif x_axis in categorical_columns and y_axis in numerical_columns:
                charts = ['Bar Chart', 'Box Plot']
                chart_type = st.sidebar.selectbox('Chart Type:', charts, key='bivariate_chart_cat')
                hue = st.sidebar.selectbox('Select Hue (optional):', ['None'] + categorical_columns, index=0, key='bivariate_hue_cat')

                if chart_type == 'Bar Chart':
                    if hue == 'None':
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X(x_axis, title=x_axis),
                            y=alt.Y(y_axis, title=y_axis),
                        ).properties(
                            title=f'Bar Chart: {y_axis} by {x_axis}'
                        ).interactive()
                    else:
                        chart = alt.Chart(df).mark_bar().encode(
                            column=alt.Column(x_axis, header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
                            x=alt.X(hue, axis=None),  
                            y=alt.Y(y_axis, title=y_axis),
                            color=hue,
                            tooltip=[x_axis, y_axis, hue]
                        ).properties(
                            title=f'Grouped Bar Chart: {y_axis} by {x_axis} & {hue}'
                        ).interactive()
                    st.altair_chart(chart, use_container_width=True)

                elif chart_type == 'Box Plot':
                    if hue == 'None':
                        chart = alt.Chart(df).mark_boxplot().encode(
                            x=alt.X(x_axis, title=x_axis),
                            y=alt.Y(y_axis, title=y_axis)
                        ).properties(
                            title=f'Box Plot: {y_axis} by {x_axis}'
                        ).interactive()
                    else:
                        chart = alt.Chart(df).mark_boxplot().encode(
                            x=alt.X(x_axis, title=x_axis),
                            y=alt.Y(y_axis, title=y_axis),
                            color=hue
                        ).properties(
                            title=f'Box Plot: {y_axis} by {x_axis} & {hue}'
                        ).interactive()
                    st.altair_chart(chart, use_container_width=True)

            elif x_axis in numerical_columns and y_axis in categorical_columns:
                st.warning("Please select a numerical column for the Y-axis and a categorical column for the X-axis for a bar or box plot.")
            elif x_axis in categorical_columns and y_axis in categorical_columns:
                st.info("Try a Heatmap (Multivariate Analysis) to compare two categorical columns.")

    elif selected_analysis == 'Multivariate Analysis':
        st.header("Analyzing 3 or More Variables")
        
        # 1. Bubble Chart (3+ Variables)
        st.subheader("Bubble Chart (3 Variables)")
        if len(numerical_columns) < 2 or not categorical_columns:
            st.warning("Requires at least 2 Numerical Columns (X/Y) and 1 Categorical Column (Color).")
        else:
            col1, col2, col3 = st.columns(3)
            with col1: x_bubble = st.selectbox('X-Axis (Numerical):', numerical_columns, key='x_bubble')
            with col2: y_bubble = st.selectbox('Y-Axis (Numerical):', numerical_columns, key='y_bubble')
            with col3: hue_bubble = st.selectbox('Color (Categorical):', categorical_columns, key='hue_bubble')
            size_bubble = st.sidebar.selectbox('Size (Numerical/3rd Variable):', ['None'] + numerical_columns, key='size_bubble')
            
            tooltip_list = [x_bubble, y_bubble, hue_bubble]
            if size_bubble != 'None':
                tooltip_list.append(size_bubble)

            bubble_chart = alt.Chart(df).mark_circle().encode(
                x=alt.X(x_bubble, title=x_bubble),
                y=alt.Y(y_bubble, title=y_bubble),
                color=hue_bubble,
                size=size_bubble if size_bubble != 'None' else alt.value(100),
                tooltip=tooltip_list
            ).properties(
                title=f'Bubble Chart: {x_bubble} vs {y_bubble} (Colored by {hue_bubble})'
            ).interactive()
            st.altair_chart(bubble_chart, use_container_width=True)
            
            st.markdown("---")

        
        st.subheader("Heatmap (Correlation/Intensity)")
        if len(numerical_columns) < 1 or len(categorical_columns) < 2:
            st.warning("Requires 2 Categorical Columns (X/Y) and 1 Numerical Column (Color).")
        else:
            col_1, col_2, col_3 = st.columns(3)
            with col_1: x_heatmap = st.selectbox('X-Axis (Categorical):', categorical_columns, key='x_heatmap')
            with col_2: y_heatmap = st.selectbox('Y-Axis (Categorical):', categorical_columns, key='y_heatmap')
            with col_3: color_heatmap = st.selectbox('Color (Numerical/Intensity):', numerical_columns, key='color_heatmap')

            
            heatmap_data = df.groupby([x_heatmap, y_heatmap], as_index=False)[color_heatmap].mean()
            
            
            heatmap_data = heatmap_data.rename(columns={color_heatmap: 'Mean Value'})
            

            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X(x_heatmap, title=x_heatmap),
                y=alt.Y(y_heatmap, title=y_heatmap),
                color=alt.Color('Mean Value', scale=alt.Scale(scheme='viridis'), title=f'Avg {color_heatmap}'),
                tooltip=[x_heatmap, y_heatmap, alt.Tooltip('Mean Value', format='.2f')]
            ).properties(
                title=f'Heatmap: Average {color_heatmap} by {x_heatmap} and {y_heatmap}'
            ).interactive()
            
            st.altair_chart(heatmap, use_container_width=True)

else:
    st.info('Please upload a CSV file to begin.')






