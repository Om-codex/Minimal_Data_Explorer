import streamlit as st
import pandas as pd
import altair as alt

# 1. Page Configuration and Theme Setup
st.set_page_config(layout="wide", page_title="Minimal Data Explorer", initial_sidebar_state="expanded")

# Inject Custom CSS for a Clean, Minimal Dark Theme
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

st.title("Minimal Data Explorer")

uploaded_file = st.file_uploader("Upload a CSV file here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.header("Display Options")

    selected_columns = st.sidebar.multiselect(
        'Select columns to display:',
        options=df.columns.tolist(),
        default=df.columns.tolist()[:5]
    )

    num_rows = len(df)
    start_row = st.sidebar.number_input(
        'Start row (0-indexed):',
        min_value=0,
        max_value=num_rows - 1,
        value=0
    )
    end_row = st.sidebar.number_input(
        'End row (0-indexed, exclusive):',
        min_value=0,
        max_value=num_rows,
        value=min(20, num_rows)
    )

    if not selected_columns:
        st.warning("Please select at least one column to display.")
    elif start_row >= end_row:
        st.warning("End row must be greater than the start row.")
    else:
        
        df = df[selected_columns].iloc[start_row:end_row]
        st.write('### Selected Data Preview ###')
        st.dataframe(df)

        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        if not numerical_columns and not categorical_columns:
            st.warning("The uploaded file does not contain any valid columns for plotting.")
        else:
            st.sidebar.header("Chart Options")
            
            charts = ['Line Chart', 'Area Chart', 'Bar Chart', 'Scatter Chart', 'Map Chart']
            chart_type = st.sidebar.selectbox('Please select a Chart Type:', charts)
            
            if chart_type == 'Line Chart':
                if not numerical_columns:
                    st.warning("No numerical columns for a Line Chart.")
                else:
                    x_axis = st.sidebar.selectbox('Select X-axis:', numerical_columns, index=0)
                    y_axis = st.sidebar.selectbox('Select Y-axis:', numerical_columns, index=min(1, len(numerical_columns) - 1))
                    st.write(f"### {chart_type}: {y_axis} vs {x_axis} ###")
                    st.line_chart(df.set_index(x_axis)[[y_axis]])
                    
            elif chart_type == 'Area Chart':
                if not numerical_columns:
                    st.warning("No numerical columns for an Area Chart.")
                else:
                    x_axis = st.sidebar.selectbox('Select X-axis:', numerical_columns, index=0)
                    y_axis = st.sidebar.selectbox('Select Y-axis:', numerical_columns, index=min(1, len(numerical_columns) - 1))
                    st.write(f"### {chart_type}: {y_axis} vs {x_axis} ###")
                    st.area_chart(df.set_index(x_axis)[[y_axis]])
                    
            elif chart_type == 'Bar Chart':
                if not categorical_columns or not numerical_columns:
                    st.warning("A Bar Chart requires both categorical and numerical columns.")
                else:
                    x_axis = st.sidebar.selectbox('Select X-axis:', categorical_columns)
                    y_axis = st.sidebar.selectbox('Select Y-axis:', numerical_columns)
                    
                    # We will use this categorical column to group the bars
                    grouping_column = st.sidebar.selectbox('Group by (optional):', ['None'] + categorical_columns, index=0)
                    
                    st.write(f"### {chart_type}: {y_axis} by {x_axis} ###")

                    # The core change is here. We are no longer using 'hue'.
                    if grouping_column == 'None':
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X(x_axis),
                            y=alt.Y(y_axis),
                            tooltip=[x_axis, y_axis]
                        ).interactive()
                    else:
                        # Create a grouped bar chart using alt.Column()
                        chart = alt.Chart(df).mark_bar().encode(
                            # Use the grouping column to separate the bars
                            column=alt.Column(grouping_column, header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
                            x=alt.X(x_axis, axis=None),  # Hide the x-axis label for cleaner look
                            y=y_axis,
                            # Add color based on the x_axis to differentiate groups within a single bar
                            color=x_axis,
                            tooltip=[x_axis, y_axis, grouping_column]
                        ).interactive()

                st.altair_chart(chart, use_container_width=True)
                    
            elif chart_type == 'Scatter Chart':
                if not numerical_columns:
                    st.warning("A Scatter Chart requires numerical columns.")
                else:
                    x_axis = st.sidebar.selectbox('Select X-axis:', numerical_columns)
                    y_axis = st.sidebar.selectbox('Select Y-axis:', numerical_columns)
                    hue = st.sidebar.selectbox('Select Hue (optional):', ['None'] + categorical_columns, index=0)

                    st.write(f"### {chart_type}: {y_axis} vs {x_axis} ###")
                    
                    if hue == 'None':
                        chart = alt.Chart(df).mark_circle().encode(
                            x=x_axis,
                            y=y_axis,
                            tooltip=[x_axis, y_axis]
                        ).interactive()
                    else:
                        chart = alt.Chart(df).mark_circle().encode(
                            x=x_axis,
                            y=y_axis,
                            color=hue,
                            tooltip=[x_axis, y_axis, hue]
                        ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                    
            elif chart_type == 'Map Chart':
                st.write("### Map Chart ###")
                
                lat_cols = ['lat', 'latitude']
                lon_cols = ['lon', 'longitude']
                
                has_lat = any(col.lower() in df.columns for col in lat_cols)
                has_lon = any(col.lower() in df.columns for col in lon_cols)
                
                if has_lat and has_lon:
                    map_data = df.copy()
                    map_data.rename(columns={col: 'lat' for col in map_data.columns if col.lower() in lat_cols}, inplace=True)
                    map_data.rename(columns={col: 'lon' for col in map_data.columns if col.lower() in lon_cols}, inplace=True)
                    
                    map_data.dropna(subset=['lat', 'lon'], inplace=True)
                    
                    if not map_data.empty:
                        st.map(map_data)
                    else:
                        st.error("No valid geographical data found after cleaning.")
                else:
                    st.error("The uploaded file does not contain columns for latitude ('lat', 'latitude') and longitude ('lon', 'longitude').")
                    st.info("A Map Chart requires data with geographical coordinates.")
else:
    st.info('Please upload a CSV file to begin.')