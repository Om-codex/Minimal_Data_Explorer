# 📊 Minimal Data Explorer: Univariate, Bivariate, and Multivariate Analysis

## 🧭 Project Overview
**Minimal Data Explorer** is a fast, interactive web application built entirely in **Python** using **Streamlit** and **Altair**.  
It allows users to upload any CSV file and perform in-depth **Exploratory Data Analysis (EDA)** across three levels:
- **Univariate**
- **Bivariate**
- **Multivariate**

The app features a **clean, minimalistic dark-mode design** using custom CSS, delivering a professional and user-friendly interface.

---

## ✨ Features

### 🎨 Custom Dark Theme
- Injected CSS creates a modern, sleek dark-mode layout.
- Spacious UI designed for improved readability and aesthetics.

### 🔁 Multi-Stage App Flow
- Managed using `st.session_state` and `st.rerun()` for a seamless multi-step flow:
Upload → Preview → Analyze



### 🧾 Data Preview & Summary
- Displays:
- Raw data and shape (rows × columns)
- Null counts
- Summary statistics (`df.describe()`)

### 📈 Comprehensive Charting
- Built using **Altair** for interactive visualizations.
- Explicit axis naming and custom encodings for clarity.

---

## 🔍 Analysis Modules

### 🟩 Univariate Analysis
- **Numerical Columns:** Line/Area Charts vs. index  
- **Categorical Columns:** Bar Charts vs. frequency

### 🟨 Bivariate Analysis
- **Numerical vs. Numerical:** Line or Scatter Plots  
- **Categorical vs. Numerical:** Bar or Box Plots  
- **Hue Parameter:** Add a third categorical variable for color encoding (grouped bars or colored scatter points)

### 🟥 Multivariate Analysis
- **Bubble Chart:** Uses X, Y, Color (Hue), and Size encodings (4 variables)  
- **Heatmap:** Displays the mean intensity of a numerical column across two categorical dimensions

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-------------|----------|
| **Python** | Core programming language |
| **Streamlit** | Web application framework for fast UI prototyping |
| **Pandas** | Data loading, manipulation, and cleaning |
| **Altair** | Declarative statistical data visualization |
| **Custom CSS** | Frontend styling and theme customization |

---

## 🚀 Getting Started

### ✅ Prerequisites
Make sure you have **Python 3.8+** installed.

Check Streamlit version:
```bash
streamlit --version
📦 Installation
1. Clone the Repository (or create the main file)
Assuming your main file is named data_explorer_app.py.

2. Create requirements.txt
Create a file named requirements.txt in the same directory as your Python script and add the following:

nginx
Copy code
streamlit
pandas
altair
numpy
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Running the App
Run the application with:

bash
Copy code
streamlit run data_explorer_app.py
Once started, the app will automatically open in your default web browser:
👉 http://localhost:8501

💡 Future Enhancements
Correlation matrix visualization

Support for Excel files (.xlsx)

Downloadable EDA report summary

🧑‍💻 Author
Om-codex — Built with ❤️ using Python, Streamlit, and Altair.
