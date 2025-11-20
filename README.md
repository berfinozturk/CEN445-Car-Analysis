# CEN445-Car-Analysis

# Used Car Price Analysis Dashboard 🚗

## Project Description
This interactive dashboard analyzes the used car market to understand price dynamics, identify trends over time, and segment vehicles based on their features. Developed as part of the **CEN445 Introduction to Data Visualization** course.

## Team Members & Contributions
* **Ali Sait Öz:** Project Management, Data Preprocessing (cleaning outliers/missing values), Hierarchical Visualizations (Treemap, Sunburst), and Dashboard Layout.
* **Berfin Öztürk:** Trend Analysis, Time-Series Visualizations (Animated Scatter, Line Chart), Multivariate Analysis (Parallel Coordinates), and Documentation.
* **Arda Murat Abay:** Machine Learning Implementation (K-Means Clustering), Statistical Analysis (Heatmap, Box Plot), and Project Report.

## Dataset
* **Source:** Kaggle - Used Car Price Prediction Dataset
* **Content:** The dataset includes attributes such as Price, Year, Manufacturer, Model, Odometer, Fuel Type, and Transmission for over 2,000 vehicles.
* **Preprocessing:** Missing values were removed, and outliers (e.g., unrealistic prices or mileage) were filtered out to ensure data quality.

## Visualizations
The dashboard features 9 visualizations (6 Advanced, 3 Basic):
1.  **Treemap:** Market share by Brand/Model.
2.  **Sunburst Chart:** Hierarchy of Brand > Fuel > Transmission.
3.  **Animated Scatter Plot:** Evolution of Price vs. Mileage over years.
4.  **Parallel Coordinates:** Relationships between Price, Year, and Odometer.
5.  **K-Means Clustering:** Segmentation of cars into Economy, Mid-range, and Luxury groups.
6.  **Density Heatmap:** Correlation density between Price and Odometer.
7.  **Basic Charts:** Bar, Line, and Box plots for general insights.

## How to Run Locally
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
