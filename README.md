Used Car Price Analysis Dashboard 

Project Overview

This interactive dashboard analyzes the used car market to understand price dynamics, identify trends over time, and segment vehicles based on their features. The project was developed using Python, Streamlit, and Plotly as part of the CEN445 Introduction to Data Visualization course.

The dashboard cleans raw data, processes outliers, and presents 9 distinct visualizations to explore relationships between price, mileage, year, and technical specifications.

Team Members & Contributions

1. Ali Sait Öz

Role: Project Management & Hierarchical Analysis

Responsibilities:

Data Preprocessing: Implemented the load_data function to clean missing values, handle outliers (removing prices <$500 or >$500k), and standardize column names (e.g., converting 'make' to 'manufacturer').

Visualization Design (Tab 1):

Treemap: Market Share by Brand & Transmission.

Sunburst Chart: Hierarchical view of Brand > Fuel > Transmission.

Bar Chart: Top 10 Brands with Highest Average Price.

2. Berfin Öztürk

Role: Trend Analysis & Time-Series Visualizations

Responsibilities:

Time-Series Analysis (Tab 2): Focused on how car prices and features evolve over time (1990-2020).

Visualization Design:

Animated Scatter Plot: Evolution of Price vs. Odometer over the years (Interactive animation).

Parallel Coordinates: Multivariate analysis of Price, Year, and Odometer.

Line Chart: Average Price Change Trend over the years.

Documentation: Prepared the project documentation and README file.

3. Arda Murat Abay

Role: Machine Learning & Statistical Analysis

Responsibilities:

Advanced Analysis (Tab 3): Implemented Machine Learning models and statistical distributions.

Visualization Design:

K-Means Clustering: ML algorithm to segment cars into 3 categories (Economy, Mid-range, Luxury).

Density Heatmap: Correlation density between Price and Odometer.

Box Plot: Price Distribution analyzed by Fuel Type.

Reporting: Prepared the final project report.

Dataset Details

Source: Kaggle "Used Car Price Prediction Dataset" (or similar UK Used Car Data).

Scope: The analysis focuses on vehicles manufactured between 1990 and 2020.

Key Attributes:

price: Selling price of the vehicle.

year: Registration year (Filtered for 1990-2020).

manufacturer: Brand of the car (Standardized).

odometer: Mileage / Kilometers driven.

fuel: Fuel type (Petrol, Diesel, etc.).

transmission: Gearbox type (Manual, Automatic, Semi-Auto).

Visualizations Included

The dashboard is divided into three analytical sections:

Hierarchical Analysis: Focuses on the market structure.

Treemap, Sunburst Chart, Bar Chart.

Trend Analysis: Focuses on temporal changes.

Animated Scatter Plot, Parallel Coordinates, Line Chart.

ML & Stats: Focuses on segmentation and distribution.

K-Means Clustering (Scatter), Density Heatmap, Box Plot.

How to Run Locally

Clone the repository:

git clone (https://github.com/berfinozturk/CEN445-Car-Analysis.git)
cd CEN445-Car-Analysis


Install required libraries:

pip install -r requirements.txt


Run the Streamlit App:

streamlit run app.py


Open in Browser:
The app should automatically open at http://localhost:8501.

File Structure

app.py: Main application code containing all visualizations and logic.

vehicles.csv: Cleaned dataset used for analysis.

requirements.txt: List of Python dependencies.

README.md: Project documentation.
