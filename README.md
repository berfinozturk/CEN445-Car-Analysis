Used Car Price Analysis Dashboard 

Project Overview

This interactive dashboard analyzes the used car market to understand price dynamics, identify trends over time, and segment vehicles based on their features. The project was developed using Python, Streamlit, and Plotly as part of the CEN445 Introduction to Data Visualization course.

The dashboard cleans raw data, processes outliers, filters for the relevant timeframe (1990-2020), and presents 9 distinct visualizations to explore relationships between price, mileage, year, and technical specifications.

Team Members & Contributions


1. Ali Sait Öz

Role: Project Management & Hierarchical Analysis

Key Contributions:

Data Preprocessing: Implemented the load_data function.

Standardized column names (e.g., converting 'make', 'brand' to 'manufacturer').

Handled missing values (dropna) and data type conversions.

Removed outliers (Prices <$500 or >$500k).

Implemented the master year filter (1990-2020).

Visualization Design (Tab 1: Hierarchical Analysis):

Treemap: Market Share by Brand & Transmission.

Sunburst Chart: Hierarchical view of Brand > Fuel > Transmission.

Bar Chart: Top 10 Brands with Highest Average Price.


2. Berfin Öztürk

Role: Trend Analysis & Time-Series Visualizations

Key Contributions:

Time-Series Logic: Designed the logic to analyze how car attributes evolve over the 30-year period.

Visualization Design (Tab 2: Trend Analysis):

Animated Scatter Plot: Evolution of Price vs. Mileage over the years (Interactive animation frame).

Parallel Coordinates: Multivariate analysis of Price, Year, and Odometer.

Line Chart: Average Price Change Trend over the years.

Documentation: Prepared the GitHub repository structure and README file.


3. Arda Murat Abay

Role: Machine Learning & Statistical Analysis

Key Contributions:

Advanced Analysis Logic: Implemented the K-Means algorithm and statistical groupings.

Visualization Design (Tab 3: ML & Stats):

K-Means Clustering: ML algorithm to segment cars into 3 categories (Economy, Mid-range, Luxury).

Density Heatmap: Correlation density between Price and Odometer.

Box Plot: Price Distribution analyzed by Fuel Type.

Reporting: Authored the final project report.


Dataset Details

Source: Kaggle "Used Car Price Prediction Dataset" (or similar UK Used Car Data).

Timeframe: The analysis focuses strictly on vehicles manufactured between 1990 and 2020 to ensure data relevance.


Key Attributes:

price: Selling price of the vehicle.

year: Registration year (Filtered for 1990-2020 range).

manufacturer: Brand of the car (Standardized).

odometer: Mileage / Kilometers driven.

fuel: Fuel type (Petrol, Diesel, etc.).

transmission: Gearbox type (Manual, Automatic, Semi-Auto).


Visualizations Included

The dashboard is organized into three analytical tabs:

Hierarchical Analysis: Focuses on the market structure and brand dominance.

Trend Analysis: Focuses on temporal changes and multivariate relationships.

ML & Stats: Focuses on automated segmentation and statistical distributions.


How to Run Locally

Clone the repository:

git clone (https://github.com/berfinozturk/CEN445-Car-Analysis.git)
cd CEN445-Car-Analysis


Install required libraries:
Ensure you have Python installed, then run:

pip install -r requirements.txt


Run the Streamlit App:

streamlit run app.py



File Structure

app.py: Main application code containing all visualizations and logic.

vehicles.csv: Cleaned dataset used for analysis.

requirements.txt: List of Python dependencies (streamlit, pandas, plotly, scikit-learn).

README.md: Project documentation and contribution details.
