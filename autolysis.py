# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
# ]
# ///
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai

# Try importing seaborn; if unavailable, skip related functionality.
try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False

# Perform data analysis: Summary statistics, missing values, and correlation matrix
def analyze_dataset(dataframe):
    """
    Analyze the given dataset to generate summary statistics, 
    identify missing values, and calculate correlation matrix for numeric columns.

    Args:
        dataframe (pd.DataFrame): The input dataset as a DataFrame.

    Returns:
        dict: A dictionary containing summary statistics, missing values, and correlation matrix.
    """
    print("Performing data analysis...")

    # Generate summary statistics
    summary = dataframe.describe()

    # Identify missing values
    missing = dataframe.isnull().sum()

    # Generate correlation matrix for numeric columns
    numeric_data = dataframe.select_dtypes(include=[np.number])
    correlation = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()

    print("Data analysis completed.")
    return {
        "summary": summary,
        "missing": missing,
        "correlation": correlation,
    }

# Detect outliers in numeric columns using the IQR method
def identify_outliers(dataframe):
    """
    Identify outliers in numeric columns using the IQR method.

    Args:
        dataframe (pd.DataFrame): The input dataset as a DataFrame.

    Returns:
        pd.Series: Count of outliers per numeric column.
    """
    print("Identifying outliers...")
    numeric_data = dataframe.select_dtypes(include=[np.number])
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers_count = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
    print("Outliers identification completed.")
    return outliers_count

# Generate visualizations: heatmaps, outliers plots, and distribution plots
def generate_visuals(analysis_results, outliers, dataframe, output_folder):
    """
    Create visualizations such as heatmaps, outliers plots, and distribution plots.

    Args:
        analysis_results (dict): Analysis results including correlation matrix.
        outliers (pd.Series): Count of outliers per numeric column.
        dataframe (pd.DataFrame): The input dataset.
        output_folder (str): Directory to save the output visualizations.

    Returns:
        dict: Paths to the saved visualizations.
    """
    print("Creating visualizations...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    visualization_paths = {}

    # Heatmap of the correlation matrix (if seaborn is available)
    correlation = analysis_results["correlation"]
    if seaborn_available and not correlation.empty:
        heatmap_path = os.path.join(output_folder, 'correlation_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.savefig(heatmap_path)
        plt.close()
        visualization_paths["heatmap"] = heatmap_path
    else:
        print("Seaborn unavailable or correlation matrix is empty. Skipping heatmap.")

    # Outliers plot
    if not outliers.empty and outliers.sum() > 0:
        outliers_path = os.path.join(output_folder, 'outliers.png')
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Count by Column')
        plt.xlabel('Columns')
        plt.ylabel('Outliers Count')
        plt.savefig(outliers_path)
        plt.close()
        visualization_paths["outliers"] = outliers_path

    # Distribution plot for the first numeric column
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if numeric_columns.any():
        distribution_path = os.path.join(output_folder, 'distribution.png')
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe[numeric_columns[0]].dropna(), bins=30, color='blue', alpha=0.7)
        plt.title(f'Distribution of {numeric_columns[0]}')
        plt.xlabel(numeric_columns[0])
        plt.ylabel('Frequency')
        plt.savefig(distribution_path)
        plt.close()
        visualization_paths["distribution"] = distribution_path

    print("Visualizations creation finished.")
    return visualization_paths

# Write a detailed README.md report for the dataset
def create_report(analysis_results, outliers, visualizations, output_folder):
    """
    Generate a Markdown report summarizing the analysis and integrating visualizations.

    Args:
        analysis_results (dict): Analysis results including summary, missing values, and correlation matrix.
        outliers (pd.Series): Count of outliers per numeric column.
        visualizations (dict): Paths to the generated visualizations.
        output_folder (str): Directory to save the README file.

    Returns:
        str: Path to the generated README file.
    """
    print("Generating README report...")
    readme_path = os.path.join(output_folder, 'README.md')
    try:
        with open(readme_path, 'w') as file:
            file.write("# Dataset Analysis Report\n\n")
            file.write("## Summary Statistics\n")
            file.write(analysis_results["summary"].to_markdown() + "\n\n")

            file.write("## Missing Values\n")
            file.write(analysis_results["missing"].to_markdown() + "\n\n")

            file.write("## Correlation Matrix\n")
            if "heatmap" in visualizations:
                file.write("![Correlation Matrix]({})\n\n".format(visualizations["heatmap"]))

            file.write("## Outliers\n")
            if "outliers" in visualizations:
                file.write("![Outliers Count]({})\n\n".format(visualizations["outliers"]))

            file.write("## Distribution Plot\n")
            if "distribution" in visualizations:
                file.write("![Distribution]({})\n\n".format(visualizations["distribution"]))

            print("README report generated successfully.")
    except Exception as e:
        print(f"Error creating README.md: {e}")
    return readme_path

# Main execution
def main():
    """
    Main function to orchestrate data analysis, visualization, and report generation.
    """
    parser = argparse.ArgumentParser(description="Analyze datasets and generate a report.")
    parser.add_argument('csv_file', help="Path to the CSV dataset")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.csv_file, encoding='ISO-8859-1')
        print("Dataset successfully loaded.")

        # Analyze, visualize, and generate outputs
        analysis_results = analyze_dataset(data)
        detected_outliers = identify_outliers(data)
        output_directory = "."
        visualizations = generate_visuals(analysis_results, detected_outliers, data, output_directory)
        create_report(analysis_results, detected_outliers, visualizations, output_directory)

        print("Analysis complete. Report saved.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
