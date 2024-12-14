import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai

# Perform data analysis: Summary statistics, missing values, and correlation matrix
def analyze_dataset(dataframe):
    """
    Analyze the given dataset to generate summary statistics, 
    identify missing values, and calculate correlation matrix for numeric columns.

    Args:
        dataframe (pd.DataFrame): The input dataset as a DataFrame.

    Returns:
        tuple: Summary statistics, missing values, and correlation matrix.
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
    return summary, missing, correlation

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
def generate_visuals(correlation, outliers, dataframe, output_folder):
    """
    Create visualizations such as heatmaps, outliers plots, and distribution plots.

    Args:
        correlation (pd.DataFrame): Correlation matrix of numeric columns.
        outliers (pd.Series): Count of outliers per numeric column.
        dataframe (pd.DataFrame): The input dataset.
        output_folder (str): Directory to save the output visualizations.

    Returns:
        tuple: Paths to the saved heatmap, outliers plot, and distribution plot.
    """
    print("Creating visualizations...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Heatmap of the correlation matrix
    heatmap_path = os.path.join(output_folder, 'correlation_matrix.png')
    if not correlation.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.savefig(heatmap_path)
        plt.close()

    # Outliers plot
    outliers_path = None
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Count by Column')
        plt.xlabel('Columns')
        plt.ylabel('Outliers Count')
        outliers_path = os.path.join(output_folder, 'outliers.png')
        plt.savefig(outliers_path)
        plt.close()

    # Distribution plot for the first numeric column
    distribution_path = None
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if numeric_columns.any():
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe[numeric_columns[0]], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {numeric_columns[0]}')
        plt.xlabel(numeric_columns[0])
        plt.ylabel('Frequency')
        distribution_path = os.path.join(output_folder, 'distribution.png')
        plt.savefig(distribution_path)
        plt.close()

    print("Visualizations creation finished.")
    return heatmap_path, outliers_path, distribution_path

# Write a detailed README.md report for the dataset
def create_report(summary, missing, correlation, outliers, output_folder):
    """
    Generate a Markdown report summarizing the analysis and integrating visualizations.

    Args:
        summary (pd.DataFrame): Summary statistics of the dataset.
        missing (pd.Series): Count of missing values per column.
        correlation (pd.DataFrame): Correlation matrix of numeric columns.
        outliers (pd.Series): Count of outliers per numeric column.
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
            file.write(summary.to_markdown() + "\n\n")

            file.write("## Missing Values\n")
            file.write(missing.to_markdown() + "\n\n")

            file.write("## Correlation Matrix\n")
            if not correlation.empty:
                file.write("![Correlation Matrix](correlation_matrix.png)\n\n")

            file.write("## Outliers\n")
            if not outliers.empty:
                file.write("![Outliers Count](outliers.png)\n\n")

            file.write("## Distribution Plot\n")
            file.write("![Distribution](distribution.png)\n\n")

            file.write("## Analysis Highlights\n")
            file.write("- Key statistical insights and visual patterns highlighted.\n")
            file.write("- Data relationships and significant features identified through correlation.\n")

            print("README report generated successfully.")
    except Exception as e:
        print(f"Error creating README.md: {e}")
    return readme_path

# Communicate with OpenAI API via proxy to generate a narrative
def generate_story(prompt, analysis_context):
    """
    Use OpenAI API to generate a narrative based on the analysis.

    Args:
        prompt (str): The input prompt for the narrative.
        analysis_context (str): Contextual information for the prompt.

    Returns:
        str: Generated narrative or error message.
    """
    try:
        print("Connecting to OpenAI API for narrative generation...")
        token = os.getenv("AIPROXY_TOKEN")
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\nContext: {analysis_context}"}
            ],
            "temperature": 0.7
        }

        response = requests.post(api_url, headers=headers, json=request_data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
            return "Failed to generate story."
    except Exception as e:
        print(f"Error communicating with API: {e}")
        return "Failed to generate story."

# Main execution
def main():
    """
    Main function to orchestrate data analysis, visualization, report generation, and narrative creation.
    """
    parser = argparse.ArgumentParser(description="Analyze datasets and generate a report.")
    parser.add_argument('csv_file', help="Path to the CSV dataset")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.csv_file, encoding='ISO-8859-1')
        print("Dataset successfully loaded.")

        # Analyze, visualize, and generate outputs
        stats, missing_vals, corr_matrix = analyze_dataset(data)
        detected_outliers = identify_outliers(data)
        output_directory = "."
        visuals = generate_visuals(corr_matrix, detected_outliers, data, output_directory)
        report_path = create_report(stats, missing_vals, corr_matrix, detected_outliers, output_directory)

        # Story generation
        narrative = generate_story(
            "Generate a story based on the dataset analysis.",
            f"Summary: {stats}\nMissing: {missing_vals}\nCorrelation: {corr_matrix}\nOutliers: {detected_outliers}"
        )

        if report_path:
            with open(report_path, 'a') as f:
                f.write("\n## Story\n" + narrative + "\n")

        print(f"Analysis complete. Report saved at: {report_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
