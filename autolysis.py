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
        dict: A dictionary containing summary statistics, missing values, skewness, kurtosis, and correlation matrix.
    """
    print("Performing data analysis...")

    # Generate summary statistics
    summary = dataframe.describe()

    # Identify missing values
    missing = dataframe.isnull().sum()

    # Skewness and Kurtosis
    skewness = dataframe.skew(numeric_only=True)
    kurtosis = dataframe.kurt(numeric_only=True)

    # Generate correlation matrix for numeric columns
    numeric_data = dataframe.select_dtypes(include=[np.number])
    correlation = numeric_data.corr() if not numeric_data.empty else pd.DataFrame()

    print("Data analysis completed.")
    return {
        "summary": summary,
        "missing": missing,
        "skewness": skewness,
        "kurtosis": kurtosis,
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

    # Heatmap of the correlation matrix
    heatmap_path = os.path.join(output_folder, 'correlation_matrix.png')
    correlation = analysis_results["correlation"]
    if not correlation.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.savefig(heatmap_path)
        plt.close()
        visualization_paths["heatmap"] = heatmap_path

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

    # Pairplot for numeric data relationships
    pairplot_path = os.path.join(output_folder, 'pairplot.png')
    sns.pairplot(dataframe.select_dtypes(include=[np.number]).dropna())
    plt.savefig(pairplot_path)
    plt.close()
    visualization_paths["pairplot"] = pairplot_path

    print("Visualizations creation finished.")
    return visualization_paths

# Write a detailed README.md report for the dataset
def create_report(analysis_results, outliers, visualizations, output_folder):
    """
    Generate a Markdown report summarizing the analysis and integrating visualizations.

    Args:
        analysis_results (dict): Analysis results including summary, missing values, skewness, kurtosis, and correlation matrix.
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

            file.write("## Skewness and Kurtosis\n")
            file.write("### Skewness\n")
            file.write(analysis_results["skewness"].to_markdown() + "\n\n")
            file.write("### Kurtosis\n")
            file.write(analysis_results["kurtosis"].to_markdown() + "\n\n")

            file.write("## Correlation Matrix\n")
            if "heatmap" in visualizations:
                file.write("![Correlation Matrix]({})\n\n".format(visualizations["heatmap"]))

            file.write("## Outliers\n")
            if "outliers" in visualizations:
                file.write("![Outliers Count]({})\n\n".format(visualizations["outliers"]))

            file.write("## Pairplot\n")
            if "pairplot" in visualizations:
                file.write("![Pairplot]({})\n\n".format(visualizations["pairplot"]))

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
        analysis_results = analyze_dataset(data)
        detected_outliers = identify_outliers(data)
        output_directory = "."
        visualizations = generate_visuals(analysis_results, detected_outliers, data, output_directory)
        report_path = create_report(analysis_results, detected_outliers, visualizations, output_directory)

        # Story generation
        narrative = generate_story(
            "Generate a story based on the dataset analysis.",
            f"Summary: {analysis_results['summary']}\nMissing: {analysis_results['missing']}\nSkewness: {analysis_results['skewness']}\nKurtosis: {analysis_results['kurtosis']}\nCorrelation: {analysis_results['correlation']}\nOutliers: {detected_outliers}"
        )

        if report_path:
            with open(report_path, 'a') as f:
                f.write("\n## Story\n" + narrative + "\n")

        print(f"Analysis complete. Report saved at: {report_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
