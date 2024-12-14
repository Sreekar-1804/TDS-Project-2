import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import openai

# Placeholder: Replace this with your actual API key
api_key = "your_actual_api_key"  # Enter your OpenAI API key here
openai.api_key = api_key

def analyze_dataset(file_path, output_dir):
    # Load the dataset
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    # Data Cleaning
    missing_summary = data.isnull().sum().to_dict()
    data = data.dropna()  # Basic cleaning, can be extended

    # Perform basic analysis
    summary_stats = data.describe(include='all').to_dict()
    column_info = {col: str(data[col].dtype) for col in data.columns}

    # Use GPT-4o-Mini for additional insights
    try:
        prompt = (
            f"You are an AI assistant analyzing a dataset. Here are the column names and their types: {json.dumps(column_info)}.\n"
            f"Here are some summary statistics: {json.dumps(summary_stats)}.\n"
            f"What are the key insights, trends, or interesting aspects of this dataset?\n"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4.0-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        insights = response['choices'][0]['message']['content']
    except Exception as e:
        insights = f"Error obtaining insights from LLM: {e}"

    # Visualizations
    os.makedirs(output_dir, exist_ok=True)

    # Select only numeric columns for correlation heatmap
    numeric_data = data.select_dtypes(include=['number'])
    if numeric_data.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = numeric_data.corr()
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        plt.title('Correlation Heatmap')
        heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()
    else:
        heatmap_path = "No numeric columns available for correlation heatmap."

    # Generate README.md
    readme_content = f"""# Dataset Analysis

## Dataset Overview
This analysis is based on the provided dataset. It includes details such as:
- Column names and types: {json.dumps(column_info)}
- Missing values summary: {json.dumps(missing_summary)}

## Key Findings
{insights}

## Visualizations
"""
    if os.path.exists(heatmap_path):
        readme_content += f"### Correlation Heatmap\n![Correlation Heatmap](correlation_heatmap.png)\n"

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as readme_file:
        readme_file.write(readme_content)

    print(f"Analysis complete. Outputs saved to: {output_dir}")

if __name__ == "__main__":
    # Placeholder: Define datasets and output directories
    datasets = [
        # Example placeholders - Replace these with actual paths
        ("path/to/your/goodreads.csv", "path/to/your/goodreads_output"),
        ("path/to/your/happiness.csv", "path/to/your/happiness_output"),
        ("path/to/your/media.csv", "path/to/your/media_output"),
    ]

    # Process each dataset
    for file_path, output_dir in datasets:
        print(f"Processing dataset: {file_path}")
        analyze_dataset(file_path, output_dir)
