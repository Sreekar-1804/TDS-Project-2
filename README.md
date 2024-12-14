# TDS-Project-2

TDS Project 2 :

This project automates the analysis of datasets by generating insights, visualizations, and a narrative summary using OpenAI's GPT-4o-Mini model. It supports multiple datasets, making it flexible for various use cases.

Features
Automated Data Analysis:

Generates summary statistics and missing value analysis.
Extracts key insights using GPT-4o-Mini.
Dynamic Visualizations:

Creates correlation heatmaps for numeric data.
Generates distributions of numerical columns.
Narrative Storytelling:

Produces a README.md for each dataset with:
Dataset insights.
Summary of findings.
Visualizations embedded.
How It Works
Input:

The script accepts one or more CSV files as input datasets.
Processing:

Cleans the data (e.g., removes missing values).
Performs statistical analysis and generates insights.
Dynamically creates visualizations based on the dataset.
Output:

A README.md file summarizing the dataset analysis.
PNG visualizations stored in the specified output directory.
Project Structure
bash
Copy code
.
├── autolysis.py       # Main Python script for dataset analysis
├── goodreads/         # Outputs for the Goodreads dataset
│   ├── README.md      # Summary of the analysis for Goodreads
│   ├── *.png          # Visualizations for Goodreads
├── happiness/         # Outputs for the Happiness dataset
│   ├── README.md      # Summary of the analysis for Happiness
│   ├── *.png          # Visualizations for Happiness
├── media/             # Outputs for the Media dataset
│   ├── README.md      # Summary of the analysis for Media
│   ├── *.png          # Visualizations for Media
└── LICENSE            # MIT License for the project
Setup Instructions
1. Clone the Repository
bash
Copy code
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
2. Install Dependencies
Ensure Python 3.11 or higher is installed, then run:

bash
Copy code
pip install pandas matplotlib openai python-dotenv
3. Set API Key
Add your OpenAI API key directly in autolysis.py:

python
Copy code
api_key = "your_actual_api_key"
4. Add Your Datasets
Place your datasets in the project directory or specify their full paths in autolysis.py:

python
Copy code
datasets = [
    ("path/to/your/dataset1.csv", "path/to/your/output1"),
    ("path/to/your/dataset2.csv", "path/to/your/output2"),
]
Usage
Run the Script
Execute the script in your terminal:

bash
Copy code
python autolysis.py
Outputs
For each dataset:
A README.md file with a detailed analysis.
PNG files for visualizations.
Example Outputs
1. README.md (Goodreads Dataset)
markdown
Copy code
# Dataset Analysis

## Dataset Overview
This analysis is based on the Goodreads dataset. It includes details such as:
- Column names and types: {...}
- Missing values summary: {...}

## Key Findings
- Fiction books have the highest average ratings.
- The dataset shows strong correlations between review counts and ratings.

## Visualizations
![Correlation Heatmap](correlation_heatmap.png)
2. Visualization Samples
Correlation Heatmap:
Displays the relationships between numeric columns.
Distribution Plot:
Highlights the spread of values for a specific numeric column.
Contribution Guidelines
Fork the repository and create a new branch for your changes.
Make improvements or add features.
Submit a pull request for review.
License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

Acknowledgements
OpenAI for GPT-4o-Mini.
Contributors to pandas, matplotlib, and seaborn librarie
