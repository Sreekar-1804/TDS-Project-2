# Dataset Analysis

## Dataset Overview
This analysis is based on the provided dataset. It includes details such as:
- Column names and types: {"book_id": "int64", "goodreads_book_id": "int64", "best_book_id": "int64", "work_id": "int64", "books_count": "int64", "isbn": "object", "isbn13": "float64", "authors": "object", "original_publication_year": "float64", "original_title": "object", "title": "object", "language_code": "object", "average_rating": "float64", "ratings_count": "int64", "work_ratings_count": "int64", "work_text_reviews_count": "int64", "ratings_1": "int64", "ratings_2": "int64", "ratings_3": "int64", "ratings_4": "int64", "ratings_5": "int64", "image_url": "object", "small_image_url": "object"}
- Missing values summary: {"book_id": 0, "goodreads_book_id": 0, "best_book_id": 0, "work_id": 0, "books_count": 0, "isbn": 700, "isbn13": 585, "authors": 0, "original_publication_year": 21, "original_title": 590, "title": 0, "language_code": 1084, "average_rating": 0, "ratings_count": 0, "work_ratings_count": 0, "work_text_reviews_count": 0, "ratings_1": 0, "ratings_2": 0, "ratings_3": 0, "ratings_4": 0, "ratings_5": 0, "image_url": 0, "small_image_url": 0}

## Key Findings
Error obtaining insights from LLM: 

You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.

You can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. 

Alternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`

A detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742


## Visualizations
### Correlation Heatmap
![Correlation Heatmap](correlation_heatmap.png)
