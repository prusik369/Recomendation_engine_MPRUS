# Recommendation Engine

### This project is a product recommendation engine built on historical ratings data. It provides:
 - Top-N best-rated products over a chosen time frame.
 - Top-N personalized recommendations for existing users using a pre-trained model.

### The engine uses a carefully selected model (chosen from multiple candidates) for recommendations.
### All paths to ratings data and pre-trained model are set to defaults, so no arguments are required if files are in the expected locations.






## File structure


| File                  | Purpose                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `cli_app.py`          | Main CLI application. Computes top products and generates recommendations for users.         |
| `Utils_cleaning.py`   | Functions for cleaning and preparing the dataset (e.g., timestamps, ratings).                |
| `Utils_app.py`        | Core application functions: weighted ratings, filtering by time, generating recommendations. |
| `Utils_eda.py`        | Exploratory data analysis helper functions.                                                  |
| `model_comparison.py` | Used for model selection and comparison (SVD selected from multiple candidates).             |
| `requirements.txt`    | Python dependencies (tested on latest Python version).                                       |
| `EDA.ipynb`           | Notebook for initial data exploration.                                                       |
| `playground.ipynb`    | Notebook for testing functions and recommendations interactively.                            |

Note: Default paths for ratings data (data/ratings.csv) and model (best_model/best_model.pkl) are correct out-of-the-box; no changes are required.



# CLI Application Usage

### The CLI application (cli_app.py) allows you to:
 - Compute weighted ratings for products (filtered by time range if needed).
 - Generate top-N recommendations for selected users using the pre-trained model.

## App parameters


| Parameter          | Type        | Default                     | Applies to           | Description                                                                                                                                                                                                                                                                               |
| ------------------ | ----------- | --------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--file`           | str         | `data/ratings.csv`          | Both                 | Path to ratings CSV. Default path works without modification.                                                                                                                                                                                                                             |
| `--top`            | int         | `20`                        | Both                 | Number of top products to display in weighted ranking **and** to generate top-N recommendations for users.                                                                                                                                                                                |
| `--time`           | str         | `all`                       | Weighted rating      | Time filter for weighted ranking: <br>• `all` → all ratings (default) <br>• `last_month` → last 30 days <br>• `last_year` → last 365 days <br>• `custom` → prompts for **start date** and **end date** in `YYYY-MM-DD` format. <br>**Note:** Does not affect model-based recommendations. |
| `--n_random_users` | int         | `3`                         | User recommendations | Number of random users to generate recommendations for if no `--user_ids` provided.                                                                                                                                                                                                       |
| `--user_ids`       | list of int | `None`                      | User recommendations | List of specific user IDs to generate recommendations for. If omitted, random users are chosen.                                                                                                                                                                                           |
| `--model`          | str         | `best_model/best_model.pkl` | User recommendations | Path to the saved model. Default path works without modification.                                                                                                                                                                                                                         |


Important: Model-based recommendations ignore the time range; they always use the full rating dataset. Time ranges are only for calculating top products over a period.

## Examples of CLI Usage

1. Default execution – top 20 products weighted ranking + recommendations for 3 random users:
  - python cli_app.py

2. Top 10 products and recommendations for 5 random users:
  -  python cli_app.py --top 10 --n_random_users 5

3. Custom time range – affects only weighted ranking (prompts for start/end dates):
  -  python cli_app.py --time custom
      Input:
         Enter start date (YYYY-MM-DD): 2024-01-01
         Enter end date (YYYY-MM-DD): 2024-06-30
     
4. Generate recommendations for specific users:
  - python cli_app.py --user_ids 1 2 5 --top 15

5. Combine options:
  -  python cli_app.py --top 10 --time last_year --user_ids 1 2



## Notes

### Weighted ranking is influenced by the selected time range.
### Recommendations are generated using the chosen pre-trained model, which was selected from multiple candidates after careful evaluation.
### Custom time range allows you to examine top products for a specific period.

### Default paths for ratings data and model work without modification.
--top affects both weighted ranking and recommendations.

--time affects only weighted ranking.

--user_ids and --n_random_users affect only recommendations.


# Workflow 



<img width="414" height="770" alt="image" src="https://github.com/user-attachments/assets/a461bce8-455c-4f2e-a3e8-d85a6edef3b4" />



