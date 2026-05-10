# MBTI Personality Prediction Project

## Project Overview
This project aims to classify Myers-Briggs Type Indicator (MBTI) personality types based on text posts. We leverage natural language processing (NLP) techniques to preprocess text data and machine learning models to predict the personality type.

## Dataset
The dataset used for this project contains text posts and their corresponding MBTI personality types.

## Methodology

### 1. Data Loading and Initial Exploration
- Loaded the dataset from a CSV file.
- Performed initial data checks: shape, missing values, duplicate rows, and sample data viewing.
- Visualized the distribution of personality types to understand class balance.

### 2. Text Preprocessing
A robust text preprocessing pipeline was applied to the 'posts' data:
- **Lowercasing**: Converted all text to lowercase.
- **Punctuation Removal**: Eliminated punctuation marks.
- **Extra Space Removal**: Consolidated multiple spaces into single spaces.
- **Link Removal**: Removed URLs (http, https, www).
- **Stopword Removal**: Filtered out common English stopwords (e.g., 'the', 'is', 'a').
- **Lemmatization**: Reduced words to their base form (e.g., 'running' to 'run').

### 3. Feature Extraction
- **TF-IDF Vectorization**: Transformed the preprocessed text into numerical features using `TfidfVectorizer` with a maximum of 5000 features.

### 4. Target Encoding
- The categorical 'type' (MBTI personality type) was converted into numerical labels using `LabelEncoder`.

### 5. Data Splitting
- The data was split into training and testing sets (80% train, 20% test) to evaluate model performance on unseen data.

### 6. Model Training and Evaluation
Initially, two models were trained and evaluated:
- **LinearSVC**: A Support Vector Classifier (SVC) with a linear kernel.
- **Logistic Regression**: A linear model for classification.

Both models were evaluated using `accuracy_score` and `classification_report`, which provided precision, recall, and F1-score for each personality type. Visual comparisons of these metrics were generated.

### 7. Hyperparameter Tuning with GridSearchCV
To optimize model performance, GridSearchCV was applied to both models:
- **LinearSVC**: Tuned the `C` (regularization parameter) and `loss` parameters.
- **Logistic Regression**: Tuned `C` and `penalty` (L1/L2 regularization) with the `liblinear` solver.

After tuning, the Logistic Regression model achieved a slightly higher accuracy of **0.66** on the test set, making it the better-performing model.

### 8. Model Persistence
The best-performing model (Tuned Logistic Regression), the TF-IDF vectorizer, and the LabelEncoder were saved using `joblib` for future use in predictions.

### 9. Gradio Application
A Gradio application (`app_tuned_lr`) was developed to provide an interactive interface for predicting MBTI personality types. Users can input text, and the application will use the saved tuned Logistic Regression model to predict the personality type.

## How to Use the Gradio Application
1. Run the notebook cells sequentially up to the Gradio application cell.
2. Execute the Gradio application cell.
3. A public URL will be provided in the output, allowing you to access the interactive predictor in your web browser.
4. Enter any text into the provided textbox, and the application will return the predicted MBTI personality type.
