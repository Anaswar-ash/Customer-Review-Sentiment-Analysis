# Customer Review Sentiment Analysis

This project automatically classifies written customer reviews into "Positive", "Negative", and "Neutral" categories. It uses a machine learning model trained on the Amazon Fine Food Reviews dataset.

## Features

-   **Data Preprocessing:** Cleans and prepares text data for modeling by removing HTML tags, non-alphanumeric characters, and stopwords.
-   **Model Training:** Trains a Logistic Regression model on a large dataset of customer reviews.
-   **Sentiment Prediction:** Provides an interactive command-line interface to classify new reviews.
-   **Evaluation:** Generates a classification report and a confusion matrix to evaluate model performance.
-   **Visualization:** Creates plots for sentiment distribution and the confusion matrix.

## Technologies & Libraries

-   **Python**
-   **Scikit-learn:** For machine learning (Logistic Regression, TF-IDF).
-   **Pandas:** For data manipulation.
-   **NLTK:** For natural language processing tasks (stopwords, lemmatization).
-   **Matplotlib & Seaborn:** For data visualization.
-   **KaggleHub:** To download the dataset.
-   **Joblib:** For saving and loading the trained model.

## Project Structure

```
Customer-Review-Sentiment-Analysis/
│
├── Sentiment_Project.py      # Main script for training, evaluation, and prediction
├── sentiment_model.pkl       # Saved trained model (generated after running the script)
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer (generated after running the script)
├── requirements.txt          # (To be created) List of dependencies
├── confusion_matrix.png      # Output plot of the confusion matrix
├── sentiment_distribution.png # Output plot of the sentiment distribution
└── README.md                 # This file
```

## Getting Started

### Prerequisites

-   Python 3.x
-   Kaggle account and API token set up for authentication. See the [Kaggle API documentation](https://www.kaggle.com/docs/api) for instructions on how to set up your `kaggle.json` file.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Anaswar-ash/Customer-Review-Sentiment-Analysis.git
    cd Customer-Review-Sentiment-Analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is not provided, but you can install the dependencies directly:
    ```bash
    pip install pandas scikit-learn nltk kagglehub matplotlib seaborn
    ```

### Usage

1.  **Run the script:**
    ```bash
    python Sentiment_Project.py
    ```
    The script will first download the necessary NLTK data (`stopwords` and `wordnet`) and the Amazon Fine Food Reviews dataset from Kaggle. Then, it will train the model, save the model files (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`), and generate the output plots.

2.  **Interactive Prediction:**
    After training, the script will enter an interactive mode. You can type a review and press Enter to see the predicted sentiment.
    ```
    Enter your review: This was a great purchase, I'm very happy!
    Predicted Sentiment: Positive

    Enter your review: The product was broken and unusable.
    Predicted Sentiment: Negative

    Enter your review: quit
    Exiting interactive session.
    ```
    Type `quit` to exit the interactive session.