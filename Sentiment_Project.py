# Install dependencies as needed:
# pip install pandas scikit-learn nltk kagglehub[pandas-datasets] matplotlib seaborn

import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Changed from MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import kagglehub
import os

# --- Download NLTK data if not present ---
def download_nltk_data():
    """Checks for and downloads required NLTK data packages."""
    try:
        nltk.data.find('corpora/stopwords')
        print("NLTK 'stopwords' resource found.")
    except LookupError:
        print("NLTK 'stopwords' not found. Downloading...")
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
        print("NLTK 'wordnet' resource found.")
    except LookupError:
        print("NLTK 'wordnet' not found. Downloading...")
        nltk.download('wordnet')

print("Checking for NLTK data...")
download_nltk_data()


# --- Step 1 & 2: Data Acquisition, Setup & Exploration ---

print("\nStep 1 & 2: Loading and exploring the data...")

try:
    # Step 1: Download the dataset. This function downloads and extracts the files,
    # returning the path to the directory where they are stored.
    print("Downloading dataset from Kaggle...")
    dataset_dir = kagglehub.dataset_download(
        "snap/amazon-fine-food-reviews"
    )
    print(f"Dataset downloaded and extracted to: {dataset_dir}")

    # Step 2: Construct the full path to the CSV file and load it into pandas.
    csv_file_path = os.path.join(dataset_dir, 'Reviews.csv')
    print(f"Loading data from: {csv_file_path}")
    
    # Read the file into a pandas DataFrame, specifying encoding and nrows
    # Limiting to first 250,000 rows for performance (Change as needed)
    df = pd.read_csv(csv_file_path, nrows=250000, encoding='latin-1')

    print("Dataset loaded successfully!")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nPlease ensure you have authenticated with Kaggle.")
    print("See documentation: https://github.com/Kaggle/kagglehub/blob/main/README.md")
    exit()


# Explore the data
print("\nFirst 5 records:")
print(df.head())

print("\nDataset Information:")
df.info()

# Drop rows with missing 'Text' or 'Score'
df.dropna(subset=['Text', 'Score'], inplace=True)
# Ensure score is an integer
df['Score'] = df['Score'].astype(int)

print(f"\nShape of the dataset after dropping nulls: {df.shape}")


# --- Step 3: Data Preprocessing and Cleaning ---

print("\nStep 3: Preprocessing and cleaning text data...")

# 1. Define Sentiment Labels
def map_sentiment(score):
    if score >= 4:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else: # score <= 2
        return 'Negative'

df['Sentiment'] = df['Score'].apply(map_sentiment)
print("\nSentiment distribution:")
print(df['Sentiment'].value_counts())


# 2. Clean the Review Text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply the cleaning function to the 'Text' column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

print("\nExample of original vs. cleaned text:")
print("Original:", df['Text'].iloc[5])
print("Cleaned:", df['Cleaned_Text'].iloc[5])


# --- Step 4: Feature Engineering (Text to Numbers) ---

print("\nStep 4: Performing TF-IDF Vectorization...")

# For simplicity, we'll focus on Positive vs. Negative for the model
df_subset = df[df['Sentiment'] != 'Neutral']

X = df_subset['Cleaned_Text']
y = df_subset['Sentiment']

# Initialize the TF-IDF Vectorizer
# max_features limits the vocabulary size to the top 5000 words
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the text data
X_tfidf = tfidf_vectorizer.fit_transform(X)

print(f"Shape of the TF-IDF matrix: {X_tfidf.shape}")


# --- Step 5: Building and Training the Model ---

print("\nStep 5: Splitting data and training the model...")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Choose and train the Logistic Regression model with balanced class weights
model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

print("Model training complete!")


# --- Step 6: Evaluating Your Model ---

print("\nStep 6: Evaluating the model performance...")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Check performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=['Positive', 'Negative'])
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)


# --- Step 7: Visualizing the Results ---

print("\nStep 7: Visualizing the results...")

# Plotting the sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
print("Saved sentiment distribution plot to 'sentiment_distribution.png'")

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix plot to 'confusion_matrix.png'")
# Use plt.show() if you are running this in an environment that supports GUI pop-ups.
# plt.show()


# --- Step 8: Saving the Model and Vectorizer ---

print("\nStep 8: Saving the model and vectorizer...")

# Save the trained model
joblib.dump(model, 'sentiment_model.pkl')
# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved to 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl'")


# --- Step 9: Predicting on New Data (Function Definition) ---

print("\nStep 9: Defining prediction function...")

# This function encapsulates the whole process for new predictions
def predict_sentiment(text, model_path='sentiment_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return "Error: Model or vectorizer not found. Please train and save them first."

    # Load the model and vectorizer
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Vectorize the cleaned text
    vectorized_text = loaded_vectorizer.transform([cleaned_text])
    
    # Make a prediction
    prediction = loaded_model.predict(vectorized_text)
    
    return prediction[0]

# Example usage with new reviews for initial script run
new_review_positive = "This is a fantastic product! I loved the taste and the quality was amazing. Highly recommend."
new_review_negative = "Terrible experience. The product was stale and tasted awful. I would not buy this again."

print(f"\nInitial Test Review: '{new_review_positive}'")
print(f"Predicted Sentiment: {predict_sentiment(new_review_positive)}")

print(f"\nInitial Test Review: '{new_review_negative}'")
print(f"Predicted Sentiment: {predict_sentiment(new_review_negative)}")


# --- Step 10: Interactive Prediction Loop ---
print("\n--- Starting Interactive Session ---")
print("Enter a review to analyze its sentiment. Type 'quit' to exit.")

while True:
    user_input = input("\nEnter your review: ")
    if user_input.lower() == 'quit':
        print("Exiting interactive session.")
        break
    
    prediction = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {prediction}")


print("\n--- Project Script Finished ---")

