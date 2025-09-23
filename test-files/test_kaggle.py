# kaggle_auth_test.py
# A small script to test authentication with Kaggle Hub.

import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

print("--- Kaggle Authentication Test ---")

# Method 1: Check for kaggle.json file
kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
print(f"\nChecking for credentials file at: {kaggle_json_path}")
if os.path.exists(kaggle_json_path):
    print("Found kaggle.json file.")
else:
    print("kaggle.json file NOT FOUND at the default location.")

# Method 2: Check for environment variables
print("\nChecking for KAGGLE_USERNAME and KAGGLE_KEY environment variables...")
username = os.environ.get('KAGGLE_USERNAME')
key = os.environ.get('KAGGLE_KEY')

if username and key:
    print("Found KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
    # Masking the key for security
    print(f"   - KAGGLE_USERNAME: {username}")
    print(f"   - KAGGLE_KEY: {'*' * len(key)}")
else:
    print("KAGGLE_USERNAME or KAGGLE_KEY environment variables are NOT SET.")


print("\nAttempting to download a small part of the dataset...")
print("This will fail if authentication is not set up correctly.")

try:
    # Attempt to load just the first 10 rows to test the connection
    df_test = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        handle="snap/amazon-fine-food-reviews",
        path="Reviews.csv",
        pandas_kwargs={'nrows': 10, 'encoding': 'latin-1'}
    )
    print("\n-----------------------------------------")
    print("✅ SUCCESS: Authentication successful!")
    print("Successfully loaded a sample of the dataset.")
    print("-----------------------------------------")
    print("\nFirst 5 rows of the test data:")
    print(df_test.head())

except Exception as e:
    print("\n-------------------------------------------------------------")
    print("❌ FAILURE: Could not authenticate or load data from Kaggle.")
    print(f"Error message: {e}")
    print("\nTroubleshooting steps:")
    print("1. Ensure you have a stable internet connection.")
    print("2. Double-check that your kaggle.json file is in the correct folder (~/.kaggle/kaggle.json).")
    print("3. OR, ensure your KAGGLE_USERNAME and KAGGLE_KEY environment variables are set correctly in your terminal session before running the script.")
    print("4. Make sure your API key has not expired. You can generate a new one from your Kaggle account settings.")
    print("-------------------------------------------------------------")
