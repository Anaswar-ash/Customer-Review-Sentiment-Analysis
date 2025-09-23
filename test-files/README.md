Kaggle Authentication Test Script
This directory contains a small, standalone Python script (kaggle_auth_test.py) designed to diagnose and verify authentication with the Kaggle API.

Purpose
The main sentiment analysis project relies on downloading a dataset directly from Kaggle. If the main script fails with an authentication error, this utility can be run to quickly determine the cause of the problem without executing the entire data processing pipeline.

It is a diagnostic tool to ensure your environment is correctly configured to communicate with Kaggle before running the main analysis.

How It Works
The script performs the following steps:

Checks for kaggle.json: It first looks for the kaggle.json credential file in the default user directory (~/.kaggle/kaggle.json).

Checks for Environment Variables: It then checks if the KAGGLE_USERNAME and KAGGLE_KEY environment variables are set in the current session.

Attempts Download: Finally, it tries to connect to the Kaggle API and download the first 10 rows of the "Amazon Fine Food Reviews" dataset.

Provides Feedback: The script will print a clear success or failure message along with the status of your credential setup.

How to Run
Navigate to the directory containing the script and execute it using Python:

python kaggle_auth_test.py

Interpreting the Output
Success Scenario
If your authentication is configured correctly, you will see a success message and the first few rows of the test DataFrame:

✅ SUCCESS: Authentication successful!
Successfully loaded a sample of the dataset.
-----------------------------------------

First 5 rows of the test data:
   Id  ...                                               Text
0   1  ...  I have bought several of the Vitality canned d...
1   2  ...  Product arrived labeled as Jumbo Salted Peanut...
2   3  ...  This is a confection that has been around a fe...
3   4  ...  If you are looking for the secret ingredient i...
4   5  ...  Great taffy at a great price.  There was a wid...

Failure Scenario
If the script cannot authenticate, it will print a failure message and provide common troubleshooting steps to help you resolve the issue:

❌ FAILURE: Could not authenticate or load data from Kaggle.
Error message: <Specific error from the library>

Troubleshooting steps:
1. Ensure you have a stable internet connection.
2. Double-check that your kaggle.json file is in the correct folder (~/.kaggle/kaggle.json).
3. OR, ensure your KAGGLE_USERNAME and KAGGLE_KEY environment variables are set correctly...
4. Make sure your API key has not expired...
