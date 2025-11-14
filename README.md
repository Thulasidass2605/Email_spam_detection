Email_spam_detection using Machine Learning

This project builds a machine learning model to detect whether an email is Spam or Ham (not spam) using TF-IDF text features and multiple classifiers (Naive Bayes, Logistic Regression, and Random Forest).

Project Overview

Email spam detection is one of the most common and useful NLP classification problems.
In this project, we:

Clean and preprocess raw email text
Extract features using TF-IDF Vectorization
Train multiple models (Naive Bayes, Logistic Regression, and Random Forest)
Evaluate and compare models using standard metrics
Save the best-performing model using joblib

Project Structure

├── README.md        #project documentation
├── spam_detection.py                # # Main training & evaluation script
└── emails.csv               # Dataset file (not included)

Key Features

 Text normalization (lowercasing, removing URLs, special chars, etc.)
 Model comparison between different ML classifiers
 Evaluation with Accuracy, Precision, Recall, F1, ROC-AUC
 Model persistence using joblib
 Ready for prediction on new unseen emails

Dataset

The dataset used is emails.csv, which should contain:

Column Name	Description
text	The email body/content
spam	Label (1 = Spam, 0 = Ham)

Example:

text,spam
"Congratulations, you won a free ticket!",1
"Let's catch up for lunch tomorrow",0

 Libraries Used
  numpy
  pandas
  scikit-learn
  joblib
  re
  html
  
Run the script
python spam_detection.py

Model Training & Evaluation

The code trains three models:

Multinomial Naive Bayes
Logistic Regression
Random Forest

Each model is evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC AUC

Example Output
>>> Training: nb
Accuracy: 0.9762
Precision: 0.9723
Recall: 0.9794
F1: 0.9758
ROC AUC: 0.9921

Best model: nb  metrics: {'accuracy': 0.9762, 'precision': 0.9723, 'recall': 0.9794, 'f1': 0.9758, 'roc_auc': 0.9921}
Saved model to: best_spam_model_nb.joblib

Text: Congratulations! You've won a $1000 gift card. Click here to claim now.
Predicted label: spam, probability: 0.987

License
This project is licensed under the MIT License – feel free to use and modify.








