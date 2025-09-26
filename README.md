# Email-Spam-Classifier
📁 BayesClassifier.ipynb - Data Preparation
Imports libraries (pandas, nltk, matplotlib)

Loads email data from spam/ham directories

Cleans data - removes empty messages, adds IDs

EDA - spam/ham distribution pie chart

Text preprocessing - lowercase, tokenize, remove stopwords, stemming

Word frequency analysis for spam vs ham emails<img width="486" height="394" alt="image" src="https://github.com/user-attachments/assets/5e075b49-1848-4dde-b320-fcc79f78d98b" />

word cloud
<img width="486" height="500" alt="image" src="https://github.com/user-attachments/assets/48b154c0-3671-4901-a472-bfd563492c99" />

# Naive bayes training
Setup - Import libraries, set constants (vocab size: 2500), define file paths

Data Loading - Read sparse training/test data (doc_id, word_id, label, frequency)

Matrix Conversion - Convert sparse data to dense DataFrame (emails × token counts)

Model Training - Calculate:

P(Spam) and P(Ham)

Word counts per category

P(Token|Spam) and P(Token|Ham) with Laplace smoothing

Export - Save probability vectors and test features for classification

Output: Trained probability parameters and prepared test set for spam classification.

# testing the model
