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

<img width="610" height="633" alt="image" src="https://github.com/user-attachments/assets/4bf988a6-ab57-40b2-876d-af9c68dc4c36" />

Imports: pandas, numpy, matplotlib, seaborn

Data: Load test features, labels, token probabilities (spam, ham, overall)

Joint Probabilities: Compute log probs for spam & ham (dot product + logs)

Prior: Apply spam prior (0.3116)

Prediction: Pick higher log-prob → classify email

Evaluation: 1685 correct, 39 wrong → 97.74% accuracy

Visualization: Scatter plot (spam vs ham log-probs) + decision boundary


# now we are gonna start making our model
Model Performance
Accuracy: 95.46%

Precision: 99.17% (few false positives)

Recall: 86.46% (catches most spam)

F1-Score: 92.38%

Technical Details
Dataset: 5,796 emails (70% training, 30% testing)

Features: 102,694 unique words after stop words removal

Algorithm: Multinomial Naive Bayes with Count Vectorization

Key Strength
High precision (99.17%) ensures legitimate emails are rarely misclassified as spam, making it suitable for real-world email filtering.



