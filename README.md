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

<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#2c3e50">
    Spam Email Classification Model Architecture
  </text>
  
  <!-- Input Data -->
  <rect x="50" y="80" width="120" height="80" rx="10" fill="#3498db" stroke="#2980b9" stroke-width="2"/>
  <text x="110" y="115" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Email Text Data</text>
  <text x="110" y="135" text-anchor="middle" font-size="10" fill="white">5,796 samples</text>
  
  <!-- Count Vectorization -->
  <rect x="220" y="80" width="120" height="80" rx="10" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
  <text x="280" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Count Vectorizer</text>
  <text x="280" y="125" text-anchor="middle" font-size="10" fill="white">Remove stop words</text>
  <text x="280" y="140" text-anchor="middle" font-size="10" fill="white">102,694 features</text>
  
  <!-- Train/Test Split -->
  <rect x="390" y="50" width="120" height="60" rx="10" fill="#f39c12" stroke="#e67e22" stroke-width="2"/>
  <text x="450" y="75" text-anchor="middle" font-size="11" font-weight="bold" fill="white">Training Set</text>
  <text x="450" y="90" text-anchor="middle" font-size="9" fill="white">4,057 samples (70%)</text>
  
  <rect x="390" y="130" width="120" height="60" rx="10" fill="#9b59b6" stroke="#8e44ad" stroke-width="2"/>
  <text x="450" y="155" text-anchor="middle" font-size="11" font-weight="bold" fill="white">Test Set</text>
  <text x="450" y="170" text-anchor="middle" font-size="9" fill="white">1,739 samples (30%)</text>
  
  <!-- Naive Bayes Model -->
  <rect x="560" y="80" width="140" height="80" rx="10" fill="#27ae60" stroke="#229954" stroke-width="2"/>
  <text x="630" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Multinomial</text>
  <text x="630" y="125" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Naive Bayes</text>
  <text x="630" y="145" text-anchor="middle" font-size="10" fill="white">Binary Classification</text>
  
  <!-- Arrows -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Arrows between components -->
  <line x1="170" y1="120" x2="220" y2="120" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="340" y1="120" x2="390" y2="80" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="340" y1="120" x2="390" y2="160" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  <line x1="510" y1="80" x2="560" y2="100" stroke="#34495e" stroke-width="2" marker-end="url(#arrowhead)"/>
  
  <!-- Performance Metrics Box -->
  <rect x="50" y="220" width="700" height="160" rx="15" fill="#ecf0f1" stroke="#bdc3c7" stroke-width="2"/>
  <text x="400" y="250" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">
    Model Performance Metrics
  </text>
  
  <!-- Metrics Grid -->
  <!-- Accuracy -->
  <rect x="80" y="270" width="140" height="80" rx="8" fill="#2ecc71" stroke="#27ae60" stroke-width="1"/>
  <text x="150" y="295" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Accuracy</text>
  <text x="150" y="315" text-anchor="middle" font-size="20" font-weight="bold" fill="white">95.46%</text>
  <text x="150" y="335" text-anchor="middle" font-size="10" fill="white">1660/1739 correct</text>
  
  <!-- Precision -->
  <rect x="240" y="270" width="140" height="80" rx="8" fill="#3498db" stroke="#2980b9" stroke-width="1"/>
  <text x="310" y="295" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Precision</text>
  <text x="310" y="315" text-anchor="middle" font-size="20" font-weight="bold" fill="white">99.17%</text>
  <text x="310" y="335" text-anchor="middle" font-size="10" fill="white">Low false positives</text>
  
  <!-- Recall -->
  <rect x="400" y="270" width="140" height="80" rx="8" fill="#e74c3c" stroke="#c0392b" stroke-width="1"/>
  <text x="470" y="295" text-anchor="middle" font-size="14" font-weight="bold" fill="white">Recall</text>
  <text x="470" y="315" text-anchor="middle" font-size="20" font-weight="bold" fill="white">86.46%</text>
  <text x="470" y="335" text-anchor="middle" font-size="10" fill="white">Spam detection rate</text>
  
  <!-- F1-Score -->
  <rect x="560" y="270" width="140" height="80" rx="8" fill="#9b59b6" stroke="#8e44ad" stroke-width="1"/>
  <text x="630" y="295" text-anchor="middle" font-size="14" font-weight="bold" fill="white">F1-Score</text>
  <text x="630" y="315" text-anchor="middle" font-size="20" font-weight="bold" fill="white">92.38%</text>
  <text x="630" y="335" text-anchor="middle" font-size="10" fill="white">Balanced measure</text>
  
  <!-- Example Predictions -->
  <rect x="50" y="400" width="700" height="150" rx="15" fill="#ffffff" stroke="#bdc3c7" stroke-width="2"/>
  <text x="400" y="430" text-anchor="middle" font-size="18" font-weight="bold" fill="#2c3e50">
    Sample Predictions
  </text>
  
  <!-- Example emails -->
  <rect x="70" y="450" width="300" height="25" rx="5" fill="#ecf0f1" stroke="#bdc3c7"/>
  <text x="80" y="468" font-size="11" fill="#2c3e50">"I want to gets some free viagra"</text>
  <rect x="380" y="450" width="60" height="25" rx="5" fill="#95a5a6"/>
  <text x="410" y="468" text-anchor="middle" font-size="11" font-weight="bold" fill="white">HAM</text>
  
  <rect x="70" y="480" width="300" height="25" rx="5" fill="#ecf0f1" stroke="#bdc3c7"/>
  <text x="80" y="498" font-size="11" fill="#2c3e50">"Lets go to Murree"</text>
  <rect x="380" y="480" width="60" height="25" rx="5" fill="#95a5a6"/>
  <text x="410" y="498" text-anchor="middle" font-size="11" font-weight="bold" fill="white">HAM</text>
  
  <rect x="70" y="510" width="300" height="25" rx="5" fill="#ecf0f1" stroke="#bdc3c7"/>
  <text x="80" y="528" font-size="11" fill="#2c3e50">"need Mortage? reply to arrange call..."</text>
  <rect x="380" y="510" width="60" height="25" rx="5" fill="#e74c3c"/>
  <text x="410" y="528" text-anchor="middle" font-size="11" font-weight="bold" fill="white">SPAM</text>
  
  <!-- Legend -->
  <text x="500" y="468" font-size="12" font-weight="bold" fill="#2c3e50">Classification:</text>
  <rect x="500" y="480" width="15" height="15" fill="#95a5a6"/>
  <text x="520" y="492" font-size="11" fill="#2c3e50">Ham (0) - Legitimate</text>
  <rect x="500" y="500" width="15" height="15" fill="#e74c3c"/>
  <text x="520" y="512" font-size="11" fill="#2c3e50">Spam (1) - Unwanted</text>
  
  <!-- Footer -->
  <text x="400" y="580" text-anchor="middle" font-size="12" fill="#7f8c8d">
    Built with scikit-learn | CountVectorizer + MultinomialNB
  </text>
</svg>

