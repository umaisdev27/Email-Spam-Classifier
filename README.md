# 📧 Email Spam Classifier  
*Because your inbox deserves some peace of mind!* 🚫 (No more “You won a free iPhone” emails )  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/ML-Naive%20Bayes-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Accuracy-97.74%25-success?logo=github" />
  <img src="https://img.shields.io/badge/Spam-Hunter-red?logo=mailchimp" />
</p>  

---

## 🗂️ Project Structure  
📁 **`BayesClassifier.ipynb`** → Data Preparation Notebook  
- 📚 Imports: `pandas`, `nltk`, `matplotlib`, `numpy`  
- 📥 Load spam/ham emails from directories  
- 🧹 Clean & preprocess: remove empties, lowercase, tokenize, remove stopwords, stem words  
- 🔬 EDA (Exploratory Data Analysis):  
  - 📊 Spam vs Ham distribution  
  - 🔠 Word frequency analysis  
  - ☁️ Word clouds  

---

## 🔍 Example Outputs  
Spam/Ham Distribution:  
<div center>
<img width="486" height="394" src="https://github.com/user-attachments/assets/5e075b49-1848-4dde-b320-fcc79f78d98b" />  
</div>
Word Cloud:  
<img width="486" height="500" src="https://github.com/user-attachments/assets/48b154c0-3671-4901-a472-bfd563492c99" />  
 

---

## 🤖 Naive Bayes Training  
- ⚙️ **Setup**: constants (vocab size = `2500`), file paths  
- 🗂️ **Data Loading**: sparse training/test → dense DataFrame  
- 🧮 **Model Training**:  
  - `P(Spam)` & `P(Ham)` priors  
  - Word counts per category  
  - `P(Token|Spam)`, `P(Token|Ham)` with Laplace smoothing (to avoid zero probs 🤌)  
- 📤 **Export**: probability vectors + prepared test set  

**🎯 Output:** Model ready to say *“Spam detected, bhai!”* 🚔  

---

## 🧪 Testing the Model  
<img width="610" height="633" src="https://github.com/user-attachments/assets/4bf988a6-ab57-40b2-876d-af9c68dc4c36" />  

Steps:  
1. Load test features, labels, and token probs  
2. Compute joint log-probs (spam vs ham)  
3. Apply prior (0.3116)  
4. Predict → higher log-prob wins 🏆  
5. Accuracy check → ✅ `97.74%` (1685 correct / 39 wrong)  

📊 Bonus: Scatter plot with decision boundary included!  

---

## 📈 Model Performance  
| Metric      | Score   |
|-------------|---------|
| 🎯 Accuracy | **95.46%** |
| ✅ Precision | **99.17%** |
| 🔍 Recall   | **86.46%** |
| ⚖️ F1-Score | **92.38%** |

> 💡 *High precision = no false alarms, so legit mails from your crush won’t land in spam 😏*  

---

## ⚙️ Technical Details  
- 📑 **Dataset:** 5,796 emails (70% train / 30% test)  
- 🔤 **Features:** 102,694 unique words (after stopword removal)  
- 🤓 **Algorithm:** Multinomial Naive Bayes + Count Vectorization  

### 💪 Key Strength  
- **Precision 99.17%** → Fake lottery emails: *busted* ✅  
- Real emails: safe 🛡️  

---

## 🖼️ Full Model Diagram  
<img width="845" height="567" src="https://github.com/user-attachments/assets/b71c89d7-4566-4016-9404-711171eb3c13" />  

---
## ✨ Credits  
Built with ❤️, Python 🐍, and fueled by 1000+ spam emails promising free Bitcoin 😅.  

---
