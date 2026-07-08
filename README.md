# 🌳 Decision Tree Classifier for Customer Purchase Prediction

## 📌 Project Overview

This project builds a **Decision Tree Classifier** to predict whether a customer will subscribe to a bank term deposit based on demographic and behavioral information. The model is trained using the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

This project was completed as part of my **Data Science Internship (Task 03)**.

---

## 🎯 Objective

Develop a Decision Tree machine learning model to predict whether a customer will purchase a product or service based on customer-related features such as age, job, marital status, education, account balance, previous marketing campaigns, and more.

---

## 📂 Dataset

**Dataset:** Bank Marketing Dataset

**Source:** UCI Machine Learning Repository

The dataset contains customer demographic information, financial details, and previous marketing campaign information.

### Target Variable

* **yes** → Customer subscribed to the term deposit.
* **no** → Customer did not subscribe.

---

## 🛠 Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* UCI ML Repository (`ucimlrepo`)

---

## 📚 Machine Learning Workflow

1. Load the dataset from the UCI Machine Learning Repository.
2. Explore and understand the dataset.
3. Handle missing values.
4. Encode categorical variables using One-Hot Encoding.
5. Split the dataset into training and testing sets.
6. Train a Decision Tree Classifier.
7. Make predictions on the test dataset.
8. Evaluate the model using:

   * Accuracy Score
   * Confusion Matrix
   * Classification Report
9. Visualize the Decision Tree.
10. Display Feature Importance.

---

## 📁 Project Structure

```text
SCT_SC_03/
│── prediction_customer_purchase.py
│── feature_importance.png
│── decision_tree.png
│── README.md
│── requirements.txt
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/sunilvk2008-dev/SCT_SC_03.git
```

Move into the project folder:

```bash
cd SCT_SC_03
```

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn ucimlrepo
```

---

## ▶️ Run the Project

```bash
python prediction_customer_purchase.py
```

---

## 📊 Output

The program displays:

* Dataset information
* Missing values
* Target variable distribution
* Accuracy Score
* Confusion Matrix
* Classification Report
* Feature Importance
* Decision Tree Visualization

The following images are generated:

* `feature_importance.png`
* `decision_tree.png`

---

## 🌳 How Decision Tree Works

A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks.

It works by asking a sequence of questions about the input features. At each step, it selects the feature that best separates the data into different classes using **Entropy (Information Gain)**. The process continues until the model reaches a final prediction.

---

## 📈 Results

The trained Decision Tree classifier successfully predicts whether a customer is likely to subscribe to a bank term deposit based on demographic and behavioral features.

The project also provides visual insights through:

* Feature Importance Graph
* Decision Tree Visualization

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Random Forest Classifier
* XGBoost Classifier
* Cross-validation
* Interactive dashboard using Streamlit

---

## 👨‍💻 Author

**Sunil Kumar**

B.Tech Artificial Intelligence & Machine Learning Student

GitHub: https://github.com/sunilvk2008-dev

---

## ⭐ Acknowledgements

* UCI Machine Learning Repository
* Scikit-learn Documentation
* Pandas Documentation
* Matplotlib Documentation
