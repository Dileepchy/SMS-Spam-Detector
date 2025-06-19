# SMS Spam Detector

 <!-- You can create your own banner or remove this line -->

A Machine Learning-powered web application built with Streamlit to classify SMS messages as "Spam" or "Ham" (not spam). This project uses Natural Language Processing (NLP) techniques to process text data and a trained Naive Bayes classifier to make predictions in real-time.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.12%2B-red.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Live Demo

_A screenshot or GIF of your application is highly recommended here. It's the best way to show what your project does._

_To create a GIF like this, you can use tools like [LICEcap](https://www.cockos.com/licecap/) or [ScreenToGif](https://www.screentogif.com/)._

---

## ✨ Features

- **Real-time Classification:** Instantly classify any SMS message you enter.
- **User-Friendly Interface:** A clean and simple UI built with Streamlit.
- **NLP Preprocessing:** Implements text cleaning, tokenization, stop-word removal, and stemming.
- **High Accuracy:** Utilizes a Multinomial Naive Bayes model, which performs well for text classification tasks.
- **Data-Driven:** Trained on the well-known SMS Spam Collection Dataset from UCI.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **NLP:** NLTK (Natural Language Toolkit)
- **Data Manipulation:** Pandas, NumPy
- **Web Framework:** Streamlit
- **Model Persistence:** Pickle
- **Plotting:** Matplotlib, Seaborn (for notebook analysis)

---

## 📂 Project Structure

```
SMS-Spam-Detector/
├── .gitignore
├── 1. EDA and Preprocessing.ipynb   # Jupyter Notebook for analysis and model training
├── app.py                          # The main Streamlit web application
├── model.pkl                       # Trained Naive Bayes model file
├── preprocessor.py                 # Python script for text preprocessing function
├── requirements.txt                # List of project dependencies
├── vectorizer.pkl                  # Trained TF-IDF vectorizer file
└── README.md
```

---

## ⚙️ Setup and Installation

To run this project on your local machine, follow these steps:

**1. Prerequisites:**

- Python 3.9 or higher
- pip & virtualenv

**2. Clone the Repository:**

```bash
git clone https://github.com/Dileepchy/SMS-Spam-Detector.git
cd SMS-Spam-Detector
```

**3. Create and Activate a Virtual Environment:**

- **For macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **For Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

**4. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**5. Download NLTK Data:**
Open a Python interpreter in your terminal and run the following commands:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**6. Run the Streamlit App:**

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

---

## 📈 Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading:** The [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) is loaded using Pandas.
2.  **Exploratory Data Analysis (EDA):** The dataset is analyzed to understand the distribution of spam vs. ham messages, message lengths, and most common words.
3.  **Text Preprocessing:**
    - **Cleaning:** Removing special characters and converting text to lowercase.
    - **Tokenization:** Splitting sentences into individual words (tokens).
    - **Stop Word Removal:** Eliminating common words that don't add much meaning (e.g., "the", "a", "is").
    - **Stemming:** Reducing words to their root form (e.g., "running" -> "run").
4.  **Feature Extraction:** The preprocessed text data is converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.
5.  **Model Training:** A **Multinomial Naive Bayes** classifier is trained on the vectorized data. This model is well-suited for classification with discrete features like word counts.
6.  **Model Evaluation:** The model's performance is checked using metrics like accuracy, precision, and recall.
7.  **Deployment:** The trained TF-IDF vectorizer and Naive Bayes model are saved as `.pkl` files and served via a Streamlit web interface.

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description:** The dataset contains 5,572 SMS messages in English, tagged as either "ham" (legitimate) or "spam".

---

## 🙏 Acknowledgements

- This project was inspired by the tutorial. A big thanks to them for their excellent educational content.
- The dataset creators from the UCI Machine Learning Repository.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
