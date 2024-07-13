# Comment-Toxicity-Prediction-App
Deployed App Link : https://sanjeevk11-comment-toxicity-prediction-app-app-z20khk.streamlit.app/


Demo Video Link : https://drive.google.com/file/d/1_fnVdqGFYLQM8hU91Yoz521-hN2xTji4/view?usp=sharing


# Comment Toxicity Prediction App

## Overview

This project, the "Comment-Toxicity-Prediction-App," utilizes a machine learning algorithm known as Naive Bayes for predicting the toxicity of comments. The model calculates the probability of a comment falling into categories such as toxic, severe toxic, obscene, insult, threat, identity hate, etc.

## Machine Learning Model

The toxicity prediction model is built using the Naive Bayes algorithm. This algorithm is commonly used for text classification tasks, making it suitable for analyzing and categorizing comments based on their toxicity levels. The model has been trained on a labeled dataset to make predictions on new comments.

## Prediction Categories

The model categorizes comments into the following toxicity levels:

- Toxic
- Severe Toxic
- Obscene
- Insult
- Threat
- Identity Hate

## Usage

To use the Comment Toxicity Prediction App, follow these steps:

1. **Clone Repository:**
   ```bash
   [git clone https://github.com/sanjeevk11/Comment-Toxicity-Prediction-App.git]
   cd Comment-Toxicity-Prediction-App
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```

4. **Access the App:**
   Open your web browser and navigate to `http://localhost:8501` to interact with the Comment-Toxicity-Prediction-Model.

5. Enter a comment in the input field, and the app will provide the predicted probabilities for each toxicity category.

## Sample output:

### Good Comment
.
.
![good comment](https://github.com/user-attachments/assets/c7a0e4c9-6363-438f-8482-49f3ac83249f)


### Bad Comment
.
.
![bad comment](https://github.com/user-attachments/assets/de71f7f4-f887-4170-88cc-ca8b235fd4c4)

