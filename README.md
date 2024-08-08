# Diabetes-Prediction

This project demonstrates the use of a Support Vector Machine (SVM) for predicting whether a person is diabetic based on certain medical attributes. The dataset used is the PIMA Indian Diabetes Dataset.

Table of Contents
Installation
Usage
Project Structure
Data Collection and Processing
Model Training and Evaluation
Making Predictions
Contributing
License
Installation
To run this project, you need to have Python installed. Clone this repository and install the required dependencies using pip:

bash
Copy code
git clone https://github.com/yourusername/diabetes-prediction-svm.git
cd diabetes-prediction-svm
pip install -r requirements.txt
Usage
Ensure the dataset diabetes.csv is in the root directory of the project.
Run the main.py script to train the model and make predictions.
bash
Copy code
python main.py
Project Structure
css
Copy code
diabetes-prediction-svm/
├── main.py
├── diabetes.csv
├── requirements.txt
└── README.md
main.py: Contains the code for data processing, model training, evaluation, and prediction.
diabetes.csv: The dataset used for training and testing the model.
requirements.txt: Lists the Python dependencies required for the project.
README.md: This file.
Data Collection and Processing
The dataset used for this project is the PIMA Indian Diabetes Dataset. Each instance contains information such as the number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.

Key steps in data processing:

Load the dataset using pandas.
Display basic statistical measures using describe().
Separate features and labels.
Standardize the data using StandardScaler.
Model Training and Evaluation
The SVM model is trained using the svm.SVC class from sklearn.svm. The dataset is split into training and testing sets using train_test_split.

The model's performance is evaluated using accuracy scores on both the training and testing data.

Making Predictions
The script also includes a section for making predictions with the trained model. It reshapes the input data and uses the model to predict whether the person is diabetic.

python
Copy code
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.
