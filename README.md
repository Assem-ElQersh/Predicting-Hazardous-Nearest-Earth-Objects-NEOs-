# **Predicting Hazardous NEOs (Nearest Earth Objects)**

## **Project Overview**

This project focuses on predicting whether a Nearest Earth Object (NEO) is hazardous to Earth. The dataset used is sourced from NASA (uploaded on Kaggle and can be downloaded from [here](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024)) and spans from 1910 to 2024, tracking 338,199 objects. The main goal is to develop a machine-learning model that accurately classifies these NEOs into hazardous and non-hazardous categories, which is crucial for planetary defense.

## **Dataset Information**

The dataset consists of 338,199 records of NEOs tracked by NASA, and it includes information such as the NEO’s size, velocity, distance from Earth, and whether NASA classified the object as hazardous.

### **Key Features:**
- `neo_id`: Unique identifier for each NEO.
- `name`: Name of the NEO.
- `absolute_magnitude`: The brightness of the NEO.
- `estimated_diameter_min`: Minimum estimated diameter of the NEO (km).
- `estimated_diameter_max`: Maximum estimated diameter of the NEO (km).
- `orbiting_body`: The celestial body the NEO is orbiting.
- `relative_velocity`: The velocity of the NEO relative to Earth (km/h).
- `miss_distance`: The distance between Earth and the NEO at its closest point (km).
- `is_hazardous`: Boolean value indicating if the NEO is classified as hazardous (True/False).

## **Project Structure**

The project consists of the following steps:

1. **Data Importing and Cleaning**:
   - I handled missing values by filling numeric columns with the median and categorical columns with the mode.
   - Outliers are identified and removed using the IQR (Interquartile Range) method.
   
2. **Exploratory Data Analysis (EDA)**:
   - Visualizations such as **box plots**, **histograms**, and **pie charts** are created using libraries like **Seaborn** and **Plotly** to understand data distribution and relationships.
   - A correlation matrix is plotted to identify relationships between numerical features.

3. **Handling Class Imbalance**:
   - The dataset is highly imbalanced, with far fewer hazardous objects compared to non-hazardous ones.
   - I applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the classes for improved model performance.

4. **Feature Selection**:
   - Features such as `absolute_magnitude`, `estimated_diameter_min`, `estimated_diameter_max`, `relative_velocity`, and `miss_distance` are selected for model training.
   - Columns like `neo_id`, `name`, and `orbiting_body` are dropped as they do not provide predictive value.

5. **Model Training**:
   - I built two machine learning models: 
     - **RandomForestClassifier**
     - **XGBoostClassifier**
   - Both models are trained and evaluated on a balanced dataset.

6. **Evaluation**:
   - Evaluation is performed using metrics such as:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1 Score**
     - **Confusion Matrix**
   - I also visualized the **learning curves** of the models to understand their performance on training and test sets.

## **Code Execution**

### 1. Install Necessary Packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Script:
Execute the Jupyter Notebook included in the repository. Ensure the dataset is downloaded.

```python
# Load Data
file_name = "nearest-earth-objects(1910-2024).csv"
file_path = os.path.join(os.path.dirname(os.path.abspath(file_name)), file_name)
```

### 3. Model Building:
I split the dataset into 80% training and 20% testing, and then trained both models:
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Model Building and Evaluation
rf_model = RandomForestClassifier(random_state=42)
xg_model = XGBClassifier(random_state=42)
```

### 4. Evaluation:
Both models are evaluated using classification reports and confusion matrices:
```python
from sklearn.metrics import classification_report, confusion_matrix

# Random Forest Evaluation
print(classification_report(y_test, rf_preds))
cm = confusion_matrix(y_test, rf_preds)

# XGBoost Evaluation
print(classification_report(y_test, xg_preds))
cm = confusion_matrix(y_test, xg_preds)
```

## **Results and Evaluation**

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| **Random Forest**       | 95.75%    | 0.96      | 0.95   | 0.96     |
| **XGBoost Classifier**  | 89.33%    | 0.84      | 0.97   | 0.90     |

Both models perform well with high accuracy, but the **RandomForestClassifier** slightly outperforms the **XGBoostClassifier**.

### **Confusion Matrix Visualization**:
I plotted the confusion matrix to visually assess model performance.

## **Conclusion**

- The dataset shows a significant imbalance between hazardous and non-hazardous NEOs, which I effectively managed using **SMOTE**.
- The **RandomForestClassifier** achieved an F1 Score of **96%**, making it the best-performing model in this project.
- This project demonstrates a full pipeline from **data cleaning** and **exploratory analysis** to **model building** and **evaluation**, ready to be used in real-world applications for planetary defense.

## **How to Run the Project**

1. Clone the repository:
   ```bash
   git clone https://github.com/Assem-ElQersh/Predicting-Hazardous-Nearest-Earth-Objects-NEOs-.git
   cd NEO-hazardous-prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to train and evaluate the model.

## **Repository Structure**
```
├── data/
│   ├── nearest-earth-objects(1910-2024).csv
├── src/
│   ├── Model.ipynb
├── README.md
└── requirements.txt
```

##
- Please note that this project is part of the **MLSC Data Science Graduation Project**.
