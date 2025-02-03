# ML-Prediction-Decision-Tree

This project demonstrates how to build and persist a **Decision Tree** model for predicting music genres based on user data such as age and gender. It uses the **`sklearn`** library for model training, and **`joblib`** for model persistence. The project also includes code for visualizing the decision tree and making predictions.

## Steps Involved

### 1. **Upload the Data**
The dataset is in CSV format, containing information about users' **age**, **gender**, and the music **genre** they like. The values are encoded as follows:
- **Gender**: 1 = Male, 0 = Female
- **Genre**: Categorical variable (e.g., HipHop, Jazz, Classical, Dance, etc.)

```python
import pandas as pd
music_data = pd.read_csv('music.csv')
```

Example data:

| age | gender | genre    |
| --- | ------ | -------- |
| 20  | 1      | HipHop   |
| 23  | 1      | HipHop   |
| 25  | 1      | HipHop   |
| 26  | 1      | Jazz     |
| 20  | 0      | Dance    |
| 21  | 0      | Dance    |

### 2. **Preparing the Data**
Prepare the input features (age and gender) and the target variable (genre) for training the model.

```python
X = music_data.drop(columns=['genre'])
y = music_data['genre']
```

### 3. **Splitting the Dataset**
The dataset is split into **input sets** (X) and **output sets** (y). The input set contains the age and gender of users, while the output set contains the genres they prefer.

```python
X = music_data.drop(columns=['genre'])  # Input features
y = music_data['genre']  # Target variable
```

### 4. **Training the Model**
A **Decision Tree Classifier** is used to build the model using the training data.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)
```

### 5. **Making Predictions**
After the model is trained, you can make predictions based on new input data, such as a 21-year-old male and a 22-year-old female.

```python
X_test = pd.DataFrame([[21, 1], [22, 0]], columns=X.columns)
predictions = model.predict(X_test)
```

The output will be an array of predicted genres:

```python
array(['HipHop', 'Dance'], dtype=object)
```

### 6. **Calculating Model Accuracy**
The model's accuracy is calculated using the **accuracy_score** function after splitting the data into training and test sets.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
```

If the model is accurate, the score might be 1.0.

### 7. **Persisting the Model**
The trained model is saved using **joblib** so it can be loaded and used later for predictions without retraining.

```python
import joblib

joblib.dump(model, 'music-recommender.joblib')
```

### 8. **Loading the Saved Model**
To make predictions using the saved model:

```python
model = joblib.load('music-recommender.joblib')
X_test = pd.DataFrame([[21, 1]], columns=X.columns)
predictions = model.predict(X_test)
```

### 9. **Visualizing the Decision Tree**
The decision tree model can be visualized using **Graphviz**. The following code exports the decision tree as a `.dot` file:

```python
from sklearn.tree import export_graphviz

tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
```

You can open the `.dot` file in a **Graphviz viewer** to visualize the decision tree structure.

**Decision Tree:**

![image alt](https://github.com/Fakhrul-Ahmed/ML-Prediction-Decision-Tree/blob/main/Decision%20%20Tree.PNG?raw=true)

### 10. **Conclusion**
This project demonstrates how to train a **Decision Tree** model to predict music genres based on age and gender, make predictions, calculate accuracy, and persist the model for later use. It also includes code for visualizing the decision tree and exporting it for further analysis.
