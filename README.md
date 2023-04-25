# Machine-Learning-Portfolio
## Extensive use of Regression, Classification, Clustering and Neural Networks

# eCommerce Customer Salary Prediction Project

This project aims to predict the incomes of eCommerce customers to help make better business decisions, such as pricing products and targeting advertising. The project involves fitting several machine learning models to customer data, evaluating their performance, and making a recommendation on the best method.

## Dataset

The dataset contains 1,000 records and 9 columns, including Age, Sex, Education, Region, WorkType, Salary, SiteSpending, etc. The data has already been cleaned and preprocessed.

## Preprocessing

The preprocessing steps include:

- Encoding categorical variables using Label Encoding
- Splitting the dataset into train/test sets
- Scaling the features using MinMaxScaler

## Models and Algorithms

The project tests various machine learning models for regression, binary classification, and clustering:

1. Regression Models:
   - Linear Regression
   - Extra Tree Regression
   - Support Vector Regression
   - Random Forest Regression

2. Binary Classification Models:
   - Logistic Regression
   - Support Vector Machine
   - Decision Tree Classifier

3. Neural Networks for Classification

4. Clustering:
   - K-means Clustering Algorithm

## Evaluation Metrics

The evaluation metrics used for regression models are R2 score, and for binary classification models, the confusion matrix and classifier accuracy are used.

## Results

The following models performed best in their respective categories:

1. Regression: Random Forest Regression (R2 score: 87.83%)
2. Binary Classification: Decision Tree Classifier (Accuracy: 93%)
3. Neural Networks: Neural Network Model (Accuracy: 61.8%)
4. Clustering: K-means Clustering (k=3)

## Conclusion

Random Forest Regression is recommended for predicting a customer's salary, and Decision Tree Classifier is recommended for classifying customers based on their salary. K-means Clustering is also recommended as an unsupervised learning method for classification.

## Dependencies

- Python 3.x
- NumPy
- pandas
- scikit-learn
- TensorFlow
- Keras
- Matplotlib
- seaborn


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
