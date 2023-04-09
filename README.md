# Cancer-Stage-Prediction

The provided code performs a regression analysis on cancer data using different machine learning models.

Firstly, the necessary libraries are imported, and the data is read from a CSV file. Then, the features and the target variable are separated from the data, and converted to numpy arrays for further analysis.

Next, five regression models are initialized: ExtraTreesRegressor, RandomForestRegressor, DecisionTreeRegressor, LinearRegression, and GradientBoostingRegressor.

The KFold function is used to split the data into five folds for cross-validation. Then, for each fold, the data is split into training and testing sets, and the chosen regression model is trained on the training set. The trained model is then used to predict the target variable for the test set, and the accuracy is calculated using the r2_score function from the sklearn.metrics library. The r2_score is a statistical measure of how close the data are to the fitted regression line.

The average r2_score for each model is calculated over all folds, and printed to the console. The ExtraTreesRegressor, RandomForestRegressor, DecisionTreeRegressor, LinearRegression, and GradientBoostingRegressor models are trained and evaluated on the same dataset using the same evaluation metric.

Finally, the results are displayed to the user.
 
