# Ridge Regression
Each row in both “X” files contain six features for a single car (plus a 1 in the 7th dimension) and the same row in the corresponding “y” file contains the miles per gallon for that car. Specifically, the dimensions are 
1. cylinders, 2. displacement, 3. horsepower, 4. weight, 5. acceleration, 6. year made and 7. represents for w_0. 

This code achieves the following functions:

1. With given λ, we calculate the relationship between the 7 values in wRR as a function of df(λ) by ridge regression;
2. Using trained wRR to predict the given test samples with given λ, and showing the RMSE as a function of df(λ);
3. Expand the regression model to a pth-order polynomial regression model. Given user defined order p and λ, showing the test RMSE as a function of λ and finding out the optimal model for the question.
