# ML-supervised-algorithms
Machine Learning supervised algorithms viz. Linear regression, Naive Bayes
#### Shahid Mohammed Shaikbepari
##### shaikbep@usc.edu 
This repository has two ML supervised algorithms implemented from scratch viz. Linear regression, Naive Bayes and the datasets used are candyshop_data.txt and seeds_dataset.txt
#### Linear Regression:
To run: python linear_reg.py

#### Motivation
Regression analysis is a set of useful statistical modeling methods that estimates the relationships among variables. Specifically, in Machine Learning, regression is widely used a as a supervised learning for prediction (e.g. Linear Regression) and classification (e.g. Logistic Regression) problems. In this part, I've implemented Linear Regression with one variable to help predict profits for a candy shop. 

##### Dataset:
The file candyshop_data.txt contains the dataset for this linear regression problem. The first column is the population (in 10,000 s) of a city and the second column is the profit (in 10,000 $ s) of a candy shop in that city. A negative value for profit indicates a loss. 
##### Model:
The model is linear (affine) I've augmented the given input feature with a dimension containing ones. This is implemented with linear least square errors


#### Algorithm description: Linear least square errors
---------------------- 

1. Function readData() was created to read the file candyshop_data.txt using pandas
2. A X matrix was created of order Nx2 with 1st column as populations and 2nd column as 1’s
3. A Y matrix was created of order Nx1 with all the profits in it correspondingly
4. Using pandas, numpy methods calculated the w (order 2x1) as per the formula given in spec
5. Defined two methods abline(), line() for plotting the line and getting y values for the line equation
6. Using matplotlib drawn the plots for the data and calculated line

