import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

HERE = Path(__file__).resolve().parent
data_file = HERE / 'salary_data_cleaned.csv'

df = pd.read_csv(data_file) # Real average weekly earnings using consumer price inflation
df = df[df['age'].between(16, 100)] # get ages between 16 years old to 100 years old because there are bad data in the csv
df = df[['avg_salary', 'age']] # avg_salary is in in the thousands


print('initalizing linear regression model')
regression = linear_model.LinearRegression()
regression.fit(df[['age']], df[['avg_salary']])

print('predict a worker\'s salary whose age is 58')
predict_age = np.array([[58]])
print(regression.predict(predict_age))

# how it works
# basically since we got x and y value, to draw the best fit line we need m and b which is the gradient and y-intercept
# given the equation y = mx + b (normal linear algebra formula)
# what the program does is that it computes those m and b unknowns and uses it to "predict" the values of unknown x or y

print(f'Value of m: {regression.coef_}')
print(f'value of y-intercept: {regression.intercept_}')
# sub the values in
# since we predicted the x value, just sub in the x value to obtain y
what_was_y = (regression.coef_)*(58) + (regression.intercept_)
print(what_was_y)
# then plot it with the dataset
what_was_y = np.array([[what_was_y]])
plt.scatter(df.age, df.avg_salary)
plt.scatter(predict_age, what_was_y)
plt.show()

# a visual representation of the best fit line is also possible and very easy to implement.
age_range = np.linspace(df.age.min(), df.age.max(), 200).reshape(-1, 1) # plot evenly across 200 data points
salary_line = regression.predict(age_range).ravel()
plt.scatter(df.age, df.avg_salary)
plt.plot(age_range.ravel(), salary_line, color='orange', label='linear fit') # draw the line
plt.xlabel('Age')
plt.ylabel('Average salary (in thousands)')
plt.show()