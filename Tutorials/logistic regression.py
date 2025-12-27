from sklearn.linear_model import LogisticRegression
import numpy as np

# (hours studied for exams a week vs pass or not) 2d array
data = np.array([[0, 'NO'],
                 [2.5, 'NO'],
                 [5, 'YES'],
                 [15, 'YES']])

# set features
x = data[:, 0].astype(float).reshape(-1, 1) # slice the array to only take the first column then shape it into a 2d array else there would be a error
y = data[:, 1] # take the second column and sklearn doesnt expect it to be a 2d array

# create and train the model
model = LogisticRegression().fit(x,y)

print('Will the students pass the exams with 2, 3, 4, 10 hours studied in a week?')
print(model.predict([[2],[3],[4],[10]]))