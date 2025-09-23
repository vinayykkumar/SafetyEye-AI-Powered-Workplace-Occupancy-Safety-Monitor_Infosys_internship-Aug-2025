from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # feature (hours studied)
y = np.array([2, 4, 5, 4, 5])            # target (marks scored)

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[6]])
print("Predicted marks for 6 hours studied:", pred[0])

# Plotting
plt.scatter(X, y, color='blue', label='Actual Data')   # scatter plot of actual data
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # regression line
plt.scatter(6, pred, color='green', s=100, marker='*', label='Prediction (6 hrs)')

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
