## Regression

From my own understanding, regression is to build a model which can fit the training data most closely. We normally use *mean squared error* (MSE) as the measure of the fit quality and the objective function when we estimate the parameters.

The methods we usually use to do regression in Spark MLlib is `linearRegressionWithSGD.train` and we use `predict` to do the prediction with the regression model we obtain. Note that the 'SGD' here refers to Stochastic Gradient Descent.

```python
# the two lines below are added so that this code can be run as a self-containd application.
from pyspark import SparkContext
sc = SparkContext("local", "Simple App")

# load the necessary modules
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/mllib/ridge-data/lpsa.data")
parsedData = data.map(parsePoint)


# split the data into two sets for training and testing
# Here I have set the seed so that I can reproduce the result
(trainingData, testData) = parsedData.randomSplit([0.7, 0.3], seed=100)


# Build the model
model = LinearRegressionWithSGD.train(trainingData)


# Evaluate the model on training data
# --- Point 1 ---
Preds = testData.map(lambda p: (float(model.predict(p.features)), p.label))
MSE = Preds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / Preds.count()
print("Mean Squared Error = " + str(MSE))
print("\n")

# --- Point 2 ---
# More about model evaluation and regression analysis
# Instantiate metrics object
metrics = RegressionMetrics(Preds)

# Squared Error
print("MSE = %s" % metrics.meanSquaredError)
print("RMSE = %s" % metrics.rootMeanSquaredError)

# R-squared
print("R-squared = %s" % metrics.r2)

# Mean absolute error
print("MAE = %s" % metrics.meanAbsoluteError)

# Explained variance
print("Explained variance = %s" % metrics.explainedVariance)
```
We can run this script as an application with `spark-submit` command and get the output
```
Mean Squared Error = 7.35754024842

MSE = 7.35754024842
RMSE = 2.71247861714
R-squared = -4.74791121611
MAE = 2.52897021533
Explained variance = 7.89672343551
```

##### Point 1: A Small Trap
Note that we need to exclusively convert the predicted values into float, otherwise you'll encounter an error like below
```
TypeError: DoubleType can not accept object in type <type 'numpy.float64'>
```
when you call the command `metrics = RegressionMetrics(Preds)` (but everything would be okay if you don't do regression analysis with `metrics` method). 

And you'll also need to include `p.label` if you want to do regression analysis with `metrics` method.


##### Point 2: Regression Analysis
`MLlib` provided the most commonly used metrics for regressiona analysis. You may refer to https://en.wikipedia.org/wiki/Regression_analysis for the relevant information.

