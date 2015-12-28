# Spark Machine Learning Practice

In this repo, I try to introduce some basic machine learning usages of *PySpark*. The contents I'm going to cover would be quite simple. But I guess it would be helpful for some people since I would cover some questions I encountered myself from the perspective of a person who's used to more "normal" ML settings (like R language).

Some of the examples are from the official examples given by Spark. But I will give more details.

- [Random Forest](#random-forest)
- [Regression](#regression)
- [References](#references)
- [License](#license)


## Random Forest

As a fan of greedy algorithm, I would like to start with *random forest* algorithm.

What is the idea of **Random Forest**? 

To put it simpel, averaging a set of observations reduces variance. Hence a natural way to reduce the variance and hence increase the prediction accuracy of a machine learning model is to take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions [1]. This is the idea of **bagging**, a special case of random forest. 

Then we may need to subset the predictors. That is, in each training procedure, we don't use all the features we have. You may ask WHY since this seems like a 'waste' of resources we have. But let's suppose that there is a very strong predictor in the data, then in the models we produced, most of them will use that strong predictor in the top split and all of these decision trees will look similar, i.e., they're highly correlated. This may effect the reduction in variance and worsen the result. [1] This is why we only use randomly selected features in each tree model. 

This is just the idea of **random forest**. Simple, straitforward, and elegant at the same time.

Now let's have a look at the example code given by Spark. I commented the points where we may need to note (and details will be given later)

```python
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

# --- Point 1, 2 ---
# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])


# --- Point 3, 4, 5 ---
# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())

```

##### Point 1: LIBSVM data format
When I looked into LIBSVM data file for the first time, I got a little bit confused. But then I found its design is a brilliant idea.

LIBSVM data files look like below:
```
-1 1:-766 2:128 3:0.140625 4:0.304688 5:0.234375 6:0.140625 7:0.304688 8:0.234375
-1 1:-726 2:131 3:0.129771 4:0.328244 5:0.229008 6:0.129771 7:0.328244 8:0.229008
......
```
The first element of each row is the *label*, or we can say it's the *response value*. The labels can be either discrete or continuous. Normally, the labels will be discrete if we're working on classification, and continuous if we're trying to do regression. Following the labels are the *feature indices* and the *feature values* in format `index:value` (Please note that the index starts from `1` instead of `0` in LIBSVM data files, i.e., the indices are one-based and in ascending order. After loading, the feature indices are converted to zero-based [4]).

Sometimes we may find 'weird' LIBSVM data like below
```
-1 3:1 11:1 14:1 19:1 39:1 42:1 55:1 64:1 67:1 73:1 75:1 76:1 80:1 83:1 
-1 3:1 6:1 17:1 27:1 35:1 40:1 57:1 63:1 69:1 73:1 74:1 76:1 81:1 103:1 
-1 1:1 7:1 16:1 22:1 36:1 42:1 56:1 62:1 67:1 73:1 74:1 76:1 79:1 83:1 
```
The indices in it are not continuous. What's wrong? Actually the missing features are all 0. For example, in the first row, feature 1, 2, 4-10, 12-13, ... are all zero-values. This design is partially for the sake of memory usage. It would help improve the efficiency of the our programs if the data are sparse (containing quite many zero-values).


##### Point 2: Data Type "Labeled Point"

The data loaded by method `loadLibSVMFile` will be saved as `Labeled Points`. What is it?

MLlib supports local vectors and matrices stored on a single machine, as well as distributed matrices backed by one or more RDDs. Local vectors and local matrices are simple data models that serve as public interfaces. A training example used in supervised learning is called a “labeled point” in MLlib [4].

##### Point 3: How many trees we should have (`numTrees`)

This argument determines how many trees we build in a random forest. Increasing the number of trees will decrease the variance in predictions, and improve the model’s test accuracy. At the same time, training time will increaseroughly linearly in the number of trees.

Personally, I would recommend 400-500 as a 'safe' choice.


##### Point 4: How many features to use (`featureSubsetStrategy`)

As we mentioned above, the very unique charactristic of *random forest* is that in each tree we use a subset of features (predictors) instead of using all of them. Then, how many features should we use in each tree model? we can set `featureSubsetStrategy="auto"` of course so that the function we called will help us configure automatically, but we may want to tune it in some situations. Decreasing this number will speed up training, but can sometimes impact performance if too low [2].

For the function `RandomForest.trainClassifier` in PySaprk , argument `featureSubsetStrategy` supports“auto” (default), “all”, “sqrt”, “log2”, “onethird”. If “auto” is set, this parameter is set based on numTrees: if numTrees == 1, set to “all”; if numTrees > 1 (forest) set to “sqrt” [3].

Usually, given the number of features is `p`, we use `p/3` features in each model when building a random forest for regression, and use `sqrt(p)` features in each model if a random forest is built for classification [1].



##### Point 5: What is 'gini' --- the measures used to grow the trees (`impurity`)

`impurity` argument helps determine the criterion used for information gain calculation, and in PySpark the supported values are “gini” (recommended) or “entropy” [3]. Since random forest is some kind of *greedy algorithm*, we can say that `impurity` helps determine what is the objective function when the algorithm makes each decisions.

The most commonly used measures for this are just **Gini Index** and *Cross-entropy*, corresponding to the two supported values for `impurity` argument.




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




## References
[1] An Introduction to Statistical Learning with Applications in R

[2] MLlib - Ensembles, http://spark.apache.org/docs/latest/mllib-ensembles.html

[3] pyspark.mllib package, http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html

[4] MLlib - Data Types, http://spark.apache.org/docs/latest/mllib-data-types.html

## License
Please note this repostory is under the Creative Commons Attribution-ShareAlike License[https://creativecommons.org/licenses/by-sa/3.0/].