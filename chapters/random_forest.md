## Random Forest

As a fan of greedy algorithm, I would like to start with *random forest* algorithm.

What is the idea of **Random Forest**? 

To put it simple, averaging a set of observations reduces variance. Hence a natural way to reduce the variance and hence increase the prediction accuracy of a decision tree model is to take training sets repeatedly from the population, build a separate tree model using each training set, and average the resulting predictions [1]. This is the idea of **bagging**, a "special case" of random forest. 

Then we may need to subset the predictors. That is, in each split of one tree model, we don't use all the features we have. You may ask WHY since this seems like a 'waste' of resources we have. But let's suppose that there is a very strong predictor in the data, then in the models we produced, most of them will use that strong predictor in the top split and all of these decision trees will look similar, i.e., they're highly correlated. This may effect the reduction in variance and worsen the result. [1] This is why we only use randomly selected features in each split of the tree models. 

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

As we mentioned above, the very unique charactristic of *random forest* is that in each split of the tree model we use a subset of features (predictors) instead of using all of them. Then, how many features should we use in each split? we can set `featureSubsetStrategy="auto"` of course so that the function we called will help us configure automatically, but we may want to tune it in some situations. Decreasing this number will speed up training, but can sometimes impact performance if too low [2].

For the function `RandomForest.trainClassifier` in PySaprk , argument `featureSubsetStrategy` supports“auto” (default), “all”, “sqrt”, “log2”, “onethird”. If “auto” is set, this parameter is set based on numTrees: if numTrees == 1, set to “all”; if numTrees > 1 (forest) set to “sqrt” [3].

Usually, given the number of features is `p`, we use `p/3` features in each model when building a random forest for regression, and use `sqrt(p)` features in each model if a random forest is built for classification [1].



##### Point 5: What is 'gini' --- the measures used to grow the trees (`impurity`)

`impurity` argument helps determine the criterion used for information gain calculation, and in PySpark the supported values are “gini” (recommended) or “entropy” [3]. Since random forest is some kind of *greedy algorithm*, we can say that `impurity` helps determine what is the objective function when the algorithm makes each decisions.

The most commonly used measures for this are just **Gini Index** and *Cross-entropy*, corresponding to the two supported values for `impurity` argument.

