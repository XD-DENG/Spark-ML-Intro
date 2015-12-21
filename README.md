# Spark Machine Learning Practice

In this repo, I would try to introduce some basic usages of PySpark in machine learning. The contents I'm going to cover would be quite simple. I guess it would be helpful for some people since I would cover some questions I encountered myself from the perspective of a person who's used to the machine learning in R or more "normal" ML settings.

Some of the examples are from the official examples given by Spark. But more options will be explored and some comments will be given to make it clearer. Just like I mentioned, even if with some experience of machine learning, there are still some points we may need additional information.

## Random Forest

As a fan of greedy algorithm, I would like to start with random forest algorithm.

What is the idea of *Random Forest*? 

To put it simpel, averaging a set of observations reduces variance (given a set of `n` independent observations Z1,...,Zn, each with variance `sigma^2`, the variance of the mean of the observations is given by `sigma^2`/n). Hence a natural way to reduce the variance and hence increase the prediction accuracy of a machine learning model is to take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions [1]. 

Then we may need to subset the predictors. That is, in each training procedure, we don't use all the features we have. This seems like a 'waste' of resources we have. But let's suppose that there is a very strong predictor in the data, then in the models we produced, most of them will use that strong predictor in the top split and all of these trees will look similar, i.e., they're highly correlated. This may effect the reduction in variance and worsen the result. [1]

Now let's have a look at the example code given by Spark. I commented the points where we may need to note (and details will be given later)

```python
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

# ---!!!!!!---
# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])


# ---!!!!!!---
# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
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

#### LIBSVM data format
When I looked into LIBSVM data file for the first time, I got a little bit confused. But then I found its design is a brilliant idea.

TO-DO

#### How many trees we should have (`numTrees`)

This argument determines how many trees we build in a random forest. Increasing the number of trees will decrease the variance in predictions, and improve the model’s test accuracy. At the same time, training time will increaseroughly linearly in the number of trees.

Personally, I would say 400-500 is a 'safe' choice.

#### How many features to use (`featureSubsetStrategy`)

As we mentioned above, the very unique charactristic of *random forest* is that in each tree we use a subset of all features (predictors). Then how many features should we use in each tree model? we can set `featureSubsetStrategy="auto"` of course, but we may want to tune it in some situations. Decreasing this number will speed up training, but can sometimes impact performance if too low [2].

Usually, given the number of features is `p`, we use `p/3` features in each model when building a random forest for regression, and use `sqrt(p)` features in each model if a random forest is built for classification [1].


#### What is 'gini' --- the measures used to grow the trees
TO-DO



## References
[1] An Introduction to Statistical Learning with Applications in R

[2] MLlib - Ensembles, http://spark.apache.org/docs/latest/mllib-ensembles.html