# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # ALL State Insurance Diversity ML Pipeline
# MAGIC 
# MAGIC ** What you will learn:**
# MAGIC * How to create a Machine Learning Pipeline.
# MAGIC * How to train a Machine Learning model.
# MAGIC * How to save & read the model.
# MAGIC * How to make predictions with the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Data
# MAGIC 
# MAGIC Allstate,  How to improve the claims service for the over 16 million households they protect.
# MAGIC automated methods of predicting the cost, and hence severity, of claims
# MAGIC 
# MAGIC <a href="https://www.kaggle.com/c/allstate-claims-severity" target="_blank">Allstate Claims Severity</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the data
# MAGIC 
# MAGIC We begin by loading our data, which is stored in the CSV format</a>.

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/train.csv"))

# COMMAND ----------

fileName = "/FileStore/tables/train.csv"

trainDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "true") # Infering the schema (it is a small dataset)
  .csv(fileName)                 # Location of our data
  .cache()                       # Mark the DataFrame as cached.
)

trainDF.count()                # Materialize the cache

trainDF.printSchema()

# COMMAND ----------

trainDF.count() 


# COMMAND ----------

testDF = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("com.databricks.spark.csv")
    .load("/FileStore/tables/test.csv")
    .cache()
   )

testDF.printSchema() 

# COMMAND ----------

testDF.count()

# COMMAND ----------

# MAGIC %md ## Data Understanding
# MAGIC 
# MAGIC 1. Each row in this dataset represents an insurance claim. 
# MAGIC 2. Feature : 
# MAGIC     1. Variables prefaced with 'cat' are categorical. 
# MAGIC     2. Variables prefaced with 'cont' are continuous.
# MAGIC 3. Label :
# MAGIC       You must predict the value for the 'loss' column.

# COMMAND ----------

 # Impressive Study -  https://www.kaggle.com/sharmasanthosh/exploratory-study-on-ml-algorithms

# COMMAND ----------

# MAGIC %md ##Exploratory Data Anaysis:
# MAGIC Question : dataset.skew() ?? - see command 19

# COMMAND ----------

# Let's show some selected columns. 
trainDF.select("id", "cat1", "cat2", "cat3", "cont1", "cont2", "cont3", "loss").show()

# COMMAND ----------

#scatter plot showing high correlation of some columns
display(trainDF)

# COMMAND ----------

#bar chart comparing cat1 groups A and B
display(trainDF)

# COMMAND ----------

#showing bar chart categories for cat98
display(trainDF)

# COMMAND ----------

#bar chart showing too many categories for cat116
display(trainDF)

# COMMAND ----------

trainDF.show()

# COMMAND ----------

#  you will see some categorical columns contains too many categories. 
trainDF.select("cat109", "cat110", "cat112", "cat113", "cat116").show()

# COMMAND ----------

#skewness
from pyspark.sql import functions as f
display(trainDF.select(f.skewness(trainDF['cont1']),f.skewness(trainDF['cont2']),f.skewness(trainDF['cont3']),f.skewness(trainDF['cont10']) ))

# COMMAND ----------

#show high correlation observed from scatter plot 
trainDF.stat.corr("cont12","cont11")

# COMMAND ----------

trainDF.stat.corr("cont1","cont9")

# COMMAND ----------

trainDF.stat.corr("cont14","loss")

# COMMAND ----------

trainDF.createOrReplaceTempView("insurance")

spark.sql("SELECT avg(insurance.loss) as AVG_LOSS FROM insurance").show()
spark.sql("SELECT min(insurance.loss) as MIN_LOSS FROM insurance").show()
spark.sql("SELECT max(insurance.loss) as MAX_LOSS FROM insurance").show()

# COMMAND ----------

# MAGIC %md ## Data Cleaning
# MAGIC 
# MAGIC 
# MAGIC 1. Rename Loss to Label
# MAGIC 2. Naïve approach : If the TrainDF contains any null values, we completely drop those rows

# COMMAND ----------

trainSample = 1.0
testSample = 1.0
data = trainDF.withColumnRenamed("loss", "label").sample(False, trainSample)

# COMMAND ----------

print("Preparing data for training model")
DF = data.na.drop()

# COMMAND ----------

DF.count()

# COMMAND ----------

# Null check
if data == DF:
  print("No null values in the DataFrame")
else:
  print("Null values exist in the DataFrame")
  data = DF

# COMMAND ----------

# MAGIC %md ## Feature Assesments
# MAGIC 
# MAGIC  Need to identify Numerical and Categorical features in training dataset

# COMMAND ----------

#common functions in python for feature assessment
def onlyFeatureCols(c):  
  return not(c in ["id", "label"])
   
#Function to remove categorical columns with too many categories
def removeTooManyCategs(c):  
  return not (c in [ "cat109" ,  "cat110" , "cat110" , "cat112" , "cat113" , "cat116"] )
    
# Method identify is Category Column
def isCateg(c):
  return c.startswith('cat')

def categNewCol(c):
   if (isCateg(c) == True) : 
      return 'idx_' + c
   else: 
    return c

def categNewEncoderCol(c):
   if (isCateg(c) == True) : 
      return  c + '_Vector'
   else: 
    return c
  
# Method identify is continuos Column
def isCont(c):
  return c.startswith('cont')
    


# COMMAND ----------

 
data = data.select([c for c in data.columns if removeTooManyCategs(c)])    
data.show()

# COMMAND ----------

# Definitive set of feature columns
featureCols = [categNewCol(c) for c in data.columns]
featureCols = [c for c in featureCols if onlyFeatureCols(c)]
print(featureCols)


# COMMAND ----------

# MAGIC %md ## Split the Training Data

# COMMAND ----------

#getting much smaller sample since cv fails with large samples should be commented out in real scenario
newData = data.sample(False, 0.15,42)
#separating trainging data and validation data 
trainingData, validationData = newData.randomSplit(
  [0.75, 0.25],  # 75-25 split
  seed=12345)     # For reproducibility


trainingData.cache
validationData.cache

# COMMAND ----------

trainingData.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## StringIndexer
# MAGIC 
# MAGIC For each of the categorical columns, we are going to create one `StringIndexer` where we
# MAGIC   * Set `inputCol` to something like `weathersit`
# MAGIC   * Set `outputCol` to something like `weathersitIndex`
# MAGIC 
# MAGIC This will have the effect of treating a value like `weathersit` not as number 1 through 4, but rather four categories: **light**, **mist**, **medium** & **heavy**, for example.
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer" target="_blank">StringIndexer</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer" target="_blank">StringIndexer</a>

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

cat_columns = filter(isCateg, trainingData.columns)

indexers = [StringIndexer(inputCol=cat, outputCol=categNewCol(cat)).fit(trainingData) for cat in cat_columns ]

print(indexers)


# COMMAND ----------

from pyspark.ml.feature import OneHotEncoderEstimator 

cat_columns = filter(isCateg, trainingData.columns)

encoder = OneHotEncoderEstimator(inputCols=[categNewCol(cat) for cat in cat_columns], outputCols=[categNewEncoderCol(cat) for cat in cat_columns ])

print(encoder)

# COMMAND ----------

# MAGIC %md #### 
# MAGIC Hint : http://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=filter#pyspark.sql.Column.startswith

# COMMAND ----------

# MAGIC %md
# MAGIC ## VectorAssembler
# MAGIC 
# MAGIC The next step is to assemble the feature columns into a single feature vector.
# MAGIC 
# MAGIC To do that we will use the `VectorAssembler` where we
# MAGIC   * Set `inputCols` to the new list of feature columns
# MAGIC   * Set `outputCol` to `features`
# MAGIC   
# MAGIC   
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a>

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
# VectorAssembler for training features
assembler =  VectorAssembler(
    inputCols=featureCols,
    outputCol="features")

# COMMAND ----------

# MAGIC %md ### Create ML Model - LinearRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Machine Learning Pipeline
# MAGIC 
# MAGIC Now let's wrap all of these stages into a Pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model
# MAGIC 
# MAGIC Train the pipeline model to run all the steps in the pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the model
# MAGIC 
# MAGIC Now that we have fitted a model, we can evaluate it.

# COMMAND ----------

# MAGIC %md ##Define hyperparameters

# COMMAND ----------

# MAGIC %md ## Add parameter Grid

# COMMAND ----------

# MAGIC %md ## Preparing K-fold Cross Validation and Grid Search: Model tuning

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = ( LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
     )


# COMMAND ----------

stages=[]
print(stages)
print(assembler)
print(lr)

# COMMAND ----------

from pyspark.ml import Pipeline

stages = []
stages = indexers
stages.append(assembler)
stages.append(lr)
print(stages)




# COMMAND ----------

#predictions using pipeline model
pipeline =  Pipeline().setStages(stages)
#pipelineModel = pipeline.fit(trainingData)
#predictionsDF  = piplelineModel.transform(validationData)

# COMMAND ----------

# Cross Validation using Linear Regression
numFolds = 3  # number of folds for cross-validation
MaxIter = [10, 100]
RegParam = [0.1, 0.01] # L2 regularization param, set 1.0 with L1 regularization
Tol = [1e-8, 1e-4] # for convergence tolerance for iterative algorithms
ElasticNetParam = [0.001,0.01] #Combination of L1 & L2

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#paramGrid = ParamGridBuilder().addGrid(lr.regParam, RegParam).build()

paramGrid = ParamGridBuilder().addGrid(lr.regParam, RegParam).addGrid(lr.maxIter, MaxIter).addGrid(lr.tol, Tol).addGrid(lr.elasticNetParam, ElasticNetParam).build()

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = ( RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction"))
cv = ( CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(numFolds).setParallelism(3))
print("Training model with Linear Regression algorithm") 
cvModel = cv.fit(trainingData)








# COMMAND ----------

trainingData.count()

# COMMAND ----------

predictionsDF = cvModel.transform(validationData)

# COMMAND ----------

resultDF = predictionsDF.select("prediction", "label", "id", "cat1", "cat2", "cat3", "cont1", "cont2", "cont3")
resultDF.show(10)

# COMMAND ----------

#Analyze Results
rmse = evaluator.evaluate(predictionsDF)
r2 = evaluator.setMetricName("r2").evaluate(predictionsDF) 
mse = evaluator.setMetricName("mse").evaluate(predictionsDF) 
mae = evaluator.setMetricName("mae").evaluate(predictionsDF) 
bestModel = cvModel.bestModel
#applicable to your model to pull list of all stages
for x in range(len(bestModel.stages)):
  print(bestModel.stages[x])


print("Test RMSE = %f" % rmse)
print("R^2 = %f" % r2)
print("MSE = %f" % mse)
print("MAE = %f" % mae)
print("cvModel best parameters = %f" %  bestModel.stages[3].extractParamMap())

# COMMAND ----------

#CV using Random Forest Regressor
from pyspark.ml.regression import RandomForestRegressor

rf = (RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setSeed(27))

from pyspark.ml import Pipeline

stages = []
stages = indexers
stages.append(assembler)
stages.append(rf)
print(stages)
pipeline =  Pipeline().setStages(stages)
numFolds = 3  # number of folds for cross-validation

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 5, 10, 20, 30]).addGrid(rf.maxBins, [10, 20, 40, 80, 100]).addGrid(rf.numTrees, [10, 30,100]).build()

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = ( RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction"))
cv = ( CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(numFolds).setParallelism(3))
print("Training model with Random Forest Regression algorithm") 
cvModel = cv.fit(trainingData)
predictionsDF = cvModel.transform(validationData)

rmse = evaluator.evaluate(predictionsDF)
r2 = evaluator.setMetricName("r2").evaluate(predictionsDF) 
mse = evaluator.setMetricName("mse").evaluate(predictionsDF) 
mae = evaluator.setMetricName("mae").evaluate(predictionsDF) 
bestModel = cvModel.bestModel
#applicable to your model to pull list of all stages
for x in range(len(bestModel.stages)):
  print(bestModel.stages[x])


print("Test RMSE = %f" % rmse)
print("R^2 = %f" % r2)
print("MSE = %f" % mse)
print("MAE = %f" % mae)
print("cvModel best parameters = %f" %  bestModel.stages[3].extractParamMap())

