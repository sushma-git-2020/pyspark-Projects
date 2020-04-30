# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC # Orange Telecom Customer Churn 
# MAGIC Cleaned <a href="https://www.kaggle.com/mnassrib/telecom-churn-datasets/" target="_blank"> Orange Telecom's Churn Dataset</a>.
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
# MAGIC ### Context
# MAGIC "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs."
# MAGIC 
# MAGIC ### Content
# MAGIC The Orange Telecom's Churn Dataset, which consists of cleaned customer activity data (features), along with a churn label specifying whether a customer canceled the subscription, will be used to develop predictive models. Two datasets are made available here: The churn-80 and churn-20 datasets can be downloaded.
# MAGIC 
# MAGIC The two sets are from the same batch, but have been split by an 80/20 ratio. As more data is often desirable for developing ML models, let's use the larger set (that is, churn-80) for training and cross-validation purposes, and the smaller set (that is, churn-20) for final testing and model performance evaluation.
# MAGIC 
# MAGIC ### Inspiration
# MAGIC To explore this type of models and learn more about the subject.
# MAGIC 
# MAGIC <a href="https://www.kaggle.com/mnassrib/telecom-churn-datasets/" target="_blank"> Orange Telecom's Churn Dataset</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Goal
# MAGIC Target is to Retain the Customers who are most likely to leave

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the data
# MAGIC 
# MAGIC We begin by loading our data, which is stored in the CSV format</a>.
# MAGIC 
# MAGIC The churn-80 and churn-20 datasets can be downloaded from the following links, respectively:
# MAGIC 
# MAGIC * https://bml-data.s3.amazonaws.com/churn-bigml-80.csv
# MAGIC * https://bml-data.s3.amazonaws.com/churn-bigml-20.csv

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/telecomData"))

# COMMAND ----------


from pyspark.sql.types import *

schema = StructType([
    StructField("state_code", StringType(), True),
    StructField("account_length", IntegerType(), True),
    StructField("area_code", StringType(), True),
    StructField("international_plan", StringType(), True),
    StructField("voice_mail_plan", StringType(), True),
    StructField("num_voice_mail", DoubleType(), True),
    StructField("total_day_mins", DoubleType(), True),
    StructField("total_day_calls", DoubleType(), True),
    StructField("total_day_charge", DoubleType(), True),
    StructField("total_evening_mins", DoubleType(), True),
    StructField("total_evening_calls", DoubleType(), True),
    StructField("total_evening_charge", DoubleType(), True),
    StructField("total_night_mins", DoubleType(), True),
    StructField("total_night_calls", DoubleType(), True),
    StructField("total_night_charge", DoubleType(), True),
    StructField("total_international_mins", DoubleType(), True),
    StructField("total_international_calls", DoubleType(), True),
    StructField("total_international_charge", DoubleType(), True),
    StructField("total_international_num_calls", DoubleType(), True),
    StructField("churn", StringType(), True)
])


# COMMAND ----------

fileName = "/FileStore/tables/telecomData/churn_bigml_80-bf1a8.csv"

trainSet = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "false") # Infering the schema (it is a small dataset)
  .format("com.databricks.spark.csv")
  .csv(fileName, schema=schema, nullValue='NA') # Enforce the Schema 
  .cache()                       # Mark the DataFrame as cached.
)

trainSet.printSchema()
trainSet.count()                # Materialize the cache

# COMMAND ----------

testDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "false") # Infering the schema (it is a small dataset)
  .format("com.databricks.spark.csv")
  .csv("/FileStore/tables/telecomData/churn_bigml_20-55239.csv", schema=schema, nullValue='NA') # Enforce the Schema 
  .cache()                       # Mark the DataFrame as cached.
)

testDF.printSchema() 
testDF.count()

# COMMAND ----------

# MAGIC %md ## Data Understanding
# MAGIC 
# MAGIC 1. Each Row represents a customer and Each Column contains customer’s attributes. 
# MAGIC 2. Feature : 
# MAGIC       Listed below starting with State,...,Customer service calls with their Schemas.
# MAGIC 3. Label :
# MAGIC       You must predict the value for the 'Churn' column.

# COMMAND ----------

# MAGIC %md #### Telecom Churn Datasets
# MAGIC 
# MAGIC The datasets have the following attributes or features:
# MAGIC 
# MAGIC * State: string
# MAGIC * Account length: integer
# MAGIC * Area code: integer
# MAGIC * International plan: string
# MAGIC * Voice mail plan: string
# MAGIC * Number vmail messages: integer
# MAGIC * Total day minutes: double
# MAGIC * Total day calls: integer
# MAGIC * Total day charge: double
# MAGIC * Total eve minutes: double
# MAGIC * Total eve calls: integer
# MAGIC * Total eve charge: double
# MAGIC * Total night minutes: double
# MAGIC * Total night calls: integer
# MAGIC * Total night charge: double
# MAGIC * Total intl minutes: double
# MAGIC * Total intl calls: integer
# MAGIC * Total intl charge: double
# MAGIC * Customer service calls: integer
# MAGIC * Churn: string
# MAGIC 
# MAGIC The "churn-bigml-20" dataset contains 667 rows (customers) and 20 columns (features).
# MAGIC 
# MAGIC The "Churn" column is the target to predict.

# COMMAND ----------

# MAGIC %md ##Exploratory Data Anaysis:
# MAGIC Question : dataset.skew() ??

# COMMAND ----------

#Data skew done using pyspark functions and display as pie chart and adding to dashboard
from pyspark.sql import functions as f
display(trainSet.select(f.skewness(trainSet['total_international_charge']),f.skewness(trainSet['total_day_charge']),f.skewness(trainSet['total_evening_charge']),f.skewness(trainSet['total_night_charge']), ))


# COMMAND ----------

# MAGIC %md ### Summary statistics
# MAGIC 
# MAGIC The describe() method is a Spark DataFrame's built-in method for statistical processing. It applies summary statistics calculations on all numeric columns

# COMMAND ----------

statsDF = trainSet.describe()
statsDF.show()

# COMMAND ----------

# Let's show some selected columns. 
statsDF.select("summary", "state_code", "account_length", "num_voice_mail", "total_day_mins", "total_day_charge", "total_international_calls", "churn").show()

# COMMAND ----------

# MAGIC %md ### Variable correlation with Label : churn

# COMMAND ----------

# churn is related to the total international call charges:
trainSet.groupBy("churn").sum("total_international_charge").show()

# COMMAND ----------

# churn is related to the total international num of calls:
trainSet.groupBy("churn").sum("total_international_num_calls").show()

# COMMAND ----------

# MAGIC %md ### Spark-SQL :
# MAGIC let's see some related properties of the training set to understand its suitableness for our purposes

# COMMAND ----------

from pyspark.sql.functions import *

# create a temp view for persistence for this session
trainSet.createOrReplaceTempView("UserAccount")

# create a catalog as an interface that can be used to create, drop, alter, or query underlying databases, tables, functions
spark.catalog.cacheTable("UserAccount")

# COMMAND ----------

sqlContext.sql("SELECT churn,SUM(total_international_num_calls) as Total_intl_call FROM UserAccount GROUP BY churn").show()

# COMMAND ----------

display(trainSet)

# COMMAND ----------

# MAGIC %md #### Grouping the data by the churn label 
# MAGIC 
# MAGIC - to calculating the number of instances in each group: 
# MAGIC 
# MAGIC  RESULT :   There are around 6 times more FALSE Churn samples to True Churn samples

# COMMAND ----------

trainSet.groupBy("churn").count().show()

# COMMAND ----------

# MAGIC %md #### NOTE :
# MAGIC 
# MAGIC Target is to retain the customers who are most likely to leave, as opposed to those who are likely to stay or are staying. 
# MAGIC 
# MAGIC This also signifies that we should prepare our training set such that it ensures that our ML model is sensitive to the true churn samples—that is, having churn label true.
# MAGIC 
# MAGIC Training set is highly unbalanced - i.e, stratified sampling

# COMMAND ----------

# MAGIC %md ### Stratified Sampling
# MAGIC 
# MAGIC i.e, we're keeping all Events of the True churn, but downsampling the False churn class to a fraction of 388/2278, which is about 0.1703

# COMMAND ----------

fractions = {"False": 0.1703, "True": 1.0}
churnDF = trainSet.stat.sampleBy("churn", fractions, 12345) # seed : 12345L 

# COMMAND ----------

churnDF.groupBy("churn").count().show()

# COMMAND ----------

sqlContext.sql("SELECT churn,SUM(total_international_num_calls) as Total_intl_call FROM UserAccount GROUP BY churn").show()

# COMMAND ----------

df_total_charge = sqlContext.sql("SELECT churn, "+
                     "SUM(total_day_charge) as day_charge, SUM(total_evening_charge) as evening_charge,"+
                     " SUM(total_night_charge) as night_charge, SUM(total_international_charge) as intl_charge," + 
                     "SUM(total_day_charge) + SUM(total_evening_charge) + SUM(total_night_charge) + SUM(total_international_charge) as Total_charge" + 
                     " FROM UserAccount GROUP BY churn " +
                     "ORDER BY Total_charge DESC")

df_total_charge.show()

# COMMAND ----------

display(df_total_charge)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Total charge to the churn class
# MAGIC Total Charge = how many minutes of day, night, evening, and international voice calls

# COMMAND ----------

df_total_mins = sqlContext.sql("SELECT churn, " + 
                     " SUM(total_day_mins) + SUM(total_evening_mins) + SUM(total_night_mins) + SUM(total_international_mins) as Total_minutes "+
                     "FROM UserAccount GROUP BY churn")
df_total_mins.show()

# COMMAND ----------

display(df_total_mins)

# COMMAND ----------

# MAGIC %md ### <a href="https://databricks.com/blog/2015/06/02/statistical-and-mathematical-functions-with-dataframes-in-spark.html" target="_blank"> Statistics </a>: Covariance and Correlation
# MAGIC 
# MAGIC  Covariance is a measure of how two variables change with respect to each other. 
# MAGIC  * A positive number would mean that there is a tendency that as one variable increases, the other increases as well. 
# MAGIC  * A negative number would mean that as one variable increases, the other variable has a tendency to decrease.
# MAGIC 
# MAGIC Correlation is a normalized measure of covariance that is easier to understand, as it provides quantitative measurements of the statistical dependence between two random variables.

# COMMAND ----------

#showing correlation between various columns
trainSet.stat.corr("total_night_mins","total_night_calls")


# COMMAND ----------

#high correlation
trainSet.stat.corr("total_day_mins","total_day_charge")

# COMMAND ----------

trainSet.stat.corr("total_night_mins","total_night_charge")

# COMMAND ----------

trainSet.stat.corr("total_international_mins","total_international_charge")

# COMMAND ----------

# MAGIC %md #### NOTE : df_total_mins and df_total_charge are a highly correlated feature in this training set - trainSet.
# MAGIC which is not beneficial for our ML model training. 
# MAGIC 
# MAGIC Therefore, it would be better to remove them altogether. Moreover, the following graph shows all possible correlations

# COMMAND ----------

# MAGIC %md #### Question :  Correlation matrix, including all the features ???? 

# COMMAND ----------

#doing a scatter plot
display(trainSet)

# COMMAND ----------

# MAGIC %md # Data Pre-Processing 

# COMMAND ----------

# MAGIC %md ## Data Cleaning
# MAGIC 
# MAGIC 1. Naïve approach : If the TrainDF contains any null values, we completely drop those rows

# COMMAND ----------

# Check for null and empty strings
# churnDF.replace('', None).show()
# churnDF.replace('', 'null').na.drop(subset='cnt').show()

churnDF.na.drop().show()

# COMMAND ----------

# MAGIC %md #### Drop Columns with High Correlation to Label

# COMMAND ----------

#dropping columns with charge since highly correlated to mins
trainDF = churnDF.drop("state_code").drop("area_code").drop("voice_mail_plan").drop("total_day_charge").drop("total_evening_charge").drop("total_night_charge").drop("total_international_charge")

trainDF.printSchema()

# COMMAND ----------

trainDF.show()

# COMMAND ----------

# MAGIC %md ## Feature Assesments
# MAGIC  Need to identify Numerical and Categorical features in training dataset
# MAGIC 
# MAGIC As it turns out our features can be divided into two types:
# MAGIC  * **Numeric columns:**
# MAGIC    * `num_voice_mail`
# MAGIC    * `total_day_mins`
# MAGIC    * `total_day_calls`
# MAGIC    * `total_evening_mins`
# MAGIC    * `total_evening_calls`
# MAGIC    * `total_night_mins`
# MAGIC    * `total_night_calls`
# MAGIC    * `total_night_charge`
# MAGIC    * `total_international_mins`
# MAGIC    * `total_international_calls`
# MAGIC    * `total_international_charge`
# MAGIC    * `total_international_num_calls`
# MAGIC    
# MAGIC * **Categorical Columns:**
# MAGIC   * `international_plan`
# MAGIC   * `account_length`
# MAGIC   * `churn`

# COMMAND ----------

trainDF.columns

# COMMAND ----------

# MAGIC %md #  Data Processing <a href="https://spark.apache.org/docs/latest/ml-features.html" target="_blank"> (Feature Engineering) </a>
# MAGIC GOAL : To extract the most important features that contribute to the classification.
# MAGIC 
# MAGIC Spark ML API needs our data to be converted in a Spark DataFrame format, 
# MAGIC   - DF → Label (Double) and Features (Vector).
# MAGIC 
# MAGIC 
# MAGIC * Label    → churn: True or False
# MAGIC * Features → {"account_length", "iplanIndex", "num_voice_mail", "total_day_mins", "total_day_calls", "total_evening_mins", "total_evening_calls", "total_night_mins", "total_night_calls", "total_international_mins", "total_international_calls", "total_international_num_calls"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## StringIndexer
# MAGIC 
# MAGIC For each of the categorical columns, we are going to create one `StringIndexer` where we
# MAGIC   * Set `inputCol` to something like `international_plan`
# MAGIC   * Set `outputCol` to something like `iplanIndex`
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer" target="_blank">StringIndexer</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer" target="_blank">StringIndexer</a>

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# StringIndexer for categorical columns (OneHotEncoder should be evaluated as well)
ipindexer =  StringIndexer(
    inputCol="international_plan",
    outputCol="iplanIndex")

labelindexer =  StringIndexer(
    inputCol="churn",
    outputCol="label")

# COMMAND ----------

# MAGIC %md
# MAGIC ## VectorAssembler
# MAGIC 
# MAGIC After converted categorical labels into numeric using StringIndexer.
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

featureCols = ["account_length", "iplanIndex", 
                        "num_voice_mail", "total_day_mins", 
                        "total_day_calls", "total_evening_mins", 
                        "total_evening_calls", "total_night_mins", 
                        "total_night_calls", "total_international_mins", 
                        "total_international_calls", "total_international_num_calls"]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# VectorAssembler for training features
assembler =  VectorAssembler(
    inputCols=featureCols,
    outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression 
# MAGIC 
# MAGIC LR is one of the most widely used classifiers to predict a binary response. It is a linear ML method.
# MAGIC 
# MAGIC This is also the last step in our pipeline.
# MAGIC 
# MAGIC We will use the `LogisticRegressor` where we
# MAGIC   * Set `labelCol` to the column that contains our label.
# MAGIC   * Set `seed` to ensure reproducibility.
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.LogisticRegression" target="_blank">LogisticRegression</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression" target="_blank">LogisticRegression</a>

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression 
lr = ( LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
     )

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Pipeline
# MAGIC 
# MAGIC Now let's wrap all of these stages into a Pipeline.

# COMMAND ----------

from pyspark.ml import Pipeline



pipeline =  Pipeline().setStages([
  ipindexer, # categorize internation_plan
  labelindexer, # categorize churn
  assembler, # assemble the feature vector for all columns
  lr])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model
# MAGIC 
# MAGIC Train the pipeline model to run all the steps in the pipeline.

# COMMAND ----------

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the model

# COMMAND ----------

predictionsDF = pipelineModel.transform(testDF)
predictionsDF.printSchema()

# COMMAND ----------

predictionsDF.select("churn", "prediction", "features", "state_code", "account_length", "area_code", "international_plan").show()

# COMMAND ----------

# MAGIC %md # Preparing K-fold Cross Validation and Grid Search: Model tuning

# COMMAND ----------

# MAGIC %md ##Define hyperparameters

# COMMAND ----------

numFolds = 3
MaxIter = 5
RegParam = [0.1, 0.01] # L2 regularization param, set 1.0 with L1 regularization
Tol=1e-8 # for convergence tolerance for iterative algorithms
ElasticNetParam = [0.0, 0.5, 1.0] #Combination of L1 & L2

# COMMAND ----------

# MAGIC %md ## Add parameter Grid

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, RegParam) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, ElasticNetParam)\
    .build()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Evaluator
# MAGIC 
# MAGIC Next, we'll use `BinaryClassificationEvaluator` to assess the results. 
# MAGIC 
# MAGIC since this is a binary classification problem, we define a `BinaryClassificationEvaluator` evaluator.
# MAGIC 
# MAGIC The default metrics are 
# MAGIC * Area under the precision-recall curve and 
# MAGIC * Area under the receiver operating characteristic (ROC) curve
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.evaluation.BinaryClassificationEvaluator" target="_blank">BinaryClassificationEvaluator</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.BinaryClassificationEvaluator" target="_blank">BinaryClassificationEvaluator</a>

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = ( BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction"))

# COMMAND ----------

# MAGIC %md ## Define CrossValidator 
# MAGIC 
# MAGIC for best model selection and makes sure that there's no overfitting.

# COMMAND ----------

cv = ( CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds))

# COMMAND ----------

print("Training model with Logistic Regression algorithm") 
cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the model
# MAGIC 
# MAGIC Now that we have fitted a model, we can evaluate it.

# COMMAND ----------

predictions = cvModel.transform(testDF)
predictions.printSchema()

# COMMAND ----------

predictions.show()

# COMMAND ----------

resultDF = predictions.select("label", "prediction","churn")
resultDF.show(10)

# COMMAND ----------

accuracy = evaluator.evaluate(predictions)
print("Classification accuracy of Logistic Regression: ", accuracy)

# COMMAND ----------

from pyspark.mllib.evaluation import BinaryClassificationMetrics
predictionAndLabels = predictions.select("prediction", "label").rdd

predictionAndLabels = predictions.rdd.map(lambda lp: (float(cvModel.predict(lp.features)), lp.label))

# metrics = BinaryClassificationMetrics(predictionAndLabels)
# println("Area under the precision-recall curve: " + metrics.areaUnderPR)
# println("Area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)

# COMMAND ----------

#Confusion matrices for Logistic Regression
lp = predictions.select("label", "prediction")
counttotal = predictions.count()
correct = lp.filter(predictions.label == predictions.prediction).count()

wrong = lp.filter((predictions.label != predictions.prediction)).count()
ratioWrong = wrong / counttotal
ratioCorrect = correct / counttotal

truep = lp.filter(predictions.prediction == 0.0).filter(predictions.label == predictions.prediction).count() / counttotal

truen = lp.filter(predictions.prediction == 1.0).filter(predictions.label == predictions.prediction).count() / counttotal

falsep = lp.filter(predictions.prediction == 1.0).filter((predictions.label != predictions.prediction)).count() / counttotal

falsen = lp.filter(predictions.prediction == 0.0).filter((predictions.label != predictions.prediction)).count() / counttotal

print("Total Count : ", counttotal)
print("Correct : ",  correct)
print("Wrong: ",  wrong)
print("Ratio wrong: " , ratioWrong)
print("Ratio correct: ",  ratioCorrect)
print("Ratio true positive : ",  truep)
print("Ratio false positive : ",  falsep)

print("Ratio true negative : ",  truen)
print("Ratio false negative : ",  falsen)

# COMMAND ----------

#CV model of LSVC
from pyspark.ml.classification import LinearSVC,  LinearSVCModel
svm = ( LinearSVC()
        .setFeaturesCol("features")
        .setLabelCol("label")
     )
from pyspark.ml import Pipeline

pipeline =  Pipeline().setStages([
  ipindexer, # categorize internation_plan
  labelindexer, # categorize churn
  assembler, # assemble the feature vector for all columns
  svm])
pipelineModel = pipeline.fit(trainDF)

numFolds = 3
MaxIter = [1000]
RegParam = [0.1, 0.01] # L2 regularization param, set 1.0 with L1 regularization
Tol=[1e-8] # for convergence tolerance for iterative algorithms

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid(svm.regParam, RegParam).addGrid(svm.maxIter, MaxIter).addGrid(svm.tol, Tol).build()

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = ( BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction"))
cv = ( CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds))
print("Training model with SVM algorithm") 
cvModel = cv.fit(trainDF)

predictions = cvModel.transform(testDF)
predictions.printSchema()
resultDF = predictions.select("label", "prediction","churn")
resultDF.show(10)

accuracy = evaluator.evaluate(predictions)
print("Classification accuracy: ", accuracy)



# COMMAND ----------

#Accuracy of LSVC
accuracy = evaluator.evaluate(predictions)
print("Classification accuracy: ", accuracy)

bestModel = cvModel.bestModel

#applicable to your model to pull list of all stages
for x in range(len(bestModel.stages)):
  print(bestModel.stages[x])

print(bestModel.stages[3].extractParamMap())


# COMMAND ----------

#CV model of Decision tree
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
dt = ( DecisionTreeClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
     )
from pyspark.ml import Pipeline

pipeline =  Pipeline().setStages([
  ipindexer, # categorize internation_plan
  labelindexer, # categorize churn
  assembler, # assemble the feature vector for all columns
  dt])
pipelineModel = pipeline.fit(trainDF)

numFolds = 3

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [2, 5, 10, 20, 30]).addGrid(dt.maxBins, [10, 20, 40, 80, 100]).build()

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = ( BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction"))
cv = ( CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds))
print("Training model with Decision Tree algorithm") 
cvModel = cv.fit(trainDF)

predictions = cvModel.transform(testDF)
predictions.printSchema()
resultDF = predictions.select("label", "prediction","churn")
resultDF.show(10)

accuracy = evaluator.evaluate(predictions)
print("Classification accuracy: ", accuracy)



# COMMAND ----------

#Accuracy of Decision tree
accuracy = evaluator.evaluate(predictions)
print("DT Classification accuracy: ", accuracy)

print('AUC:', BinaryClassificationMetrics(predictions['label','prediction'].rdd).areaUnderROC)
bestModel = cvModel.bestModel

#applicable to your model to pull list of all stages
for x in range(len(bestModel.stages)):
  print(bestModel.stages[x])

print(bestModel.stages[3].extractParamMap())

# COMMAND ----------

#CV model of Random Forest Classifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
rf = ( RandomForestClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
     )
from pyspark.ml import Pipeline

pipeline =  Pipeline().setStages([
  ipindexer, # categorize internation_plan
  labelindexer, # categorize churn
  assembler, # assemble the feature vector for all columns
  rf])
pipelineModel = pipeline.fit(trainDF)

numFolds = 3

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 5, 10, 20, 30]).addGrid(rf.maxBins, [10, 20, 40, 80, 100]).addGrid(rf.numTrees, [10, 30,100]).build()

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = ( BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction"))
cv = ( CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds))
print("Training model with Random Forest Tree algorithm") 
cvModel = cv.fit(trainDF)

predictions = cvModel.transform(testDF)
predictions.printSchema()
resultDF = predictions.select("label", "prediction","churn")
resultDF.show(10)

accuracy = evaluator.evaluate(predictions)
print("Classification accuracy: ", accuracy)


# COMMAND ----------

#Accuracy and picking best model of Random Forest Classifier
accuracy = evaluator.evaluate(predictions)
print("RF Classification accuracy: ", accuracy)

print('AUC:', BinaryClassificationMetrics(predictions['label','prediction'].rdd).areaUnderROC)
bestModel = cvModel.bestModel

#applicable to your model to pull list of all stages
for x in range(len(bestModel.stages)):
  print(bestModel.stages[x])

print(bestModel.stages[3].extractParamMap())

# COMMAND ----------

#Confusion matrix of Random Forest Classifier
lp = predictions.select("label", "prediction")
counttotal = predictions.count()
correct = lp.filter(predictions.label == predictions.prediction).count()

wrong = lp.filter((predictions.label != predictions.prediction)).count()
ratioWrong = wrong / counttotal
ratioCorrect = correct / counttotal

truep = lp.filter(predictions.prediction == 0.0).filter(predictions.label == predictions.prediction).count() / counttotal

truen = lp.filter(predictions.prediction == 1.0).filter(predictions.label == predictions.prediction).count() / counttotal

falsep = lp.filter(predictions.prediction == 1.0).filter((predictions.label != predictions.prediction)).count() / counttotal

falsen = lp.filter(predictions.prediction == 0.0).filter((predictions.label != predictions.prediction)).count() / counttotal

print("Total Count : ", counttotal)
print("Correct : ",  correct)
print("Wrong: ",  wrong)
print("Ratio wrong: " , ratioWrong)
print("Ratio correct: ",  ratioCorrect)
print("Ratio true positive : ",  truep)
print("Ratio false positive : ",  falsep)
print("Ratio true negative : ",  truen)
print("Ratio false negative : ",  falsen)

# COMMAND ----------

#Learning to plot using matplotlib


# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)
y2 = y + 0.1 * np.random.normal(size=x.shape)

fig, ax = plt.subplots()
ax.plot(x, y, 'k--')
ax.plot(x, y2, 'ro')

# set ticks and tick labels
ax.set_xlim((0, 2*np.pi))
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels(['0', '$\pi$','2$\pi$'])
ax.set_ylim((-1.5, 1.5))
ax.set_yticks([-1, 0, 1])

# Only draw spine between the y-ticks
ax.spines['left'].set_bounds(-1, 1)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# COMMAND ----------

display(fig)

