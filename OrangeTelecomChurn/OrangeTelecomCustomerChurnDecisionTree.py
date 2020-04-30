# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import functions as f
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
#CV model of Decision tree
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#get schema of the data
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
#Read training data and testing data from files
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

testDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "false") # Infering the schema (it is a small dataset)
  .format("com.databricks.spark.csv")
  .csv("/FileStore/tables/telecomData/churn_bigml_20-55239.csv", schema=schema, nullValue='NA') # Enforce the Schema 
  .cache()                       # Mark the DataFrame as cached.
)

testDF.printSchema() 
testDF.count()

#Data skew done using pyspark functions and display as pie chart and adding to dashboard
trainSet.select(f.skewness(trainSet['total_international_charge']),f.skewness(trainSet['total_day_charge']),f.skewness(trainSet['total_evening_charge']),f.skewness(trainSet['total_night_charge']) )

# churn is related to the total international call charges:
trainSet.groupBy("churn").sum("total_international_charge").show()

# churn is related to the total international num of calls:
trainSet.groupBy("churn").sum("total_international_num_calls").show()

#Use sparksql to analyze data
# create a temp view for persistence for this session
trainSet.createOrReplaceTempView("UserAccount")

# create a catalog as an interface that can be used to create, drop, alter, or query underlying databases, tables, functions
spark.catalog.cacheTable("UserAccount")
sqlContext.sql("SELECT churn,SUM(total_international_num_calls) as Total_intl_call FROM UserAccount GROUP BY churn").show()

#figure out how much of data is true or false
trainSet.groupBy("churn").count().show()
#sample data take only 0.17 of false 
#Data is highly unbalanced filtering out data to take only 0.17 of false
fractions = {"False": 0.1703, "True": 1.0}
churnDF = trainSet.stat.sampleBy("churn", fractions, 12345) # seed : 12345L 

#showing correlation between various columns
trainSet.stat.corr("total_night_mins","total_night_calls")
trainSet.stat.corr("total_night_mins","total_night_charge")

# Check for null and empty strings
churnDF.na.drop().show()

#dropping columns with charge since highly correlated to mins
trainDF = churnDF.drop("state_code").drop("area_code").drop("voice_mail_plan").drop("total_day_charge").drop("total_evening_charge").drop("total_night_charge").drop("total_international_charge")

trainDF.printSchema()
trainDF.columns

# StringIndexer for categorical columns (OneHotEncoder should be evaluated as well)
ipindexer =  StringIndexer(
    inputCol="international_plan",
    outputCol="iplanIndex")

labelindexer =  StringIndexer(
    inputCol="churn",
    outputCol="label")
	
featureCols = ["account_length", "iplanIndex", 
                        "num_voice_mail", "total_day_mins", 
                        "total_day_calls", "total_evening_mins", 
                        "total_evening_calls", "total_night_mins", 
                        "total_night_calls", "total_international_mins", 
                        "total_international_calls", "total_international_num_calls"]
						


# VectorAssembler for training features
assembler =  VectorAssembler(
    inputCols=featureCols,
    outputCol="features")

dt = ( DecisionTreeClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
     )


pipeline =  Pipeline().setStages([
  ipindexer, # categorize internation_plan
  labelindexer, # categorize churn
  assembler, # assemble the feature vector for all columns
  dt])

pipelineModel = pipeline.fit(trainDF)
predictionsDF = pipelineModel.transform(testDF)
predictionsDF.printSchema()

predictionsDF.select("churn", "prediction", "features", "state_code", "account_length", "area_code", "international_plan").show()

numFolds = 3
paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [2, 5, 10, 20, 30]).addGrid(dt.maxBins, [10, 20, 40, 80, 100]).build()

evaluator = ( BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction"))

cv = ( CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds))


with open('/dbfs/FileStore/DecisionTreeResults.txt', 'w') as f:  
  print("Training model with Decision Tree  algorithm", file=f)
cvModel = cv.fit(trainDF)

predictions = cvModel.transform(testDF)
predictions.printSchema()
predictions.show()

resultDF = predictions.select("label", "prediction","churn")
resultDF.show(10)

accuracy = evaluator.evaluate(predictions)
with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f:  
  print("Classification accuracy of  Decision Tree  : ", accuracy, file=f)
  
with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f:    
  print('AUC:', BinaryClassificationMetrics(predictions['label','prediction'].rdd).areaUnderROC, file=f)

bestModel = cvModel.bestModel

#applicable to your model to pull list of all stages
for x in range(len(bestModel.stages)):
  print(bestModel.stages[x])

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f:  
  print("Param map of best model: " , bestModel.stages[3].extractParamMap(), file=f)

predictionAndLabels = predictions.select("prediction", "label").rdd
predictionAndLabels = predictions.rdd.map(lambda lp: (float(cvModel.predict(lp.features)), lp.label))	

#Confusion matrices for Decision Tree 
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

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Total Count : ", counttotal, file=f)
  
with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Correct : ",  correct, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Wrong: ",  wrong, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Ratio wrong: " , ratioWrong, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Ratio correct: ",  ratioCorrect, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Ratio true positive : ",  truep, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Ratio false positive : ",  falsep, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Ratio true negative : ",  truen, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'a+') as f: 
  print("Ratio false negative : ",  falsen, file=f)

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'r') as f:
  print(f.readlines())

# COMMAND ----------

with open('/dbfs/FileStore/DecisionTreeResults.txt', 'r') as f:
  print(f.readlines())
