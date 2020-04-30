# Databricks notebook source
from pyspark.sql import functions as f
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

fileName = "/FileStore/tables/train.csv"

trainDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "true") # Infering the schema (it is a small dataset)
  .csv(fileName)                 # Location of our data
  .cache()                       # Mark the DataFrame as cached.
)
trainDF.count()                # Materialize the cache
trainDF.printSchema()

testDF = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("com.databricks.spark.csv")
    .load("/FileStore/tables/test.csv")
    .cache()
   )
testDF.printSchema() 
testDF.count()

#skewness
trainDF.select(f.skewness(trainDF['cont1']),f.skewness(trainDF['cont2']),f.skewness(trainDF['cont3']),f.skewness(trainDF['cont10']) )

#show high correlation observed from scatter plot 
trainDF.stat.corr("cont12","cont11")

trainDF.stat.corr("cont1","cont9")
trainDF.stat.corr("cont14","loss")

trainDF.createOrReplaceTempView("insurance")

spark.sql("SELECT avg(insurance.loss) as AVG_LOSS FROM insurance").show()
spark.sql("SELECT min(insurance.loss) as MIN_LOSS FROM insurance").show()
spark.sql("SELECT max(insurance.loss) as MAX_LOSS FROM insurance").show()

#rename loss to label
trainSample = 1.0
testSample = 1.0
data = trainDF.withColumnRenamed("loss", "label").sample(False, trainSample)

print("Preparing data for training model")
DF = data.na.drop()
DF.count()

# Null check
if data == DF:
  print("No null values in the DataFrame")
else:
  print("Null values exist in the DataFrame")
  data = DF
  
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
    
data = data.select([c for c in data.columns if removeTooManyCategs(c)])    
data.show()

# Definitive set of feature columns
featureCols = [categNewCol(c) for c in data.columns]
featureCols = [c for c in featureCols if onlyFeatureCols(c)]
print(featureCols)

#getting much smaller sample since cv fails with large samples should be commented out in real scenario
#newData = data.sample(False, 0.25,42)
newData = data.sample(False, 0.10,42)
#separating trainging data and validation data 
trainingData, validationData = newData.randomSplit(
  [0.75, 0.25],  # 75-25 split
  seed=12345)     # For reproducibility


trainingData.cache
validationData.cache

trainingData.count()


cat_columns = filter(isCateg, trainingData.columns)

indexers = [StringIndexer(inputCol=cat, outputCol=categNewCol(cat)).fit(trainingData) for cat in cat_columns ]

print(indexers)



cat_columns = filter(isCateg, trainingData.columns)

encoder = OneHotEncoderEstimator(inputCols=[categNewCol(cat) for cat in cat_columns], outputCols=[categNewEncoderCol(cat) for cat in cat_columns ])


# VectorAssembler for training features
assembler =  VectorAssembler(
    inputCols=featureCols,
    outputCol="features")
	

rf = (RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setSeed(27))

stages = []
stages = indexers
stages.append(assembler)
stages.append(rf)
print(stages)

#predictions using pipeline model
pipeline =  Pipeline().setStages(stages)
#pipelineModel = pipeline.fit(trainingData)
#predictionsDF  = piplelineModel.transform(validationData)

# Cross Validation 
numFolds = 3  # number of folds for cross-validation

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [2, 5, 10, 20, 30]).addGrid(rf.numTrees, [10, 30,100]).build()


evaluator = ( RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction"))
cv = ( CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(numFolds).setParallelism(3))
print("Training model with Random Forest") 
cvModel = cv.fit(trainingData)	 

predictionsDF = cvModel.transform(validationData)

resultDF = predictionsDF.select("prediction", "label", "id", "cat1", "cat2", "cat3", "cont1", "cont2", "cont3")
resultDF.show(10)

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


