# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Machine Learning Pipeline
# MAGIC 
# MAGIC ** What you will learn:**
# MAGIC * How to create a Machine Learning Pipeline.
# MAGIC * How to train a Machine Learning model.
# MAGIC * How to save & read the model.
# MAGIC * How to make predictions with the model.
# MAGIC 
# MAGIC API Docs : https://spark.apache.org/docs/latest/api.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Data
# MAGIC 
# MAGIC The dataset contains bike rental info from 2011 and 2012 in the Capital bikeshare system, plus additional relevant information such as weather.  
# MAGIC 
# MAGIC This dataset is from Fanaee-T and Gama (2013) and is hosted by the <a href="http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset" target="_blank">UCI Machine Learning Repository</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Goal
# MAGIC We want to learn to predict bike rental counts (per hour) from information such as day of the week, weather, month, etc.  
# MAGIC 
# MAGIC Having good predictions of customer demand allows a business or service to prepare and increase supply as needed.  

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Extraction
# MAGIC ## Loading the data
# MAGIC 
# MAGIC We begin by loading our data, which is stored in the CSV format</a>.

# COMMAND ----------

# hadoop fs -ls hdfs://FileStore/tables/
# s3://FileStore/tables/hour.csv

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

# COMMAND ----------

fileName = "/FileStore/tables/hour.csv"

initialDF = (spark.read          # Our DataFrameReader
  .option("header", "true")      # Let Spark know we have a header
  .option("inferSchema", "true") # Infering the schema (it is a small dataset)
  .csv(fileName)                 # Location of our data
  .cache()                       # Mark the DataFrame as cached.
)

initialDF.count()                # Materialize the cache

initialDF.printSchema()

# COMMAND ----------

select3 = initialDF.select("atemp","registered","dteday")

# COMMAND ----------

select3.show()

# COMMAND ----------

initialDF.show()

# COMMAND ----------

# MAGIC %md #### You can also Load Data from
# MAGIC * Different DataFormats 
# MAGIC * * JSON, CSV, Parquet, ORC, Avro, LibSVM, Image
# MAGIC * Different DataStores
# MAGIC * * mySQL , Hbase, Hive, cassandra, MongoDB, Kafka, ElasticSearch
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/sql-data-sources.html

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Understanding
# MAGIC 
# MAGIC According to the <a href="http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset" target="_blank">UCI ML Repository description</a>, we have the following schema:
# MAGIC 
# MAGIC **Feature columns**:
# MAGIC * **dteday**: date
# MAGIC * **season**: season (1:spring, 2:summer, 3:fall, 4:winter)
# MAGIC * **yr**: year (0:2011, 1:2012)
# MAGIC * **mnth**: month (1 to 12)
# MAGIC * **hr**: hour (0 to 23)
# MAGIC * **holiday**: whether the day was a holiday or not
# MAGIC * **weekday**: day of the week
# MAGIC * **workingday**: `1` if the day is neither a weekend nor holiday, otherwise `0`.
# MAGIC * **weathersit**: 
# MAGIC   * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# MAGIC   * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# MAGIC   * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# MAGIC   * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# MAGIC * **temp**: Normalized temperature in Celsius. The values are derived via `(t-t_min)/(t_max-t_min)`, `t_min=-8`, `t_max=+39` (only in hourly scale)
# MAGIC * **atemp**: Normalized feeling temperature in Celsius. The values are derived via `(t-t_min)/(t_max-t_min)`, `t_min=-16`, `t_max=+50` (only in hourly scale)
# MAGIC * **hum**: Normalized humidity. The values are divided to 100 (max)
# MAGIC * **windspeed**: Normalized wind speed. The values are divided to 67 (max)
# MAGIC 
# MAGIC **Label columns**:
# MAGIC * **casual**: count of casual users
# MAGIC * **registered**: count of registered users
# MAGIC * **cnt**: count of total rental bikes including both casual and registered
# MAGIC 
# MAGIC **Extraneous columns**:
# MAGIC * **instant**: record index
# MAGIC 
# MAGIC For example, the first row is a record of hour 0 on January 1, 2011---and apparently, 16 people rented bikes around midnight!

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA : Exploratory DataAnalysis
# MAGIC ###Visualize your data
# MAGIC 
# MAGIC Now that we have preprocessed our features, we can quickly visualize our data to get a sense of whether the features are meaningful.
# MAGIC 
# MAGIC We want to compare bike rental counts versus the hour of the day. 
# MAGIC 
# MAGIC To plot the data:
# MAGIC * Run the cell below
# MAGIC * From the list of plot types, select **Line**.
# MAGIC * Click the **Plot Options...** button.
# MAGIC * By dragging and dropping the fields, set the **Keys** to **hr** and the **Values** to **cnt**.
# MAGIC 
# MAGIC Once you've created the graph, go back and select different **Keys**. For example:
# MAGIC * **cnt** vs. **windspeed**
# MAGIC * **cnt** vs. **month**
# MAGIC * **cnt** vs. **workingday**
# MAGIC * **cnt** vs. **hum**
# MAGIC * **cnt** vs. **temp**
# MAGIC * ...etc.

# COMMAND ----------

display(initialDF)

# COMMAND ----------

# MAGIC %md ### Questions:
# MAGIC   
# MAGIC *   1) At what time Rentals are Low? from 8pm to 7am
# MAGIC *   2) At what time Rentals are High?  5pm, 6pm followed by 8am
# MAGIC *   3) Which Seasons has more Rental ? summer has highest followed by spring and fall? think your definition of season is incorrect: - season : season (1:winter, 2:spring, 3:summer, 4:fall) 
# MAGIC *   4) Which day are rentals High ? ( Working day vs non Working Day) - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.  rentals are high on weekdays
# MAGIC *   5) What are the Categorical features and Numerical features - 
# MAGIC * category features - 
# MAGIC * - season : season (1:winter, 2:spring, 3:summer, 4:fall)
# MAGIC * - holiday : weather day is holiday or not (extracted from [Web Link])
# MAGIC * - weekday : day of the week
# MAGIC * - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# MAGIC * - weathersit : 
# MAGIC *   - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# MAGIC *   - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# MAGIC *   - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# MAGIC *   - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# MAGIC * - yr : year (0: 2011, 1:2012) 
# MAGIC * - mnth : month ( 1 to 12) - can be numerical?
# MAGIC * - hr : hour (0 to 23) - can be numerical?
# MAGIC * Numerical features -
# MAGIC * - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# MAGIC * - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# MAGIC * - hum: Normalized humidity. The values are divided to 100 (max)
# MAGIC * - windspeed: Normalized wind speed. The values are divided to 67 (max)
# MAGIC * - casual: count of casual users
# MAGIC * - registered: count of registered users
# MAGIC * - cnt: count of total rental bikes including both casual and registered 

# COMMAND ----------

# MAGIC %md
# MAGIC A couple of notes:
# MAGIC * Rentals are low during the night, and they peak in the morning (8 am) and in the early evening (5 pm).  
# MAGIC * Rentals are high during the summer and low in winter.
# MAGIC * Rentals are high on working days vs. non-working days

# COMMAND ----------

# MAGIC %md ### Summary Stats 
# MAGIC * Mean, 
# MAGIC * StandardDeviance, 
# MAGIC * Min ,
# MAGIC * Max , 
# MAGIC * Count

# COMMAND ----------

# Summary statistics (Mean, StandardDeviance, Min ,Max , Count) of Numerical columns
# initialDF.describe().show(5)
initialDF.select("atemp","temp","windspeed").describe().show(5,True)

# COMMAND ----------

# MAGIC %md NOTE : 
# MAGIC * Summary statistics (Mean, StandardDeviance, Min ,Max , Count) of Categorical columns
# MAGIC * output for mean, stddev will be null and
# MAGIC * * min & max values are calculated based on ASCII value of categories

# COMMAND ----------

# MAGIC %md ### SQL

# COMMAND ----------

# MAGIC %md * **weathersit**: 
# MAGIC   * 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# MAGIC   * 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# MAGIC   * 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# MAGIC   * 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# MAGIC 
# MAGIC * **season**: season (1:spring, 2:summer, 3:fall, 4:winter)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
def weathersit_values(r):
    if r == 1 : return "Clear"
    elif r == 2 : return "Mist"
    elif r == 3 : return "Light Rain-Snow"
    else: return "Heavy Rain"

def season_values(r):
    if r == 1 : return "spring"
    elif r == 2 : return "summer"
    elif r == 3 : return "fall"
    else: return "winter"
    
    
weathersitTransorm = udf(weathersit_values, StringType())
seasonTransorm = udf(season_values, StringType())

newDF = initialDF.withColumn("weathersit",weathersitTransorm(initialDF.weathersit)).withColumn("season",seasonTransorm(initialDF.season))

newDF.show()

# COMMAND ----------

newDF.filter(newDF.season != "spring").show()
newDF.filter(newDF.season != "spring").count()
newDF.filter(newDF.season.isNull()).show()




# COMMAND ----------

newDF.filter(newDF.weathersit.isNull()).show()

# COMMAND ----------

# MAGIC %md ### SQL Queries 
# MAGIC 
# MAGIC 1. Register the DataFrame as a Table 
# MAGIC 2. Use spark session.sql function 
# MAGIC 3. Returns a new DataFrame

# COMMAND ----------

newDF.createOrReplaceTempView('Bike_Prediction_Table')

# COMMAND ----------

sqlContext.sql("select * from Bike_Prediction_Table where season='summer' and windspeed > 0.4").show(5)

# COMMAND ----------

# maximum booking in each season group in the newDF .
sqlContext.sql('select season, max(cnt) from bike_prediction_table group by season').show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing
# MAGIC 
# MAGIC So what do we need to do to get our data ready for Machine Learning?
# MAGIC 
# MAGIC **Recall our goal**: We want to learn to predict the count of bike rentals (the `cnt` column).  We refer to the count as our target "label".
# MAGIC 
# MAGIC **Features**: What can we use as features to predict the `cnt` label?  
# MAGIC 
# MAGIC All the columns except `cnt`, and a few exceptions:
# MAGIC * `casual` & `registered`
# MAGIC   * The `cnt` column we want to predict equals the sum of the `casual` + `registered` columns.  We will remove the `casual` and `registered` columns from the data to make sure we do not use them to predict `cnt`.  
# MAGIC   
# MAGIC * `season` and the date column `dteday`: We could keep them, but they are well-represented by the other date-related columns like `yr`, `mnth`, and `weekday`.
# MAGIC * `holiday` and `weekday`: These features are highly correlated with the `workingday` column.
# MAGIC * row index column `instant`: This is a useless column to us.

# COMMAND ----------

# MAGIC %md #####  Warning: Make sure you do not "cheat" by using information you will not have when making predictions*

# COMMAND ----------

# MAGIC %md
# MAGIC Let's drop the columns `instant`, `dteday`, `season`, `casual`, `holiday`, `weekday`, and `registered` from our DataFrame and then review our schema:

# COMMAND ----------

preprocessedDF = initialDF.drop("instant", "dteday", "casual", "registered", "holiday", "weekday", "temp")

preprocessedDF.printSchema()

# COMMAND ----------

preprocessedDF.show()

# COMMAND ----------

# MAGIC %md ### Mising Values Check

# COMMAND ----------

# null and empty strings
preprocessedDF.replace('', None).show()
preprocessedDF.replace('', 'null').na.drop(subset='cnt').show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Assesment
# MAGIC 
# MAGIC As it turns out our features can be divided into two types:
# MAGIC  * **Numeric columns:**
# MAGIC    * `mnth`
# MAGIC    * `temp`
# MAGIC    * `hr`
# MAGIC    * `hum`
# MAGIC    * `atemp`
# MAGIC    * `windspeed`
# MAGIC 
# MAGIC * **Categorical Columns:**
# MAGIC   * `yr`
# MAGIC   * `workingday`
# MAGIC   * `weathersit`
# MAGIC   
# MAGIC We could treat both `mnth` and `hr` as categorical but we would lose the temporal relationships (e.g. 2:00 AM comes before 3:00 AM).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC Our final data preparation step will be to split our dataset into separate training and test sets.
# MAGIC 
# MAGIC Using the `randomSplit()` function, we split the data such that 70% of the data is reserved for training and the remaining 30% for testing. 
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset" target="_blank">Dataset.randomSplit()</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit" target="_blank">DataFrame.randomSplit()</a>

# COMMAND ----------

trainDF, testDF = preprocessedDF.randomSplit(
  [0.7, 0.3],  # 70-30 split
  seed=42)     # For reproducibility

print("We have %d training examples and %d test examples." % (trainDF.count(), testDF.count()))
assert (trainDF.count() == 12197)

# COMMAND ----------

#Data processing (Feature Engineering)

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

# MAGIC %md
# MAGIC Before we get started, let's review our current schema:

# COMMAND ----------

trainDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create the first `StringIndexer` for the `workingday` column.
# MAGIC 
# MAGIC After we create it, we can run a sample through the indexer to see how it would affect our `DataFrame`.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator 




workingdayStringIndexer = StringIndexer(
  inputCol="workingday", 
  outputCol="workingdayIndex")


#workingdayEncoder = OneHotEncoderEstimator(inputCols=["workingdayIndex"],
                                 outputCols=["workingdayVector"])

# Just for demonstration purposes, we will use the StringIndexer to fit and
# then transform our training data set just to see how it affects the schema
workingdayStringIndexer.fit(trainDF).transform(trainDF).printSchema()

# COMMAND ----------

seasonStringIndexer = StringIndexer(
  inputCol="season", 
  outputCol="seasonIndex")


#seasonEncoder = OneHotEncoderEstimator(inputCols=["seasonIndex"],
                                 outputCols=["seasonVector"])

seasonStringIndexer.fit(trainDF).transform(trainDF).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Next we will create the `StringIndexer` for the `yr` column and preview its effect.

# COMMAND ----------

yrStringIndexer = StringIndexer(
  inputCol="yr", 
  outputCol="yrIndex")
#yrEncoder = OneHotEncoderEstimator(inputCols=["yrIndex"],
                                 outputCols=["yrVector"])

yrStringIndexer.fit(trainDF).transform(trainDF).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC And then create our last `StringIndexer` for the `weathersit` column.

# COMMAND ----------

weathersitStringIndexer = StringIndexer(
  inputCol="weathersit", 
  outputCol="weathersitIndex")

#weathersitEncoder = OneHotEncoderEstimator(inputCols=["weathersitIndex"],
                                 outputCols=["weathersitVector"])


weathersitStringIndexer.fit(trainDF).transform(trainDF).printSchema()

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
from pyspark.ml.feature import StandardScaler

encoder = OneHotEncoderEstimator(inputCols=["seasonIndex","yrIndex", "workingdayIndex", "weathersitIndex" ],
                                 outputCols=["seasonVector", "yrVector", "workingdayVector", "weathersitVector"])


assemblerInputs  = [
  "mnth",  "hr", "hum", "atemp", "windspeed", # Our numerical features
  #"seasonIndex", "yrIndex", "workingdayIndex", "weathersitIndex"
   "seasonVector", "yrVector", "workingdayVector", "weathersitVector"
]        # Our new categorical features

vectorAssembler = VectorAssembler(
  inputCols=assemblerInputs, 
  outputCol="features_assembler")

scaler = StandardScaler(inputCol="features_assembler", outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forests
# MAGIC 
# MAGIC Random forests and ensembles of decision trees are more powerful than a single decision tree alone.
# MAGIC 
# MAGIC This is also the last step in our pipeline.
# MAGIC 
# MAGIC We will use the `RandomForestRegressor` where we
# MAGIC   * Set `labelCol` to the column that contains our label.
# MAGIC   * Set `seed` to ensure reproducibility.
# MAGIC   * Set `numTrees` to `3` so that we build 3 trees in our random forest.
# MAGIC   * Set `maxDepth` to `10` to control the depth/complexity of the tree.
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.regression.RandomForestRegressor" target="_blank">RandomForestRegressor</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.RandomForestRegressor" target="_blank">RandomForestRegressor</a>

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rfr = (RandomForestRegressor()
      .setLabelCol("cnt") # The column of our label
      .setSeed(27)        # Some seed value for consistency
      .setNumTrees(100)     # A guess at the number of trees
      .setMaxDepth(20)    # A guess at the depth of each tree
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Machine Learning Pipeline
# MAGIC 
# MAGIC Now let's wrap all of these stages into a Pipeline.

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages([
  seasonStringIndexer, # categorize seasons
  workingdayStringIndexer, # categorize workingday
  weathersitStringIndexer, # categorize weathersit
  yrStringIndexer,         # categorize yr
  encoder,
#    seasonEncoder, # categorize seasons
#    workingdayEncoder, # categorize workingday
#    weathersitEncoder, # categorize weathersit
#    yrEncoder,         # categorize yr
  vectorAssembler,         # assemble the feature vector for all columns
  scaler,
  rfr])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model
# MAGIC 
# MAGIC Train the pipeline model to run all the steps in the pipeline.

# COMMAND ----------

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the model
# MAGIC 
# MAGIC Now that we have fitted a model, we can evaluate it.
# MAGIC 
# MAGIC In the case of a random forest, one of the best things to look at is the `featureImportances`:

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressionModel

rfrm = pipelineModel.stages[-1] # The RFRM is in the last stage of the model

#  Zip the list of features with their scores
scores = zip(assemblerInputs, rfrm.featureImportances)

# And pretty print 'em
for x in scores: print("%-15s = %s" % x)

print("-"*80)

# COMMAND ----------

# MAGIC %md
# MAGIC Which features were most important?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Making Predictions
# MAGIC 
# MAGIC Next, apply the trained pipeline model to the test set.

# COMMAND ----------

# Using the model, create our predictions from the test data
predictionsDF = pipelineModel.transform(testDF)

# Reorder the columns for easier interpretation
reorderedDF = predictionsDF.select("cnt", "prediction", "yr", "yrIndex", "mnth", "hr", "workingday", "workingdayIndex", "weathersit", "weathersitIndex", "season", "seasonIndex", "atemp", "hum", "windspeed")

display(reorderedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluate
# MAGIC 
# MAGIC Next, we'll use `RegressionEvaluator` to assess the results. The default regression metric is RMSE.
# MAGIC 
# MAGIC For more information see:
# MAGIC * Scala: <a href="https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.evaluation.RegressionEvaluator" target="_blank">RegressionEvaluator</a>
# MAGIC * Python: <a href="https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.evaluation.RegressionEvaluator" target="_blank">RegressionEvaluator</a>

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator().setLabelCol("cnt")

rmse = evaluator.evaluate(predictionsDF)
r2 = evaluator.setMetricName("r2").evaluate(predictionsDF) 
mse = evaluator.setMetricName("mse").evaluate(predictionsDF) 
mae = evaluator.setMetricName("mae").evaluate(predictionsDF) 

print("Test RMSE = %f" % rmse)
print("R^2 = %f" % r2)
print("MSE = %f" % mse)
print("MAE = %f" % mae)

# COMMAND ----------


