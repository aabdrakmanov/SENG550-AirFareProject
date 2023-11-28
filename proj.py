import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable



import findspark
findspark.init()

from pyspark import SparkFiles


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, VectorIndexer, StandardScaler, PCA
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
import pandas as pd

spark = SparkSession.builder.appName("Air fare Price predictor").config('spark.driver.memory', '64g').config('spark.executor.memory', '16g').config("spark.sql.execution.arrow.pyspark.enabled", "true").config("spark.executor.extraJavaOptions", "-Xss16M").config("spark.driver.extraJavaOptions", "-Xss16M").getOrCreate()

@F.pandas_udf("int")
def parse_iso8601_duration(str_duration: pd.Series) -> pd.Series:
    return str_duration.apply(lambda duration: (pd.Timedelta(duration).seconds / 60))

url = "./archive/itineraries.csv"
spark.sparkContext.addFile(url)

df = spark.read.csv(url, header=True, inferSchema=True)

segmentsIndexer = StringIndexer(inputCols=["segmentsAirlineCode","segmentsEquipmentDescription", "segmentsCabinCode"], outputCols=["ac","ed", "cc"], handleInvalid='skip')
df = segmentsIndexer.fit(df).transform(df)

segmentsAssembler = VectorAssembler(inputCols=["ac","ed","cc"], outputCol="segmentFeatures", handleInvalid='skip')
df = segmentsAssembler.transform(df)

scaler = StandardScaler(inputCol='segmentFeatures', outputCol='scaledSegFeatures', withMean=True, withStd=True).fit(df)
df = scaler.transform(df)

pca = PCA(k=1, inputCol='scaledSegFeatures', outputCol='pcaFeatures').fit(df)
df = pca.transform(df)

df=df.withColumn("date", F.unix_timestamp("flightDate"))
df=df.withColumn("ns", df.isNonStop.cast('int'))
df=df.withColumn("econ",df.isBasicEconomy.cast('int'))
df=df.withColumn("duration", parse_iso8601_duration(F.col("travelDuration")))

indexer = StringIndexer().setInputCol("startingAirport").setOutputCol("numSA")
model1 = indexer.fit(df)
data = model1.transform(df)

model2 = model1.setInputCol("destinationAirport").setOutputCol("numDA")
data = model2.transform(data)


# Preprocessing: VectorAssembler for feature
assembler = VectorAssembler(inputCols=["date","ns","econ","duration","totalTravelDistance","seatsRemaining","numSA", "numDA", "pcaFeatures"], outputCol="features", handleInvalid='skip')
data = assembler.transform(data)

train_data,test_data = data.randomSplit([0.8, 0.2], seed=42)


#Set regressor, evaluator and cross validator
regressor = GBTRegressor(labelCol="totalFare", featuresCol="features", maxIter=250, maxMemoryInMB=4096)

evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="r2")

paramGrid = ParamGridBuilder().addGrid(regressor.stepSize, [0.01, 0.1, 0.5, 1]).addGrid(regressor.maxDepth, [5,6,7,8,9]).addGrid(regressor.subsamplingRate, [0.01,0.1,0.5,1]).build()
#cv = TrainValidationSplit(estimator=regressor, estimatorParamMaps=paramGrid, evaluator=evaluator)
cv = CrossValidator(estimator = regressor, evaluator=evaluator, estimatorParamMaps = paramGrid, numFolds=3, parallelism=4, seed=99)

model = cv.fit(train_data)


# run model on predictions
predictions = model.transform(test_data)


# Evaluate the model performance
evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print(f"RMSE: {accuracy:.2f}")

evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="var")
accuracy = evaluator.evaluate(predictions)
print(f"var: {accuracy:.2f}")

evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="mse")
accuracy = evaluator.evaluate(predictions)
print(f"mse: {accuracy:.2f}")

evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="mae")
accuracy = evaluator.evaluate(predictions)
print(f"mae: {accuracy:.2f}")

evaluator = RegressionEvaluator(labelCol="totalFare", predictionCol="prediction", metricName="r2")
accuracy = evaluator.evaluate(predictions)
print(f"R^2: {accuracy:.2f}")



model.save("GBT_AirFareModelPCA-FULL")

spark.stop()
