from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StringType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

# Spark session
spark = SparkSession.builder \
    .appName("RealTimeDataInfrastructureWithML") \
    .getOrCreate()

# Kafka configuration
# Replace with your Kafka details
kafka_bootstrap_servers = "<kafka-broker-ip>:<port>"  

# Replace with your subscribed topic
kafka_topic = "your_topic"  

# Define the schema for the incoming data
schema = StructType().add("name", StringType()).add("age", IntegerType())

# Ingest data ustilizing Spark Streaming
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic) \
    .load()

# Converting 
# binary value -> JSON string -> DataFrame
json_df = kafka_df.selectExpr("CAST(value AS STRING)")
data_df = json_df.select(from_json(json_df.value, schema).alias("data")).select("data.*")

# Insert data processing here
# Ingest real-time data 

# Creating Query to filter data with specific conditions with age example
filtered_data_df = data_df.filter(data_df.age > 25)

# Machine Learning Data Preprocessing 
feature_columns = ['name']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
ml_data_df = assembler.transform(filtered_data_df)

# Training Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="age")

# Hyperparameter Tuning
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = RegressionEvaluator(labelCol="age", predictionCol="prediction", metricName="mse")

cross_validator = CrossValidator(estimator=lr,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator,
                                 numFolds=3)

# Fitting tuned model 
lr_tuned_model = cross_validator.fit(ml_data_df)

# Making predictions 
predictions = lr_tuned_model.transform(ml_data_df)

# Sample predictions
predictions.select("name", "age", "prediction").show()

# Spark Streaming query processing real-time data
query = predictions.writeStream \
    .outputMode("append") \
    .format("console")  # Replace "console" with the desired output sink (e.g., Kafka, HDFS, etc.)
    .start()

# Data Security
# Wait for the query to terminate
query.awaitTermination()

# Spark Monitors
