from src.utils.spark_utils import create_spark_session, read_parquet_from_s3

from pyspark.sql.functions import col, udf, array
from pyspark.sql.types import DoubleType, StructType, StructField, ArrayType, FloatType, ShortType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
import argparse
import numpy as np


# To be able to read or write files on S3 from your LOCAL PC you need to launch it this way:
# spark-submit --master local[*] --packages org.apache.hadoop:hadoop-aws:3.3.1 load_parquet_files.py

# for SparkMeasure add to --packages on AWS EMR: ch.cern.sparkmeasure:spark-measure_2.12:0.27
# or if you are using scala 2.13 on your PC:
# add to --packages: ,ch.cern.sparkmeasure:spark-measure_2.13:0.27

# on an AWS cluster launch it directly with :
# spark-submit --master yarn load_parquet_files.py

default_train_files = "s3a://ubs-datasets/FRACTAL/data/train/*"
default_valid_files = "s3a://ubs-datasets/FRACTAL/data/valid/*"
default_test_files = "s3a://ubs-datasets/FRACTAL/data/test/*"

default_parq_file="s3a://ubs-datasets/FRACTAL/data/test/TEST-1176_6137-009200000.parquet"


default_executor_mem = "4g"
default_driver_mem = "4g"

def calculate_ndvi(red, infrared):
    """Calculate NDVI from red and infrared bands."""
    return float((infrared - red) / (infrared + red)) if (infrared + red) != 0 else 0.0

def main(args):
    train_files = args.train
    valid_files = args.valid
    test_files = args.test
    executor_mem = args.executor_mem
    driver_mem = args.driver_mem

    print("\n==============< Program parameters >===============")
    print(f"- Train files: {train_files}")
    print(f"- Valid files: {valid_files}")
    print(f"- Test files: {test_files}")
    print(f"- Executor memory: {executor_mem}")
    print(f"- Driver memory: {driver_mem}")
    print("=================================================\n")

    spark = create_spark_session(app_name="Land Cover Classification", executor_mem=executor_mem, driver_mem=driver_mem)
    
    taskmetrics = TaskMetrics(spark)
    
    parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]
    
    taskmetrics.begin()
    # df_train = spark.read.parquet(default_test_files).select(*parq_cols).cache()
    df_train = read_parquet_from_s3(spark, default_parq_file)
    taskmetrics.end()
    print("\n============< Read Parquet Statistics >============\n")
    taskmetrics.print_report()
    print("\n=====================================================\n")
    
        
    # Register user defined fonction for NDVI calculation
    ndvi_udf = udf(calculate_ndvi, DoubleType())

    # Extract features: x, y, z, intensity, NDVI, RGB, Infrared
    df_train = df_train \
        .withColumn("x", col("xyz")[0]) \
        .withColumn("y", col("xyz")[1]) \
        .withColumn("z", col("xyz")[2]) \
        .withColumn("ndvi", ndvi_udf(col("Red"), col("Infrared")))

    # Show schema and sample data
    df_train.printSchema()
    df_train.show(5, truncate=False)

    # Assemble features
    feature_cols = ["x", "y", "z", "Intensity", "ndvi", "Red", "Green", "Blue", "Infrared"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Standardize features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    # Define the ML pipeline
    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="Classification",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, scaler, rf])

    # Train the model
    model = pipeline.fit(df_train)

    # Make predictions on the training data (for demonstration)
    predictions = model.transform(df_train)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Classification",
        predictionCol="prediction",
        metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)
    print(f"Training F1-Score: {f1_score:.4f}")

    # Classification map for reference
    class_map = [
        (1, "Unclassified"),
        (2, "Soil"),
        (3, "Low vegetation"),
        (4, "Medium vegetation"),
        (5, "High vegetation"),
        (6, "Building"),
        (9, "Water"),
        (17, "Bridge deck"),
        (64, "Perennial surface"),
        (66, "Virtual points"),
        (67, "Miscellaneous - buildings")
    ]

    class_desc = ["Classification", "Description"]
    df_map = spark.createDataFrame(class_map, class_desc)
    print("\n============< Classification Map >============\n")
    df_map.show(truncate=False)

    # Stop SparkSession
    spark.stop()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="PySpark program arguments")
    parser.add_argument("--input", 
                        required=False, help="input file(s)",
                        default=default_parq_file)
    parser.add_argument("--executor-mem",
                        required=False, help="executor memory",
                        default=default_executor_mem)
    parser.add_argument("--driver-mem",
                        required=False, help="driver memory",
                        default=default_driver_mem)
    args = parser.parse_args()
    
    main(args)