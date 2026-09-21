import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, BooleanType

from pyspark import SparkContext


def main():
    # S3 credentials come from the environment (AWS_ACCESS_KEY_ID,
    # AWS_SECRET_ACCESS_KEY), never from the code
    spark = SparkSession.builder \
        .appName("Land Cover Classification") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"]) \
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"]) \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "50000") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "30000000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    # Path to the S3 bucket containing Parquet files
    s3_path_train = "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955257.parquet"
    

    # Load Parquet files
    df = spark.read.parquet(s3_path_train)
        
    # df_features = df.select(
    #     col("x").cast(DoubleType()),
    #     col("y").cast(DoubleType()),
    #     col("z").cast(DoubleType()),
    #     col("intensity").cast(DoubleType()),
    #     col("returnnumber").cast(IntegerType()),
    #     col("numberofreturns").cast(IntegerType()),
    #     col("red").cast(IntegerType()),
    #     col("green").cast(IntegerType()),
    #     col("blue").cast(IntegerType()),
    #     col("infrared").cast(IntegerType()),
    #     col("classification").cast(IntegerType()).alias("label")  # Target variable
    # )

    # Show the feature DataFrame schema and sample data
    df.printSchema()
    df.show(5)

    # Release the SparkSession
    spark.stop()
    

if __name__ == '__main__':
    main()