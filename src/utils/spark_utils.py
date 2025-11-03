import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession

load_dotenv()

def create_spark_session(app_name: str = "Spark Application") -> SparkSession:
    """
    Initializes and configures a SparkSession for S3 access.

    Args:
        app_name (str): The name for the Spark application.

    Returns:
        SparkSession: The configured SparkSession.
    """

    # Read credentials from environment variables
    s3_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not s3_access_key or not s3_secret_key:
        print("⚠️ WARNING: AWS credentials not found in environment variables. Spark may fail to connect to S3.")

    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", s3_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", s3_secret_key) \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "50000") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "30000000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def read_parquet_from_s3(spark: SparkSession, s3_path: str):
    """
    Reads a Parquet file from a given S3 path into a Spark DataFrame.

    Args:
        spark (SparkSession): The active SparkSession.
        s3_path (str): The full s3a:// path to the Parquet file or directory.

    Returns:
        pyspark.sql.DataFrame: The loaded DataFrame.
    """
    print(f"Reading Parquet file from: {s3_path}")
    return spark.read.parquet(s3_path)