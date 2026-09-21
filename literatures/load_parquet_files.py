# F. Raimbault
# 2025/10/21
# Loading of FRACTAL parquet files and printing of the Classification distribution

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, sum
from sparkmeasure import TaskMetrics

import argparse
import sys

# To be able to read or write files on S3 from your LOCAL PC you need to launch it this way:
# spark-submit --master local[*] --packages org.apache.hadoop:hadoop-aws:3.3.1 load_parquet_files.py
# for SparkMeasure on your PC:
# spark-submit --master local[*] --packages org.apache.hadoop:hadoop-aws:3.3.1,ch.cern.sparkmeasure:spark-measure_2.12:0.27 load_parquet_files.py
# or if you are using scala 2.13 on your PC:
# spark-submit --master local[*] --packages org.apache.hadoop:hadoop-aws:3.3.1,ch.cern.sparkmeasure:spark-measure_2.13:0.27 load_parquet_files.py

# To be able to read or write files on S3 on an AWS cluster launch it this way:
# spark-submit --master yarn load_parquet_files.py
# for SparkMeasure on AWS EMR:
# spark-submit --master local[*] --packages ch.cern.sparkmeasure:spark-measure_2.12:0.27 load_parquet_files.py

# default arguments
    
# one S3 file
default_parq_file="s3a://ubs-datasets/FRACTAL/data/test/TEST-1176_6137-009200000.parquet"
# all test files on S3: "s3a://ubs-datasets/FRACTAL/data/test/*"
# one local file: "./TEST-1176_6137-009200000.parquet"
    
default_executor_mem= "4g"
default_driver_mem= "4g"
    

def main(args):
    
    input_files= args.input
    executor_mem= args.executor_mem
    driver_mem= args.driver_mem
    
    print("\n==============< Program parameters >=============== ")
    print("- input files= {}".format(input_files))
    print("- executor memory= {}".format(executor_mem))
    print("- driver memory= {}".format(driver_mem))
    print("================================================= ")
    
    
    spark = (
        SparkSession.builder
        .appName("Read FRACTAL files")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
        .config("spark.executor.memory", executor_mem)
        .config("spark.driver.memory", driver_mem)
        .getOrCreate()
        )

    spark.sparkContext.setLogLevel("WARN")
    
    taskmetrics = TaskMetrics(spark)
    '''
    # not needed because the schema has been preserved into the parquet file
    parq_schema = StructType([ \
    StructField("X", FloatType(), True), \
    StructField("Y", FloatType(), True), \
    StructField("Z", FloatType(), True), \
    StructField("xyz", ArrayType(DoubleType()), True), \
    StructField("Intensity", ShortType(), True), \
    StructField("ReturnNumber", ShortType(), True), \
    StructField("NumberOfReturns", ShortType(), True), \
    StructField("ScanDirectionFlag", ShortType(), True), \
    StructField("Classification", ShortType(), True), \
    StructField("Synthetic", ShortType(), True), \
    StructField("Keypoint", ShortType(), True), \
    StructField("Withheld", ShortType(), True), \
    StructField("Overlap", ShortType(), True), \
    StructField("ScanAngleRank", FloatType(), True), \
    StructField("UserData", ShortType(), True), \
    StructField("PointSourceId", ShortType(), True), \
    StructField("GpsTime", DoubleType(), True), \
    StructField("ScanChannel", ShortType(), True), \
    StructField("Red", ShortType(), True), \
    StructField("Green", ShortType(), True), \
    StructField("Blue", ShortType(), True), \
    StructField("Infrared", ShortType(), True) \
    ])
    '''
    #the columns I filter at loading time (example)
    # note that the coordinates are not saved into "X","Y","Z" but into the array "xyz"
    parq_cols=["xyz","Intensity","Classification","Red","Green","Blue","Infrared"]

    # not needed because the schema has been preserved into the parquet file
    #df = spark.read.schema(parq_schema).parquet(parq_file).select(*parq_cols)

    taskmetrics.begin()
    
    # read and prune unwanted columns and persist it
    df = spark.read.parquet(input_files).select(*parq_cols)
    df.cache()

    taskmetrics.end()
    print("\n============< read.parquet() statistics >============\n")
    taskmetrics.print_report()
    print("\n=====================================================\n")

    # the schema of with the selected columns
    df.printSchema()

    # print some rows
    df.show(10,truncate=False)

    # view the dataframe as a SQL table named "input_table"
    df.createOrReplaceTempView("input_table")

    # Classification map
    class_map= [
        (1,"Unclassified"),
        (2,"Soil"),
        (3,"Low vegetation"),
        (4,"Medium vegetation"),
        (5,"High vegetation"),
        (6,"Building"),
        (9,"Water"),
        (17,"Bridge deck"),
        (64,"Perennial surface"),
        (66,"Virtual points"),
        (67,"Miscellaneous - buildings")
    ]
    
   # Classification columns
    class_desc= ["Classification","Description"]
    
    # build a DF containing the Classification Map
    df_map= spark.createDataFrame(class_map,class_desc)

    # view the DF
    df_map.show()
    
if __name__ == "__main__":
    
    # parse command line arguments
    
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

    # launch the computation
    
    main(args)
