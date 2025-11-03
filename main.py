from src.utils.spark_utils import create_spark_session, read_parquet_from_s3
# You can remove many of the other imports from your original file since 
# they are not used in this connection/read step.

def main():
    # 1. Create the SparkSession using the modular function
    spark = create_spark_session(app_name="Land Cover Classification")
    
    # Path to the S3 bucket containing Parquet files
    s3_path_train = "s3a://ubs-datasets/FRACTAL/data/train/TRAIN-0436_6399-002955257.parquet"
    
    # 2. Load Parquet files using the modular function
    df = read_parquet_from_s3(spark, s3_path_train)
        
    # The rest of your processing logic can go here.
    # The code below is a placeholder from your original script.

    df.printSchema()
    df.show(5)

    # Release the SparkSession
    spark.stop()
    

if __name__ == '__main__':
    main()