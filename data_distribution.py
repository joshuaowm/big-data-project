#!/usr/bin/env python3
"""
FRACTAL Dataset Land Cover Classification Distribution Script
Analyzes land cover distribution across TEST, VAL, and TRAIN datasets
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit
import random

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("FRACTAL_Distribution_Analysis") \
    .getOrCreate()

# Configuration
S3_BASE_PATH = "s3a://ubs-datasets/FRACTAL/data/"
SAMPLE_SIZE = 30

def get_files_by_prefix(base_path, prefixes=['TEST-', 'VAL-', 'TRAIN-']):
    """
    Get files from S3 bucket and categorize by prefix
    """
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(
        sc._jvm.java.net.URI.create(base_path),
        hadoop_conf
    )
    
    path = sc._jvm.org.apache.hadoop.fs.Path(base_path)
    files_dict = {prefix: [] for prefix in prefixes}
    
    try:
        file_status = fs.listStatus(path)
        for status in file_status:
            file_name = status.getPath().getName()
            for prefix in prefixes:
                if file_name.startswith(prefix):
                    files_dict[prefix].append(base_path + file_name)
                    break
    except Exception as e:
        print(f"Error listing files: {e}")
    
    return files_dict

def sample_files(files_dict, sample_size=30):
    """
    Sample files from each category
    """
    sampled_files = {}
    for prefix, files in files_dict.items():
        if len(files) > sample_size:
            sampled_files[prefix] = random.sample(files, sample_size)
        else:
            sampled_files[prefix] = files
            print(f"Warning: Only {len(files)} files found for {prefix}, using all")
    
    return sampled_files

def analyze_distribution(file_paths, dataset_type):
    """
    Analyze land cover classification distribution for given files
    Files are in Parquet format with 'classification' column
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_type} dataset")
    print(f"{'='*60}")
    print(f"Number of files: {len(file_paths)}")
    
    if not file_paths:
        print(f"No files found for {dataset_type}")
        return None
    
    try:
        # Read parquet files
        df = spark.read.parquet(*file_paths)
        
        # Use 'classification' column
        label_col = 'classification'
        
        if label_col not in df.columns:
            print(f"Error: 'classification' column not found")
            print(f"Available columns: {df.columns}")
            return None
        
        # Calculate distribution
        distribution = df.groupBy(label_col) \
            .agg(count("*").alias("count")) \
            .orderBy(col("count").desc())
        
        # Add percentage
        total = df.count()
        distribution = distribution.withColumn(
            "percentage",
            (col("count") / lit(total) * 100)
        )
        
        print(f"\nTotal samples: {total}")
        print(f"\nLand Cover Distribution for {dataset_type}:")
        distribution.show(truncate=False)
        
        return distribution
        
    except Exception as e:
        print(f"Error reading files: {e}")
        print("Please verify the file format and schema")
        return None

def main():
    """
    Main execution function
    """
    print("Starting FRACTAL Dataset Distribution Analysis")
    print(f"Base Path: {S3_BASE_PATH}")
    print(f"Sample Size: {SAMPLE_SIZE} files per category")
    
    # Step 1: Get all files categorized by prefix
    print("\nStep 1: Listing files from S3...")
    files_dict = get_files_by_prefix(S3_BASE_PATH)
    
    for prefix, files in files_dict.items():
        print(f"{prefix}: {len(files)} files found")
    
    # Step 2: Sample files
    print(f"\nStep 2: Sampling {SAMPLE_SIZE} files from each category...")
    sampled_files = sample_files(files_dict, SAMPLE_SIZE)
    
    # Step 3: Analyze distribution for each dataset type
    print("\nStep 3: Analyzing distributions...")
    
    distributions = {}
    for prefix in ['TEST-', 'VAL-', 'TRAIN-']:
        dataset_type = prefix.replace('-', '')
        if sampled_files.get(prefix):
            distributions[dataset_type] = analyze_distribution(
                sampled_files[prefix], 
                dataset_type
            )
    
    # Step 4: Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    for dataset_type, dist in distributions.items():
        if dist:
            print(f"\n{dataset_type} Dataset:")
            dist.select("*").show(5, truncate=False)
    
    print("\nAnalysis complete!")
    
    # Optional: Save results to S3
    # Uncomment if you want to save results
    # output_path = "s3a://ubs-datasets/FRACTAL/analysis_results/"
    # for dataset_type, dist in distributions.items():
    #     if dist:
    #         dist.write.mode("overwrite").parquet(
    #             f"{output_path}{dataset_type}_distribution.parquet"
    #         )

if __name__ == "__main__":
    main()
    spark.stop()