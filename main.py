from src.utils.spark_utils import create_spark_session, read_parquet_from_s3

from pyspark.sql.functions import col, udf, array, min as spark_min
from pyspark.sql.types import DoubleType, StructType, StructField, ArrayType, FloatType, ShortType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
from pyspark.sql.window import Window
import numpy as np
from scipy.spatial import KDTree
import argparse

# To be able to read or write files on S3 from your LOCAL PC you need to launch it this way:
# spark-submit --master local[*] --packages org.apache.hadoop:hadoop-aws:3.3.1 load_parquet_files.py

# for SparkMeasure add to --packages on AWS EMR: ch.cern.sparkmeasure:spark-measure_2.12:0.27
# or if you are using scala 2.13 on your PC:
# add to --packages: ,ch.cern.sparkmeasure:spark-measure_2.13:0.27

# on an AWS cluster launch it directly with :
# spark-submit --master yarn load_parquet_files.py

train_files = "s3a://ubs-datasets/FRACTAL/data/train/*"
valid_files = "s3a://ubs-datasets/FRACTAL/data/valid/*"
test_files = "s3a://ubs-datasets/FRACTAL/data/test/*"

default_parq_file="s3a://ubs-datasets/FRACTAL/data/test/TEST-1176_6137-009200000.parquet"

default_executor_mem = "4g"
default_driver_mem = "4g"

def normalize_height(df):
    """
    Normalize height by subtracting the global minimum z value.
    Simple and fast approach.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        DataFrame with 'z' column
    
    Returns:
    --------
    df : pyspark.sql.DataFrame
        DataFrame with additional 'z_normalized' column
    """
    # Find global minimum z
    min_z = df.agg(spark_min("z")).collect()[0][0]
    print(f"Minimum z value: {min_z:.2f}")
    
    # Subtract minimum from all z values
    df = df.withColumn("z_normalized", col("z") - min_z)
    
    return df

def compute_neighbor_features(df, radius=2.0):
    """
    Compute basic neighbor statistics using KDTree.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        DataFrame with x, y, z columns
    radius : float
        Search radius for neighbors (in meters)
    
    Returns:
    --------
    df : pyspark.sql.DataFrame
        DataFrame with additional columns:
        - neighbor_count: number of neighbors within radius
        - z_std_local: standard deviation of z in neighborhood
    """
    print(f"Computing neighbor features with radius={radius}m...")
    
    # Collect points for KDTree
    points_data = df.select("x", "y", "z").collect()
    points = np.array([[row.x, row.y, row.z] for row in points_data])
    
    print(f"Building KDTree for {len(points)} points...")
    
    # Build KDTree on x,y coordinates only
    xy_coords = points[:, :2]
    tree = KDTree(xy_coords)
    
    # Compute features for each point
    neighbor_counts = []
    z_stds = []
    
    for i, (x, y, z) in enumerate(points):
        # Find neighbors within radius
        indices = tree.query_ball_point([x, y], radius)
        
        # Get neighbor count
        n_neighbors = len(indices)
        neighbor_counts.append(n_neighbors)
        
        # Get z standard deviation in neighborhood
        if n_neighbors > 1:
            z_values = points[indices, 2]
            z_std = float(np.std(z_values))
        else:
            z_std = 0.0
        z_stds.append(z_std)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(points)} points...")
    
    print("KDTree computation complete!")
    
    # Add row IDs for joining
    from pyspark.sql.functions import monotonically_increasing_id
    from pyspark.sql import Row
    
    df_with_id = df.withColumn("row_id", monotonically_increasing_id())
    
    # Create DataFrame with neighbor features
    feature_rows = [
        Row(row_id=i, neighbor_count=nc, z_std_local=zs)
        for i, (nc, zs) in enumerate(zip(neighbor_counts, z_stds))
    ]
    feature_df = df.sparkSession.createDataFrame(feature_rows)
    
    # Join back to original DataFrame
    df_result = df_with_id.join(feature_df, "row_id").drop("row_id")
    
    return df_result

def calculate_ndvi(red, infrared):
    """Calculate NDVI from red and infrared bands."""
    return float((infrared - red) / (infrared + red)) if (infrared + red) != 0 else 0.0

def main(args):
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
    
    # parq_cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]
    
    # taskmetrics.begin()
    # df_train = spark.read.parquet(default_test_files).select(*parq_cols).cache()
    df_train = read_parquet_from_s3(spark, default_parq_file)
    # taskmetrics.end()
    # print("\n============< Read Parquet Statistics >============\n")
    # taskmetrics.print_report()
    # print("\n=====================================================\n")
    
        
    # Register user defined fonction for NDVI calculation
    ndvi_udf = udf(calculate_ndvi, DoubleType())

    # Extract x, y, z from xyz array
    df_train = df_train \
        .withColumn("x", col("xyz")[0]) \
        .withColumn("y", col("xyz")[1]) \
        .withColumn("z", col("xyz")[2])

    # Apply height normalization
    print("\n============< Height Normalization >============")
    df_train = normalize_height(df_train)

    # Apply neighbor feature computation
    print("\n============< Neighbor Features >============")
    df_train = compute_neighbor_features(df_train, radius=2.0)

    # Add ndvi column
    df_train = df_train.withColumn("ndvi", ndvi_udf(col("Red"), col("Infrared")))
    
    # Drop rows with null values in essential columns
    df_train = df_train.na.drop(subset=["x", "y", "z", "Intensity", "ndvi", "Red", "Green", "Blue", "Infrared", "Classification"])
    
    # Assemble features
    feature_cols = ["x", "y", "z", "z_normalized", "Intensity", "ndvi", 
                "Red", "Green", "Blue", "Infrared", 
                "neighbor_count", "z_std_local"]    
    
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