import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, monotonically_increasing_id, min as spark_min
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
from pyspark.sql import Row
import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(args):
    """Setup logging configuration"""
    safe_dir = Path("/tmp/spark-events")
    safe_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = args.experiment_name or f"fractal-rf-e{args.executor_memory}g-x{args.num_executors}-f{args.sample_fraction}"
    log_file = safe_dir / f"{log_name}_{timestamp}.log"

    command_line = ' '.join(sys.argv)
    with open(log_file, 'w') as f:
        f.write(f"Command: {command_line}\n")
        f.write("=" * 60 + "\n\n")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    for handler in [logging.StreamHandler(), logging.FileHandler(log_file)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.info(f"Logging to: {log_file}")
    return log_file


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment-name", default=None, help="Experiment name for Spark app")
    parser.add_argument("-m", "--master", default=None, help="Spark master URL")
    parser.add_argument("-e", "--executor-memory", type=int, default=8, help="Executor memory in GB")
    parser.add_argument("-d", "--driver-memory", type=int, default=2, help="Driver memory in GB")
    parser.add_argument("-c", "--executor-cores", type=int, default=2, help="Executor cores")
    parser.add_argument("-x", "--num-executors", type=int, default=2, help="Number of executors")
    parser.add_argument("-p", "--data", dest="data_path", default="/opt/spark/work-dir/data/FRACTAL", help="Data path")
    parser.add_argument("-f", "--fraction", dest="sample_fraction", type=float, default=0.1, help="Sample fraction")
    parser.add_argument("-o", "--output", dest="output_path", default="results", help="Output directory path")
    parser.add_argument(
        "--enable-stage-metrics",
        action="store_true",
        help="Enable stage metrics collection",
    )
    parser.add_argument(
        "--event-log-dir",
        default="/opt/spark/spark-events",
        help="Event log directory (used when stage metrics enabled)",
    )
    return parser.parse_args()


def create_spark_session(args):
    """Create and configure Spark session"""
    app_name = args.experiment_name or f"fractal-rf-e{args.executor_memory}g-x{args.num_executors}-f{args.sample_fraction}"
    
    logger.info(f"Creating Spark session: {app_name}")
    builder = SparkSession.builder.appName(app_name)

    if args.master:
        builder = builder.master(args.master)
        logger.info(f"Spark master: {args.master}")

    builder = (
        builder.config(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.executor.memory", f"{args.executor_memory}g")
        .config("spark.executor.cores", str(args.executor_cores))
        .config("spark.driver.memory", f"{args.driver_memory}g")
        .config("spark.driver.maxResultSize", "512m")
        .config("spark.executor.instances", str(args.num_executors))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.shuffle.partitions", str((args.executor_cores * args.num_executors) * 4))
        .config("spark.sql.files.maxPartitionBytes", "268435456")  # 256MB
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")  # 128MB
    )

    if args.enable_stage_metrics:
        builder = builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", args.event_log_dir
        )
        logger.info("Stage metrics enabled")

    session = builder.getOrCreate()
    logger.info(f"Spark session created successfully : executors={args.num_executors}, cores={args.executor_cores}, memory={args.executor_memory}g, fraction={args.sample_fraction}")

    return session


def read_parquet_from_s3(spark: SparkSession, s3_path: str):
    """
    Reads a Parquet file from a given S3 path into a Spark DataFrame.
    Args:
        spark (SparkSession): The active SparkSession.
        s3_path (str): The full s3a:// path to the Parquet file or directory.
    Returns:
        pyspark.sql.DataFrame: The loaded DataFrame.
    """
    logger.info(f"Reading Parquet file from: {s3_path}")
    return spark.read.parquet(s3_path)


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

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
    logger.info(f"Minimum z value: {min_z:.2f}")
    
    # Subtract minimum from all z values
    df = df.withColumn("z_normalized", col("z") - min_z)
    
    return df


# def compute_neighbor_features(df, radius=2.0):
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
    logger.info(f"Computing neighbor features with radius={radius}m...")
    
    # Collect points for KDTree
    points_data = df.select("x", "y", "z").collect()
    points = np.array([[row.x, row.y, row.z] for row in points_data])
    
    logger.info(f"Building KDTree for {len(points)} points...")
    
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
            logger.info(f"Processed {i + 1}/{len(points)} points...")
    
    logger.info("KDTree computation complete!")
    
    # Add row IDs for joining
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


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main(args):
    """Main training function"""
    logger.info("\n==============< Program Parameters >===============")
    logger.info(f"- Data path: {args.data_path}")
    logger.info(f"- Sample fraction: {args.sample_fraction}")
    logger.info(f"- Executor memory: {args.executor_memory}g")
    logger.info(f"- Driver memory: {args.driver_memory}g")
    logger.info(f"- Executor cores: {args.executor_cores}")
    logger.info(f"- Number of executors: {args.num_executors}")
    logger.info(f"- Output path: {args.output_path}")
    logger.info("=================================================\n")

    spark = create_spark_session(args)
    
    taskmetrics = TaskMetrics(spark)
    
    # Read data
    logger.info(f"Reading data from: {args.data_path}")
    df_train = read_parquet_from_s3(spark, args.data_path)
    
    # Sample data if fraction < 1.0
    if args.sample_fraction < 1.0:
        logger.info(f"Sampling {args.sample_fraction * 100}% of the data")
        df_train = df_train.sample(fraction=args.sample_fraction, seed=42)
    
    # Cache the dataframe
    df_train = df_train.cache()
    record_count = df_train.count()
    logger.info(f"Total records loaded: {record_count:,}")
    
    # Register user defined function for NDVI calculation
    ndvi_udf = udf(calculate_ndvi, DoubleType())

    # Extract x, y, z from xyz array
    df_train = df_train \
        .withColumn("x", col("xyz")[0]) \
        .withColumn("y", col("xyz")[1]) \
        .withColumn("z", col("xyz")[2])

    # Apply height normalization
    logger.info("\n============< Height Normalization >============")
    df_train = normalize_height(df_train)

    # Apply neighbor feature computation
    # logger.info("\n============< Neighbor Features >============")
    # df_train = compute_neighbor_features(df_train, radius=2.0)

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
    logger.info("\n============< Training Model >============")
    taskmetrics.begin()
    model = pipeline.fit(df_train)
    taskmetrics.end()
    
    logger.info("\n============< Training Statistics >============")
    taskmetrics.print_report()
    logger.info("=====================================================\n")

    # Make predictions on the training data
    logger.info("Making predictions...")
    predictions = model.transform(df_train)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Classification",
        predictionCol="prediction",
        metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)
    logger.info(f"Training F1-Score: {f1_score:.4f}")

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
    logger.info("\n============< Classification Map >============")
    df_map.show(truncate=False)

    # Save model if output path is specified
    if args.output_path:
        model_path = f"{args.output_path}/model"
        logger.info(f"Saving model to: {model_path}")
        model.write().overwrite().save(model_path)

    # Stop SparkSession
    spark.stop()


if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)
    
    logger.info("Starting Land Cover Classification")
    main(args)
    logger.info("Program completed successfully")
