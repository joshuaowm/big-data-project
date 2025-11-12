import os
import sys
from pyspark.sql import SparkSession
import logging
from pyspark.sql.functions import (
    col, udf, array, min as spark_min, max as spark_max, 
    avg, stddev, count, lit, expr, row_number, broadcast
)
from pyspark.sql.types import DoubleType, StructType, StructField, ArrayType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sparkmeasure import TaskMetrics
import argparse
from pathlib import Path
from datetime import datetime


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
        .config("spark.hadoop.fs.s3a.connection.timeout", "50000") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60000") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "30000000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
    )

    if args.enable_stage_metrics:
        builder = builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", args.event_log_dir
        )
        logger.info("Stage metrics enabled")

    session = builder.getOrCreate()
    logger.info(f"Spark session created successfully : executors={args.num_executors}, cores={args.executor_cores}, memory={args.executor_memory}g, fraction={args.sample_fraction}")

    return session

# Configuration
train_files = "s3a://ubs-datasets/FRACTAL/data/train/"
valid_files = "s3a://ubs-datasets/FRACTAL/data/valid/"
test_files = "s3a://ubs-datasets/FRACTAL/data/test/"
default_parq_file = "s3a://ubs-datasets/FRACTAL/data/test/TEST-1176_6137-009200000.parquet"

logger = logging.getLogger(__name__)


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

def load_sample(spark, path, fraction):
    logger.info(f"Loading data from {path} with fraction={fraction}")    
    
    # Features of interest to load
    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]    
    
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    
    uri = sc._jvm.java.net.URI(path)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
    file_path = sc._jvm.org.apache.hadoop.fs.Path(path)
    
    all_files = [
        str(f.getPath()) for f in fs.listStatus(file_path)
        if str(f.getPath()).endswith(".parquet")
    ]
    
    num_files = max(1, int(len(all_files) * fraction))
    selected_files = sorted(all_files)[:num_files]
    
    logger.info(f"Loading {num_files}/{len(all_files)} files ({fraction*100:.1f}%)")
    
    df = spark.read.parquet(*selected_files).select(*cols)
    row_count = df.count()
    
    if row_count == 0:
        raise ValueError(f"No data loaded from {path}. Check data path and fraction.")
    
    num_partitions = df.rdd.getNumPartitions()
    logger.info(f"Loaded {row_count} rows, partitions: {num_partitions}")
    return df


def normalize_height(df):
    """
    Normalize height by subtracting the global minimum z value.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        DataFrame with 'z' column
    
    Returns:
    --------
    df : pyspark.sql.DataFrame
        DataFrame with additional 'z_normalized' column
    """
    # Single pass to get min/max
    stats = df.agg(
        spark_min("z").alias("min_z"),
        spark_max("z").alias("max_z")
    ).collect()[0]
    
    min_z = stats["min_z"]
    max_z = stats["max_z"]
    
    print(f"Z-value range: {min_z:.2f} to {max_z:.2f}")
    
    # Normalize to [0, 1] range for better ML performance
    df = df.withColumn("z_normalized", (col("z") - lit(min_z)) / (lit(max_z) - lit(min_z)))
    df = df.withColumn("height", col("z") - lit(min_z))
    
    return df

def compute_spatial_grid_features(df, grid_size=2.0):
    """
    Compute neighbor features using spatial grid aggregation.
    This is fully distributed and leverages PySpark's aggregation engine.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        DataFrame with x, y, z columns
    grid_size : float
        Size of spatial grid cells (in meters)
    
    Returns:
    --------
    df : pyspark.sql.DataFrame
        DataFrame with additional spatial statistics columns
    """
    print(f"Computing spatial grid features with grid_size={grid_size}m...")
    
    # Create spatial grid indices
    df = df.withColumn("grid_x", (col("x") / lit(grid_size)).cast("int"))
    df = df.withColumn("grid_y", (col("y") / lit(grid_size)).cast("int"))
    
    # Compute grid-level statistics (distributed aggregation)
    grid_stats = df.groupBy("grid_x", "grid_y").agg(
        count("*").alias("grid_point_count"),
        avg("z").alias("grid_z_mean"),
        stddev("z").alias("grid_z_std"),
        spark_min("z").alias("grid_z_min"),
        spark_max("z").alias("grid_z_max"),
        avg("Intensity").alias("grid_intensity_mean"),
        avg("Red").alias("grid_red_mean"),
        avg("Green").alias("grid_green_mean"),
        avg("Blue").alias("grid_blue_mean")
    )
    
    # Fill nulls in stddev (when only 1 point in grid)
    grid_stats = grid_stats.fillna(0.0, subset=["grid_z_std"])
    
    # Broadcast join to avoid expensive shuffling (grid stats dataframe is smaller than points dataframe)
    df_with_grid = df.join(broadcast(grid_stats), ["grid_x", "grid_y"], "left")
    
    # Compute relative features
    df_with_grid = df_with_grid.withColumn(
        "z_relative_to_grid", 
        col("z") - col("grid_z_mean")
    )
    
    df_with_grid = df_with_grid.withColumn(
        "z_range_in_grid",
        col("grid_z_max") - col("grid_z_min")
    )
    
    print("Spatial grid features computed successfully!")
    return df_with_grid

def compute_spectral_features(df):
    """
    Compute spectral indices and color-based features.
    Pure SQL transformations - very efficient in Spark.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        DataFrame with Red, Green, Blue, Infrared columns
    
    Returns:
    --------
    df : pyspark.sql.DataFrame
        DataFrame with additional spectral features
    """
    print("Computing spectral features...")
    
    # NDVI (Normalized Difference Vegetation Index)
    df = df.withColumn(
        "ndvi",
        expr("CASE WHEN (Infrared + Red) = 0 THEN 0 " +
             "ELSE (Infrared - Red) / (Infrared + Red) END")
    )
    
    # ExG (Excess Green Index)
    df = df.withColumn(
        "exg",
        expr("2 * Green - Red - Blue")
    )
    
    # Color intensity
    df = df.withColumn(
        "color_intensity",
        expr("(Red + Green + Blue) / 3.0")
    )
    
    # Normalized colors
    total_rgb = col("Red") + col("Green") + col("Blue") + lit(1e-6)
    df = df.withColumn("red_norm", col("Red") / total_rgb)
    df = df.withColumn("green_norm", col("Green") / total_rgb)
    df = df.withColumn("blue_norm", col("Blue") / total_rgb)
    
    # NIR ratio
    df = df.withColumn(
        "nir_ratio",
        col("Infrared") / (col("color_intensity") + lit(1e-6))
    )
    
    print("Spectral features computed successfully!")
    return df

def prepare_features(df, grid_size=2.0):
    """
    Orchestrate all feature engineering steps.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        Raw DataFrame
    grid_size : float
        Spatial grid size
    
    Returns:
    --------
    df : pyspark.sql.DataFrame
        DataFrame with all engineered features
    """
        
    # Extract x, y, z from xyz array
    df = df \
        .withColumn("x", col("xyz")[0]) \
        .withColumn("y", col("xyz")[1]) \
        .withColumn("z", col("xyz")[2])
    
    # Height normalization
    print("\n" + "="*50)
    print("Height Normalization")
    print("="*50)
    df = normalize_height(df)
    
    # Spectral features (fast, pure SQL)
    print("\n" + "="*50)
    print("Spectral Features")
    print("="*50)
    df = compute_spectral_features(df)
    
    # Spatial grid features (distributed aggregation)
    print("\n" + "="*50)
    print("Spatial Grid Features")
    print("="*50)
    df = compute_spatial_grid_features(df, grid_size)
        
    # Drop rows with null values
    df = df.na.drop(subset=["Classification"])
    
    # Select only the columns we need for ML
    # This dramatically reduces memory usage and shuffling overhead
    feature_cols = [
        # Label
        "Classification",
        
        # Geometric features
        "x", "y", "z_normalized", "height",
        
        # Raw spectral
        "Intensity", "Red", "Green", "Blue", "Infrared",
        
        # Spectral indices
        "ndvi", "exg", "color_intensity", "nir_ratio",
        "red_norm", "green_norm", "blue_norm",
        
        # Spatial grid features
        "grid_point_count", "grid_z_std", "z_relative_to_grid", "z_range_in_grid",
        "grid_intensity_mean", "grid_red_mean", "grid_green_mean", "grid_blue_mean",
        
    ]
    
    print(f"\nSelecting {len(feature_cols)} essential columns (1 label + {len(feature_cols)-1} features)")
    df = df.select(*feature_cols)
    
    return df
    
def build_ml_pipeline(feature_cols, num_trees=100, max_depth=10):
    """
    Build the ML pipeline with optional cross-validation.
    
    Parameters:
    -----------
    feature_cols : list
        List of feature column names
    num_trees : int
        Number of trees for Random Forest
    max_depth : int
        Maximum depth of trees
    
    Returns:
    --------
    pipeline
    """
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols, 
        outputCol="features",
        handleInvalid="skip"  # Skip rows with invalid values
    )
    
    # Standardize features
    scaler = StandardScaler(
        inputCol="features", 
        outputCol="scaled_features",
        withStd=True,
        withMean=False  # Sparse-friendly
    )
    
    # Random Forest Classifier
    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="Classification",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42,
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    
    return pipeline

def evaluate_model(predictions, spark):
    """
    Comprehensive model evaluation.
    
    Parameters:
    -----------
    predictions : pyspark.sql.DataFrame
        Predictions DataFrame
    spark : SparkSession
        Spark session
    """
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Multiple metrics
    metrics = ["f1", "accuracy", "weightedPrecision", "weightedRecall"]
    
    for metric in metrics:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="Classification",
            predictionCol="prediction",
            metricName=metric
        )
        score = evaluator.evaluate(predictions)
        print(f"{metric.upper()}: {score:.4f}")
    
    # Per-class metrics
    print("\n" + "="*50)
    print("Per-Class Performance")
    print("="*50)
    
    class_metrics = predictions.groupBy("Classification").agg(
        count("*").alias("total"),
        avg((col("prediction") == col("Classification")).cast("double")).alias("accuracy")
    ).orderBy("Classification")
    
    class_metrics.show()

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

    # taskmetrics = TaskMetrics(spark)

    # Create Spark session
    spark = create_spark_session(args)
        
    # taskmetrics = TaskMetrics(spark)
    
    # Load data
    print("\n" + "="*50)
    print("Loading Data")
    print("="*50)
    
    df_train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction)
    df_val = load_sample(spark, f"{args.data_path}/val/", args.sample_fraction)
    df_test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction)
    
    
    train_count = df_train.count()
    val_count = df_val.count()
    test_count = df_test.count()
    num_partitions = df_train.rdd.getNumPartitions()
    rows_per_partition = train_count / num_partitions if num_partitions > 0 else 0
    logger.info(f"Train: {train_count}, Val: {val_count}, Test: {test_count}, Partitions: {num_partitions}, Rows/partition: {rows_per_partition:.0f}")

    
    # Feature engineering
    print("\n" + "="*50)
    print("Feature Engineering")
    print("="*50)
    
    # Define feature columns
    feature_cols = [
        # Geometric features
        "x", "y", "z_normalized", "height",
        
        # Raw spectral
        "Intensity", "Red", "Green", "Blue", "Infrared",
        
        # Spectral indices
        "ndvi", "exg", "color_intensity", "nir_ratio",
        "red_norm", "green_norm", "blue_norm",
        
        # Spatial grid features
        "grid_point_count", "grid_z_std", "z_relative_to_grid", "z_range_in_grid",
        "grid_intensity_mean", "grid_red_mean", "grid_green_mean", "grid_blue_mean",
    ]
    
    df_train = prepare_features(df_train)
    
    print(f"\nTotal features: {len(feature_cols)}")
    print("Features:", ", ".join(feature_cols))
    
    # Build pipeline
    print("\n" + "="*50)
    print("Building ML Pipeline")
    print("="*50)
    
    pipeline = build_ml_pipeline(feature_cols)
    
    # Train model
    print("\n" + "="*50)
    print("Training Model")
    print("="*50)
    
    # taskmetrics.begin()
    model = pipeline.fit(df_train)
    # taskmetrics.end()
    
    # print("\n" + "="*50)
    # print("Training Statistics")
    # print("="*50)
    # taskmetrics.print_report()
    # print("="*50 + "\n")
    
    # Make predictions
    print("\n" + "="*50)
    print("Making Predictions")
    print("="*50)
    
    # Evaluate model
    
    predictions = model.transform(df_train)
    predictions = predictions.cache()
    evaluate_model(predictions, spark)
    
    # Classification map
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
    
    df_map = spark.createDataFrame(class_map, ["Classification", "Description"])
    print("\n" + "="*50)
    print("Classification Map")
    print("="*50)
    df_map.show(truncate=False)
    
    # Save model if specified
    if args.output_model:
        print(f"\nSaving model to: {args.output_model}")
        model.write().overwrite().save(args.output_model)
    
    # Stop SparkSession
    spark.stop()

if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)
    
    logger.info("Starting Land Cover Classification")
    main(args)
    logger.info("Program completed successfully")