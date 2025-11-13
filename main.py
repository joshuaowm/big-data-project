import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, avg, stddev, count, broadcast
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
from pyspark.sql import Row
import numpy as np

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
# FEATURE ENGINEERING PROCESS
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

def compute_spatial_grid_features(df, grid_size=2.0):
    """
    Compute neighbor features using spatial grid aggregation.
    Fully distributed using PySpark aggregation engine.
    """
    logger.info(f"Computing spatial grid features with grid_size={grid_size}m...")

    # Create spatial grid indices
    df = df.withColumn("grid_x", (col("x") / lit(grid_size)).cast("int"))
    df = df.withColumn("grid_y", (col("y") / lit(grid_size)).cast("int"))

    # Compute grid-level statistics (distributed)
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

    # Fill nulls in stddev (single-point grids)
    grid_stats = grid_stats.fillna(0.0, subset=["grid_z_std"])

    # Broadcast join for efficiency
    df_with_grid = df.join(broadcast(grid_stats), ["grid_x", "grid_y"], "left")

    # Relative features
    df_with_grid = df_with_grid.withColumn("z_relative_to_grid", col("z") - col("grid_z_mean"))
    df_with_grid = df_with_grid.withColumn("z_range_in_grid", col("grid_z_max") - col("grid_z_min"))

    logger.info("Spatial grid features computed successfully!")
    return df_with_grid


def compute_spectral_features(df):
    """
    Compute spectral indices and color-based features.
    """
    logger.info("Computing spectral features...")

    df = df.withColumn("ndvi", (col("Infrared") - col("Red")) / (col("Infrared") + col("Red") + lit(1e-6)))
    df = df.withColumn("exg", 2 * col("Green") - col("Red") - col("Blue"))
    df = df.withColumn("color_intensity", (col("Red") + col("Green") + col("Blue")) / 3.0)

    total_rgb = col("Red") + col("Green") + col("Blue") + lit(1e-6)
    df = df.withColumn("red_norm", col("Red") / total_rgb)
    df = df.withColumn("green_norm", col("Green") / total_rgb)
    df = df.withColumn("blue_norm", col("Blue") / total_rgb)

    df = df.withColumn("nir_ratio", col("Infrared") / (col("color_intensity") + lit(1e-6)))

    logger.info("Spectral features computed successfully!")
    return df


def prepare_features(df, grid_size=2.0):
    """
    Orchestrate all feature engineering steps.
    """
    logger.info("Extracting x, y, z from xyz array")
    df = df.withColumn("x", col("xyz")[0]).withColumn("y", col("xyz")[1]).withColumn("z", col("xyz")[2])

    logger.info("Applying height normalization")
    df = normalize_height(df)

    logger.info("Applying spectral feature computation")
    df = compute_spectral_features(df)

    logger.info("Applying spatial grid feature computation")
    df = compute_spatial_grid_features(df, grid_size)

    df = df.na.drop(subset=["Classification"])

    # Keep only essential features for ML
    feature_cols = [
        "Classification",
        "x", "y", "z_normalized",
        "Intensity", "Red", "Green", "Blue", "Infrared",
        "ndvi", "exg", "color_intensity", "nir_ratio",
        "red_norm", "green_norm", "blue_norm",
        "grid_point_count", "grid_z_std", "z_relative_to_grid", "z_range_in_grid",
        "grid_intensity_mean", "grid_red_mean", "grid_green_mean", "grid_blue_mean"
    ]

    df = df.select(*feature_cols)
    logger.info(f"Selected {len(feature_cols)} columns (1 label + {len(feature_cols)-1} features)")
    return df


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main(args):
    """Main training function"""
    logger.info("Starting main training pipeline")

    spark = create_spark_session(args)
    taskmetrics = TaskMetrics(spark)

    # Read and optionally sample
    df_train = read_parquet_from_s3(spark, args.data_path)
    if args.sample_fraction < 1.0:
        df_train = df_train.sample(fraction=args.sample_fraction, seed=42)

    df_train = df_train.cache()
    logger.info(f"Loaded {df_train.count():,} records")

    # Feature engineering
    df_train = prepare_features(df_train, grid_size=2.0)

    # Assemble features for ML
    feature_cols = df_train.columns
    feature_cols.remove("Classification")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="Classification",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, scaler, rf])

    # Train model
    taskmetrics.begin()
    model = pipeline.fit(df_train)
    taskmetrics.end()
    taskmetrics.print_report()

    # Evaluate
    predictions = model.transform(df_train)
    evaluator = MulticlassClassificationEvaluator(labelCol="Classification", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    logger.info(f"Training F1-Score: {f1_score:.4f}")

    # Save model
    if args.output_path:
        model_path = f"{args.output_path}/model"
        model.write().overwrite().save(model_path)
        logger.info(f"Model saved at {model_path}")

    spark.stop()

if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)
    
    logger.info("Starting Land Cover Classification")
    main(args)
    logger.info("Program completed successfully")