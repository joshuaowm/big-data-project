import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, avg, stddev, min as spark_min, max as spark_max, broadcast, expr
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    parser.add_argument("--enable-stage-metrics", action="store_true", help="Enable stage metrics collection")
    parser.add_argument("--event-log-dir", default="/tmp/spark-events", help="Event log directory")
    return parser.parse_args()


def create_spark_session(args):
    """Create and configure Spark session"""
    app_name = args.experiment_name or f"fractal-rf-e{args.executor_memory}g-x{args.num_executors}-f{args.sample_fraction}"
    print(f"Creating Spark session: {app_name}")
    builder = SparkSession.builder.appName(app_name)

    if args.master:
        builder = builder.master(args.master)
        print(f"Spark master: {args.master}")

    builder = (
        builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
               .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
               .config("spark.executor.memory", f"{args.executor_memory}g")
               .config("spark.executor.cores", str(args.executor_cores))
               .config("spark.driver.memory", f"{args.driver_memory}g")
               .config("spark.driver.maxResultSize", "512m")
               .config("spark.executor.instances", str(args.num_executors))
               .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
               .config("spark.sql.shuffle.partitions", str((args.executor_cores * args.num_executors) * 4))
               .config("spark.sql.files.maxPartitionBytes", "268435456")
               .config("spark.sql.adaptive.enabled", "true")
               .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")
    )

    if args.enable_stage_metrics:
        builder = builder.config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", args.event_log_dir)
        print("Stage metrics enabled")

    session = builder.getOrCreate()
    print(f"Spark session created: executors={args.num_executors}, cores={args.executor_cores}, memory={args.executor_memory}g, fraction={args.sample_fraction}")
    return session


def read_parquet_from_s3(spark: SparkSession, s3_path: str):
    print(f"Reading Parquet file from: {s3_path}")
    return spark.read.parquet(s3_path)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def normalize_height(df):
    min_z = df.agg(spark_min("z")).collect()[0][0]
    print(f"Minimum z value: {min_z:.2f}")
    df = df.withColumn("z_normalized", col("z") - min_z)
    return df


def compute_spectral_features(df):
    print("Computing spectral features...")
    df = df.withColumn("ndvi", expr("CASE WHEN (Infrared + Red)=0 THEN 0 ELSE (Infrared - Red)/(Infrared + Red) END"))
    df = df.withColumn("exg", expr("2*Green - Red - Blue"))
    df = df.withColumn("color_intensity", expr("(Red + Green + Blue)/3.0"))
    total_rgb = col("Red") + col("Green") + col("Blue") + lit(1e-6)
    df = df.withColumn("red_norm", col("Red") / total_rgb)
    df = df.withColumn("green_norm", col("Green") / total_rgb)
    df = df.withColumn("blue_norm", col("Blue") / total_rgb)
    df = df.withColumn("nir_ratio", col("Infrared") / (col("color_intensity") + lit(1e-6)))
    return df


def compute_spatial_grid_features(df, grid_size=2.0):
    print(f"Computing spatial grid features with grid_size={grid_size}m...")
    df = df.withColumn("grid_x", (col("x") / lit(grid_size)).cast("int"))
    df = df.withColumn("grid_y", (col("y") / lit(grid_size)).cast("int"))
    
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
    
    grid_stats = grid_stats.fillna(0.0, subset=["grid_z_std"])
    df = df.join(broadcast(grid_stats), ["grid_x", "grid_y"], "left")
    df = df.withColumn("z_relative_to_grid", col("z") - col("grid_z_mean"))
    df = df.withColumn("z_range_in_grid", col("grid_z_max") - col("grid_z_min"))
    return df


def prepare_features(df, grid_size=2.0):
    print("Preparing features...")
    df = df.withColumn("x", col("xyz")[0]).withColumn("y", col("xyz")[1]).withColumn("z", col("xyz")[2])
    df = normalize_height(df)
    df = compute_spectral_features(df)
    df = compute_spatial_grid_features(df, grid_size)
    df = df.na.drop(subset=["Classification"])
    return df


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main(args):
    print("\n" + "="*60)
    print("LAND COVER CLASSIFICATION - TRAINING")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"Executors: {args.num_executors} x {args.executor_cores} cores x {args.executor_memory}GB")
    print(f"Output path: {args.output_path}")
    print("="*60 + "\n")
    
    spark = create_spark_session(args)
    taskmetrics = TaskMetrics(spark)

    print(f"\nReading data from: {args.data_path}")
    df_train = read_parquet_from_s3(spark, args.data_path)

    if args.sample_fraction < 1.0:
        print(f"Sampling {args.sample_fraction*100}% of data")
        df_train = df_train.sample(fraction=args.sample_fraction, seed=42)

    df_train = df_train.cache()
    record_count = df_train.count()
    print(f"Total records loaded: {record_count:,}\n")

    df_train = prepare_features(df_train, grid_size=2.0)

    feature_cols = ["x", "y", "z_normalized", "Intensity", "Red", "Green", "Blue", "Infrared",
                    "ndvi", "exg", "color_intensity", "nir_ratio",
                    "grid_point_count", "grid_z_std", "z_relative_to_grid", "z_range_in_grid",
                    "grid_intensity_mean", "grid_red_mean", "grid_green_mean", "grid_blue_mean"]

    print(f"\nBuilding ML pipeline with {len(feature_cols)} features")
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

    print("\nTraining model...")
    taskmetrics.begin()
    model = pipeline.fit(df_train)
    taskmetrics.end()
    
    print("\n" + "="*60)
    print("TRAINING METRICS")
    print("="*60)
    taskmetrics.print_report()
    print("="*60 + "\n")

    print("Making predictions...")
    predictions = model.transform(df_train)
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Classification", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)
    
    print("\n" + "="*60)
    print(f"TRAINING F1-SCORE: {f1_score:.4f}")
    print("="*60 + "\n")

    if args.output_path:
        model_path = f"{args.output_path}/model"
        print(f"Saving model to: {model_path}")
        model.write().overwrite().save(model_path)
        print("Model saved successfully\n")

    spark.stop()
    print("Training completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
