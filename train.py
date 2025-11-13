import argparse
import time
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
    parser.add_argument("--event-log-dir", default="file:///tmp/spark-events", help="Event log directory")
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

def prepare_features(df):
    print("Preparing features...")
    df = df.withColumn("x", col("xyz")[0]).withColumn("y", col("xyz")[1]).withColumn("z", col("xyz")[2])
    df = normalize_height(df)
    df = compute_spectral_features(df)
    df = df.na.drop(subset=["Classification"])
    return df


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main(args):
    overall_start = time.time()
    
    print("\n" + "="*60)
    print("LAND COVER CLASSIFICATION - TRAINING")
    print("="*60)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {args.data_path}")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"Executors: {args.num_executors} x {args.executor_cores} cores x {args.executor_memory}GB")
    print(f"Output path: {args.output_path}")
    print("="*60 + "\n")
    
    spark = create_spark_session(args)
    taskmetrics = TaskMetrics(spark)

    # ===== DATA LOADING =====
    print(f"\n[1/5] Reading data from: {args.data_path}")
    t_start = time.time()
    df_train = read_parquet_from_s3(spark, args.data_path)

    if args.sample_fraction < 1.0:
        print(f"Sampling {args.sample_fraction*100}% of data")
        df_train = df_train.sample(fraction=args.sample_fraction, seed=42)

    df_train = df_train.cache()
    record_count = df_train.count()
    t_load = time.time() - t_start
    print(f"✓ Total records loaded: {record_count:,} ({t_load:.2f}s)\n")

    # ===== FEATURE ENGINEERING =====
    print("[2/5] Feature engineering")
    t_start = time.time()
    df_train = prepare_features(df_train)
    t_features = time.time() - t_start
    print(f"✓ Features prepared ({t_features:.2f}s)\n")

    # ===== PIPELINE SETUP =====
    print("[3/5] Building ML pipeline")
    t_start = time.time()
    feature_cols = ["x", "y", "z_normalized", "Intensity", "Red", "Green", "Blue", "Infrared",
                    "ndvi", "exg", "color_intensity", "nir_ratio"]

    print(f"Using {len(feature_cols)} features")
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
    t_setup = time.time() - t_start
    print(f"✓ Pipeline built ({t_setup:.2f}s)\n")

    # ===== MODEL TRAINING =====
    print("[4/5] Training Random Forest model")
    t_start = time.time()
    taskmetrics.begin()
    model = pipeline.fit(df_train)
    taskmetrics.end()
    t_train = time.time() - t_start
    print(f"✓ Model trained ({t_train:.2f}s = {t_train/60:.2f}min)\n")
    
    print("\n" + "="*60)
    print("TRAINING METRICS (Spark Measure)")
    print("="*60)
    taskmetrics.print_report()
    print("="*60 + "\n")

    # ===== EVALUATION =====
    print("[5/5] Evaluating model")
    t_start = time.time()
    predictions = model.transform(df_train)
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Classification", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)
    t_eval = time.time() - t_start
    print(f"✓ Evaluation complete ({t_eval:.2f}s)\n")
    
    print("\n" + "="*60)
    print(f"TRAINING F1-SCORE: {f1_score:.4f}")
    print("="*60 + "\n")

    # ===== MODEL SAVING =====
    if args.output_path:
        print("Saving model...")
        t_start = time.time()
        model_path = f"{args.output_path}/model"
        print(f"Model path: {model_path}")
        model.write().overwrite().save(model_path)
        t_save = time.time() - t_start
        print(f"✓ Model saved ({t_save:.2f}s)\n")

    # ===== SUMMARY =====
    total_time = time.time() - overall_start
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    print(f"Data Loading:        {t_load:8.2f}s ({t_load/total_time*100:5.1f}%)")
    print(f"Feature Engineering: {t_features:8.2f}s ({t_features/total_time*100:5.1f}%)")
    print(f"Pipeline Setup:      {t_setup:8.2f}s ({t_setup/total_time*100:5.1f}%)")
    print(f"Model Training:      {t_train:8.2f}s ({t_train/total_time*100:5.1f}%)")
    print(f"Evaluation:          {t_eval:8.2f}s ({t_eval/total_time*100:5.1f}%)")
    if args.output_path:
        print(f"Model Saving:        {t_save:8.2f}s ({t_save/total_time*100:5.1f}%)")
    print("-" * 60)
    print(f"TOTAL TIME:          {total_time:8.2f}s = {total_time/60:.2f} minutes")
    print(f"Finished at:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    spark.stop()
    print("Training completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
