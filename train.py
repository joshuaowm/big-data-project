import argparse
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, avg, stddev, min as spark_min, max as spark_max, broadcast, expr
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics
import logging
import os
import json

# Deployment Code
# spark-submit --deploy-mode client --master yarn --packages org.apache.hadoop:hadoop-aws:3.3.1,ch.cern.sparkmeasure:spark-measure_2.12:0.27 train.py --num-executors 16 --executor-cores 2 --executor-memory 8 --driver-memory 4 --data "s3a://ubs-datasets/FRACTAL/data/train" --fraction 0.01 --output "s3a://ubs-datasets/FRACTAL/results"

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
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
    app_name = f"fractal-rf-e{args.executor_memory}g-x{args.num_executors}-f{args.sample_fraction}"
    print(f"Creating Spark session: {app_name}")
    builder = SparkSession.builder.appName(app_name)

    if args.master:
        builder = builder.master(args.master)
        print(f"Spark master: {args.master}")

    builder = (
        builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
               .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
               .config("spark.executor.instances", str(args.num_executors)) # Number of executors
               .config("spark.executor.cores", str(args.executor_cores)) # Number of cores per executor (vCPUs)
               .config("spark.executor.memory", f"{args.executor_memory}g") # Executor memory in GB
               .config("spark.driver.memory", f"{args.driver_memory}g") # Driver memory in GB
               .config("spark.driver.maxResultSize", "512m") # Max result size for driver
               .config("spark.executor.instances", str(args.num_executors))
               .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
               .config("spark.sql.shuffle.partitions", str((args.executor_cores * args.num_executors) * 4)) # 4 partitions per core
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

def load_sample(spark, path, fraction):
    """Load a sample of the data from the specified path"""
    logger.info(f"Sampling {fraction*100}% of data")

    # Features of interest to select
    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()

    uri = sc._jvm.java.net.URI(path)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
    file_path = sc._jvm.org.apache.hadoop.fs.Path(path)

    # List all Parquet files in the directory
    all_files = [
        str(f.getPath())
        for f in fs.listStatus(file_path)
        if str(f.getPath()).endswith(".parquet")
    ]

    num_files = max(1, int(len(all_files) * fraction))
    selected_files = sorted(all_files)[:num_files]

    logger.info(f"Loading {num_files}/{len(all_files)} files ({fraction*100:.1f}%)")

    # Load selected files and select relevant columns
    df = spark.read.parquet(*selected_files).select(*cols)
    row_count = df.count()

    if row_count == 0:
        raise ValueError(f"No data loaded from {path}. Check data path and fraction.")

    num_partitions = df.rdd.getNumPartitions()
    logger.info(f"Loaded {row_count} rows, partitions: {num_partitions}")
    return df

def save_scaling_metrics(metrics, output_base_path):
    """
    Saves the scaling metrics dictionary to a JSON file on the local filesystem.
    The file name is descriptive based on run parameters.
    """
    try:
        # Create a descriptive filename
        file_name = f"metrics_f{metrics['sample_fraction']:.2f}_x{metrics['num_executors']}_e{metrics['executor_cores']}.json"
        
        # Combine base path and file name
        full_path = os.path.join(output_base_path, file_name)
        
        # Ensure the directory exists before attempting to write
        os.makedirs(output_base_path, exist_ok=True)
        
        # Local write only
        with open(full_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved locally to: {full_path}")

    except Exception as e:
        print(f"ERROR: Could not save metrics to JSON: {e}")
        
    return


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def normalize_height(df):
    """Normalize height (z) by subtracting the minimum z value"""
    min_z = df.agg(spark_min("z")).collect()[0][0]
    df = df.withColumn("z_normalized", col("z") - min_z)
    return df


def compute_spectral_features(df):
    """Compute spectral features from RGB and Infrared bands"""
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
    """Prepare features for model training"""
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
    print(f"Sample fraction: {args.sample_fraction*100:.2f}%")
    print(f"Executors: {args.num_executors} x {args.executor_cores} cores x {args.executor_memory}GB")
    print(f"Output path: {args.output_path}")
    print("="*60 + "\n")
    
    spark = create_spark_session(args)
    taskmetrics = TaskMetrics(spark)
    
    # Dictionary to hold all scaling metrics
    metrics = {
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "data_path": args.data_path,
        "sample_fraction": args.sample_fraction,
        "num_executors": args.num_executors,
        "executor_cores": args.executor_cores,
        "executor_memory_gb": args.executor_memory,
        "record_count": 0,
        "f1_score": 0.0,
        "t_load": 0.0,
        "t_features": 0.0,
        "t_train": 0.0,
        "t_total_pipeline": 0.0,
        "t_overall": 0.0,
    }

    # ===== DATA LOADING =====
    print(f"\n[1/5] Reading data from: {args.data_path}")
    t_start = time.time()
    df_train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction)
    df_test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction)
    

    record_count = df_train.count()
    metrics["record_count"] = record_count
    t_load = time.time() - t_start
    metrics["t_load"] = t_load
    print(f"✓ Total records loaded: {record_count:,} ({t_load:.2f}s)\n")

    # ===== FEATURE ENGINEERING =====
    print("[2/5] Feature engineering")
    t_start = time.time()
    df_train = prepare_features(df_train)
    t_features = time.time() - t_start
    metrics["t_features"] = t_features
    print(f"✓ Features prepared ({t_features:.2f}s)\n")

    # ===== PIPELINE SETUP =====
    print("[3/5] Building ML pipeline")
    t_start = time.time()
    feature_cols = ["x", "y", "z_normalized", "Intensity", "Red", "Green", "Blue", "Infrared",
                    "ndvi", "exg", "color_intensity", "nir_ratio"]

    print(f"Using {len(feature_cols)} features : {feature_cols}")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    rf = RandomForestClassifier(
        featuresCol="scaled_features", 
        labelCol="Classification", 
        numTrees=20, # Number of trees in the forest
        maxDepth=6, # To prevent overfitting
        seed=42 # For reproducibility
    )
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    t_setup = time.time() - t_start
    print(f"✓ Pipeline built ({t_setup:.2f}s)\n")

    # ===== MODEL TRAINING =====
    print("[4/5] Training Random Forest model on training data")
    t_start = time.time()
    taskmetrics.begin()
    model = pipeline.fit(df_train)
    taskmetrics.end()
    t_train = time.time() - t_start
    metrics["t_train"] = t_train
    print(f"✓ Model trained ({t_train:.2f}s = {t_train/60:.2f}min)\n")
    
    print("\n" + "="*60)
    print("TRAINING METRICS (Spark Measure)")
    print("="*60)
    taskmetrics.print_report()
    print("="*60 + "\n")

    # ===== EVALUATION =====
    print("[5/5] Evaluating model on test data")
    t_start = time.time()
    predictions = model.transform(df_test)
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Classification", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1_score = evaluator.evaluate(predictions)
    metrics["f1_score"] = f1_score
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
    metrics["t_overall"] = total_time
    metrics["t_total_pipeline"] = t_features + t_train # Core computation time
    
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
    
    # ===== SCALING METRICS OUTPUT (Local Save Only) =====
    if args.output_path:
        save_scaling_metrics(metrics, args.output_path)

    spark.stop()
    print("Training completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
