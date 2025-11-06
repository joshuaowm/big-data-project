from src.utils.spark_utils import create_spark_session, read_parquet_from_s3

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

# Configuration
train_files = "s3a://ubs-datasets/FRACTAL/data/train/*"
valid_files = "s3a://ubs-datasets/FRACTAL/data/valid/*"
test_files = "s3a://ubs-datasets/FRACTAL/data/test/*"
default_parq_file = "s3a://ubs-datasets/FRACTAL/data/test/TEST-1176_6137-009200000.parquet"

default_executor_mem = "4g"
default_driver_mem = "4g"

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
    
def build_ml_pipeline(feature_cols, num_trees=100, max_depth=10, use_cv=False):
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
    use_cv : bool
        Whether to use cross-validation
    
    Returns:
    --------
    pipeline or cross-validator
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
        maxBins=64,  # Increased for better splits
        minInstancesPerNode=10,  # Prevent overfitting
        subsamplingRate=0.8,  # Bootstrap sampling
        seed=42,
        featureSubsetStrategy="sqrt"  # Standard for classification
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    
    if use_cv:
        # Parameter grid for cross-validation
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100, 150]) \
            .addGrid(rf.maxDepth, [8, 10, 12]) \
            .build()
        
        # Cross-validator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="Classification",
            predictionCol="prediction",
            metricName="f1"
        )
        
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=4  # Parallel fold training
        )
        
        return cv
    
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
    executor_mem = args.executor_mem
    driver_mem = args.driver_mem
    grid_size = args.grid_size
    num_trees = args.num_trees
    max_depth = args.max_depth

    print("\n" + "="*50)
    print("Program Configuration")
    print("="*50)
    print(f"Train files: {train_files}")
    print(f"Valid files: {valid_files}")
    print(f"Test files: {test_files}")
    print(f"Executor memory: {executor_mem}")
    print(f"Driver memory: {driver_mem}")
    print(f"Grid size: {grid_size}m")
    print(f"Random Forest trees: {num_trees}")
    print(f"Max depth: {max_depth}")
    print("="*50 + "\n")

    # Create Spark session
    spark = create_spark_session(
        app_name="Land Cover Classification", 
        executor_mem=executor_mem, 
        driver_mem=driver_mem
    )
        
    # taskmetrics = TaskMetrics(spark)
    
    # Load data
    print("\n" + "="*50)
    print("Loading Data")
    print("="*50)
    
    # taskmetrics.begin()
    df_train = read_parquet_from_s3(spark, args.input)
    
    # Cache after loading for reuse
    df_train = df_train.cache()
    row_count = df_train.count()
    # taskmetrics.end()
    
    print(f"\nLoaded {row_count:,} points")
    # print("\n" + "="*50)
    # print("Read Parquet Statistics")
    # print("="*50)
    # taskmetrics.print_report()
    # print("="*50 + "\n")
    
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
    
    df_train = prepare_features(df_train, grid_size=grid_size)
    
    df_train.printSchema()
    
    print(f"\nTotal features: {len(feature_cols)}")
    print("Features:", ", ".join(feature_cols))
    
    # Build pipeline
    print("\n" + "="*50)
    print("Building ML Pipeline")
    print("="*50)
    
    pipeline = build_ml_pipeline(
        feature_cols, 
        num_trees=num_trees, 
        max_depth=max_depth,
        use_cv=args.use_cv
    )
    
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
    
    predictions = model.transform(df_train)
    predictions = predictions.cache()
    
    # Evaluate model
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
    parser = argparse.ArgumentParser(
        description="Optimized PySpark Land Cover Classification"
    )
    parser.add_argument(
        "--input", 
        required=False, 
        help="Input parquet file(s)",
        default=default_parq_file
    )
    parser.add_argument(
        "--executor-mem",
        required=False, 
        help="Executor memory",
        default=default_executor_mem
    )
    parser.add_argument(
        "--driver-mem",
        required=False, 
        help="Driver memory",
        default=default_driver_mem
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        required=False,
        help="Spatial grid size in meters",
        default=2.0
    )
    parser.add_argument(
        "--num-trees",
        type=int,
        required=False,
        help="Number of Random Forest trees",
        default=100
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        required=False,
        help="Maximum tree depth",
        default=10
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use cross-validation for hyperparameter tuning"
    )
    parser.add_argument(
        "--output-model",
        required=False,
        help="Path to save trained model",
        default=None
    )
    
    args = parser.parse_args()
    main(args)