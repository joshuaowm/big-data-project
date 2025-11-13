def create_spark_session(args, max_executors=None):
    builder = SparkSession.builder.appName("fractal-cv-rf")

    if args.master:
        builder = builder.master(args.master)

    max_exec = max_executors if max_executors else args.num_executors

    session = (
        builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.executor.cores", str(args.executor_cores))
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.rpc.message.maxSize", "512")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "1")
        .config("spark.dynamicAllocation.maxExecutors", str(max_exec))
        .config("spark.dynamicAllocation.initialExecutors", str(args.num_executors))
        .config("spark.dynamicAllocation.executorIdleTimeout", "60s")
        .config("spark.dynamicAllocation.shuffleTracking.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "/opt/spark/spark-events")
        .config("spark.executor.heartbeatInterval", "20s")
        .config("spark.network.timeout", "300s")
        .getOrCreate()
    )
    
    return session