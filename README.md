# Big Data Project

This repository is for Big Data course final project by Joshua Mangotang from Indonesia and Elouann Nhan Lucas from Vietnam.
The goal of this project is to build a scalable machine learning pipeline using Apache Spark to classify land cover types from large-scale geospatial data.

## Getting started

To set up the environment, follow these steps:

1. **Install dependencies**: Run `pip install -r requirements.txt` to install the required Python packages.
2. **Configure Spark**: Ensure that your Spark environment is set up correctly with the necessary packages for AWS S3 access and Spark Measure for performance metrics.
3. **Submit training**: Use the provided scripts to run your Spark jobs.
OR here is an example of running a Spark job with specific configurations:

   ```bash
   spark-submit --deploy-mode client --master yarn --packages org.apache.hadoop:hadoop-aws:3.3.1,ch.cern.sparkmeasure:spark-measure_2.12:0.27 train.py --num-executors 8 --executor-cores 2 --executor-memory 16 --driver-memory 4 --data "s3a://ubs-datasets/FRACTAL/data" --fraction 0.1
   ```
4. **Scaling evaluation**: 
After the application is completed, you can find the application's performance metrics reported in a json file located the `results` directory.
File name guide : `metrics_f0.01_x16_e2.json` : 1% sample fraction, 16GB executor memory, 2 executor cores.
To evaluate the scaling of the Spark job, you can run multiple experiments with varying configurations. For example:
   - Vary the sample fraction: 0.01, 0.05, 0.1
   - Vary the executor memory: 8GB, 16GB, 32GB
   - Vary the number of executor cores: 2, 4

5. **Analyze results**:
Use any server to open `charts.html` located in the `results` directory to visualize the scaling results with fancy charts.