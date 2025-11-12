# Big Data Project

This repository is for Big Data course final project by Joshua Mangotang from Indonesia and Elouann Nhan Lucas from Vietnam.

## Getting started

To set up the environment, follow these steps:

1. **Install dependencies**: Run `pip install -r requirements.txt` to install the required Python packages.
2. **Set up environment variables**: Create a `.env` file in the root directory and add your AWS credentials:

   ```bash
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

3. **Run the application**: Use the provided scripts to run your Spark jobs.
OR here is an example of running a Spark job:

   ```bash
   spark-submit --master yarn --deploy-mode cluster --executor-cores=7 --num-executors=55 --executor-memory=18GB optimized.py 
   ```

## Project Structure

- `spark_utils.py`: Contains utility functions to create and configure Spark sessions.
- `data_processing.py`: Scripts for processing data using Spark.