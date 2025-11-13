import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def setup_logging(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = args.experiment_name or f"fractal-rf-e{args.executor_memory}g-x{args.num_executors}-f{args.sample_fraction}"
    log_file = Path(args.event_log_dir) / f"{log_name}_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

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
    parser = argparse.ArgumentParser()
   