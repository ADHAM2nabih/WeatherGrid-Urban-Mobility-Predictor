import logging
import traceback

from generate_datasets import main as generate_main
from etl_pipeline import run_etl
from monte_carlo import run_monte_carlo
from factor_analysis import run_factor_analysis


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("===============================================")
    logger.info("   Big Data Final Project - Full Pipeline Run  ")
    logger.info("===============================================")

    # 1) Generate synthetic datasets
    try:
        logger.info("\n[1/4] Generating synthetic datasets (weather_raw.csv, traffic_raw.csv)...")
        generate_main()
        logger.info("[1/4] Synthetic dataset generation finished.")
    except Exception as e:
        logger.error("[1/4] Failed during dataset generation.")
        logger.error(e)
        traceback.print_exc()
        return

    # 2) Run ETL
    try:
        logger.info("\n[2/4] Running ETL pipeline (Bronze -> Silver -> HDFS -> Gold + merged)...")
        run_etl()
        logger.info("[2/4] ETL pipeline finished.")
    except Exception as e:
        logger.error("[2/4] Failed during ETL pipeline.")
        logger.error(e)
        traceback.print_exc()
        return

    # 3) Monte Carlo simulation
    try:
        logger.info("\n[3/4] Running Monte Carlo simulation...")
        run_monte_carlo()
        logger.info("[3/4] Monte Carlo simulation finished.")
    except Exception as e:
        logger.error("[3/4] Failed during Monte Carlo simulation.")
        logger.error(e)
        traceback.print_exc()
        return

    # 4) Factor Analysis
    try:
        logger.info("\n[4/4] Running Factor Analysis...")
        run_factor_analysis()
        logger.info("[4/4] Factor Analysis finished.")
    except Exception as e:
        logger.error("[4/4] Failed during Factor Analysis.")
        logger.error(e)
        traceback.print_exc()
        return

    logger.info("\n✅ All steps completed successfully.")
    logger.info("   - Raw data generated")
    logger.info("   - ETL done (Bronze/Silver/Gold + HDFS + merged_analytics)")
    logger.info("   - Monte Carlo results saved & uploaded to MinIO gold")
    logger.info("   - Factor Analysis outputs saved & uploaded to MinIO gold")
    logger.info("===============================================")


if __name__ == "__main__":
    main()
