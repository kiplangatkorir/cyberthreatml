"""
Download instructions for the CIC-IDS2023 dataset.
"""
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "CICIDS2023"
DATASET_URL = "https://www.unb.ca/cic/datasets/ids-2023.html"

REQUIRED_FILES = [
    "DrDoS_MSSQL.csv",
    "DrDoS_NetBIOS.csv",
    "DrDoS_SNMP.csv",
    "Infiltration.csv",
    "BotAttack.csv"
]

def main():
    logger.info("CyberThreat-ML Dataset Downloader - CIC-IDS2023")
    logger.info("=" * 80)
    
    # Create dataset directory if it doesn't exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nTo download the CIC-IDS2023 dataset, please follow these steps:")
    logger.info(f"1. Visit: {DATASET_URL}")
    logger.info("2. Fill out the registration form to get download access")
    logger.info("3. Download the following files and place them in the datasets/CICIDS2023 directory:")
    for file in REQUIRED_FILES:
        logger.info(f"   - {file}")
    
    # Check if files exist
    missing_files = []
    for file in REQUIRED_FILES:
        file_path = DATASET_DIR / file
        if not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning("\nMissing files:")
        for file in missing_files:
            logger.warning(f" - {file}")
        logger.info("\nPlease download the missing files and try again.")
    else:
        logger.info("\nAll required files are present!")
        logger.info("You can now run real_world_testing.py")

if __name__ == "__main__":
    main()
