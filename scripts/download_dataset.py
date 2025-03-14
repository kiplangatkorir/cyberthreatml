"""
Download and verify the CICIDS2017 dataset files.
"""
import os
import sys
import requests
import hashlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "CICIDS2017"
REQUIRED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv"
]

def main():
    logger.info("CyberThreat-ML Dataset Downloader")
    logger.info("=" * 80)
    
    # Create dataset directory if it doesn't exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Dataset will be downloaded to: {DATASET_DIR}")
    logger.info("\nPlease follow these steps:")
    logger.info("1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html")
    logger.info("2. Fill out the registration form to get download access")
    logger.info("3. Download the following files and place them in the datasets/CICIDS2017 directory:")
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
