# ingest_batch.py (Final Working Version)

import os
import argparse
import logging
import uuid
import sys
from pathlib import Path 
import time
from typing import Optional 

try:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
         sys.path.insert(0, str(project_root))
    print(f"DEBUG: Project root added to sys.path: {project_root}")
except Exception as e:
    print(f"Warning: Could not reliably determine project root: {e}")
# -----------------------------------------------------------

# --- Import application modules ---
try:
    from sqlalchemy.orm import Session
    from main import SessionLocal, User 
    from db_manager import DBManager 
    from dotenv import load_dotenv
    load_dotenv() # Load .env before initializing DBManager potentially
except ImportError as e:
    print(f"FATAL ERROR: Error importing modules. Make sure this script is in the project root.")
    print(f"Ensure main.py defines SessionLocal/User and db_manager.py exists.")
    print(f"Did you run 'pip install -r requirements.txt' in the venv?")
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during module imports: {e}")
    sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] 
)
logger = logging.getLogger("ingest_batch")

# --- Allowed File Extensions ---
ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md", ".json"}

# --- Main Processing Function ---
def process_directory(db_manager_instance: DBManager, db: Session, directory_path: str, admin_user_id: uuid.UUID, move_processed: bool):
    """Scans a directory and ingests allowed files."""
    source_dir = Path(directory_path)
    processed_dir = source_dir / "processed"
    failed_dir = source_dir / "failed"

    if move_processed:
        try:
            processed_dir.mkdir(exist_ok=True)
            failed_dir.mkdir(exist_ok=True)
            logger.info(f"Processed files will be moved to: {processed_dir}")
            logger.info(f"Failed files will be moved to: {failed_dir}")
        except OSError as e:
            logger.error(f"Could not create processed/failed directories: {e}. Files will not be moved.")
            move_processed = False # Disable moving if directories can't be made

    total_files_found = 0
    processed_count = 0
    failed_count = 0

    logger.info(f"Scanning directory: {directory_path}")
    for file_path_obj in source_dir.rglob('*'):
        if not file_path_obj.is_file():
            continue
        if move_processed and (file_path_obj.parent == processed_dir or file_path_obj.parent == failed_dir):
            continue

        file_path = str(file_path_obj)
        filename = file_path_obj.name
        file_ext = file_path_obj.suffix.lower()

        if file_ext in ALLOWED_EXTENSIONS:
            total_files_found += 1
            logger.info(f"--- [{total_files_found}] Processing file: {file_path} ---")
            start_time = time.time()
            ingested_ids = None 
            status = "FAILED" 

            try:
                ingested_ids = db_manager_instance.add_data_from_file(file_path)

                if ingested_ids:
                    logger.info(f"Successfully processed and added {len(ingested_ids)} chunks.")
                    db_manager_instance.add_uploaded_document_record(
                        filename=filename, 
                        user_id=admin_user_id,
                        db=db
                    )
                    db.commit() 
                    processed_count += 1
                    status = "PROCESSED"
                    logger.info(f"Committed tracking record for {filename}.")
                else: 
                    logger.warning(f"File processed but no ingestible data found or added: {filename}")
                    db.rollback() 
                    failed_count += 1
                    status = "NO_DATA"

            except Exception as e:
                db.rollback() # Rollback on any processing error for this file
                failed_count += 1
                status = "ERROR"
                logger.error(f"Failed to process file {filename}: {e}", exc_info=False) # Set exc_info=True for full traceback
                # Optionally log the full traceback to a separate file if needed

            finally:
                # Move file based on status if requested
                if move_processed:
                    destination_dir = None
                    prefix = ""
                    if status == "PROCESSED":
                        destination_dir = processed_dir
                    elif status == "NO_DATA":
                        destination_dir = failed_dir
                        prefix = "nodata_"
                    elif status == "ERROR":
                        destination_dir = failed_dir
                        prefix = "error_"

                    if destination_dir:
                        try:
                            # Ensure unique name in destination
                            target_path = destination_dir / f"{prefix}{filename}"
                            counter = 0
                            while target_path.exists():
                                counter += 1
                                target_path = destination_dir / f"{prefix}{file_path_obj.stem}_{counter}{file_path_obj.suffix}"

                            file_path_obj.rename(target_path)
                            logger.info(f"Moved {status.lower()} file to: {target_path}")
                        except OSError as move_err:
                            logger.error(f"Error moving {status.lower()} file {filename}: {move_err}")
                        except Exception as generic_move_err:
                             logger.error(f"Unexpected error moving {status.lower()} file {filename}: {generic_move_err}")


            end_time = time.time()
            logger.info(f"Finished processing {filename} in {end_time - start_time:.2f} seconds. Status: {status}")
            logger.info("-" * (len(file_path) + 20)) # Separator line


    logger.info(f"--- Scan Complete ---")
    logger.info(f"Found {total_files_found} supported files.")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed/No Data: {failed_count}")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest documents into the RAG knowledge base.")
    parser.add_argument("-d", "--directory", required=True, help="Path to the directory containing files to ingest.")
    parser.add_argument("-u", "--admin-email", required=True, help="Email address of an existing admin user for tracking.")
    parser.add_argument("--move", action="store_true", help="Move processed/failed files to subdirectories ('processed', 'failed').")
    parser.add_argument("--debug-db", action="store_true", help="Enable verbose SQLAlchemy logging.")


    args = parser.parse_args()

    # Enable SQLAlchemy logging if requested
    if args.debug_db:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
        logging.getLogger('sqlalchemy.pool').setLevel(logging.DEBUG)


    # Use Path object for robust path handling
    target_dir_path = Path(args.directory)
    if not target_dir_path.is_dir():
        logger.error(f"Error: Provided path is not a valid directory: {args.directory}")
        sys.exit(1)

    # --- Initialize DBManager (Requires DB access) ---
    db_manager_instance: Optional[DBManager] = None # Type hint
    try:
        logger.info("Initializing DBManager...")
        db_manager_instance = DBManager()
        logger.info("DBManager initialized successfully.")
    except Exception as e:
         logger.error(f"Fatal: Failed to initialize DBManager: {e}", exc_info=True)
         sys.exit(1)

    # --- Get DB Session and Admin User ---
    db: Optional[Session] = None # Type hint
    admin_user = None
    try:
        logger.info("Creating database session...")
        db = SessionLocal()
        logger.info(f"Looking up admin user: {args.admin_email}")
        admin_user = db.query(User).filter(User.email == args.admin_email, User.is_admin == True).first()
        if not admin_user:
            logger.error(f"Error: Admin user with email '{args.admin_email}' not found or is not an admin.")
            db.close()
            sys.exit(1)
        logger.info(f"Found admin user ID: {admin_user.id}")

        # --- Start Processing ---
        logger.info("Starting directory processing...")
        process_directory(db_manager_instance, db, args.directory, admin_user.id, args.move)
        logger.info("Directory processing finished.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the batch process: {e}", exc_info=True)
        if db:
            try:
                db.rollback() # Attempt rollback on major error
                logger.info("Database transaction rolled back due to error.")
            except Exception as rb_err:
                 logger.error(f"Exception during rollback attempt: {rb_err}")

    finally:
        if db:
            logger.info("Closing database session.")
            try:
                db.close()
            except Exception as close_err:
                 logger.error(f"Exception during DB session close: {close_err}")

    logger.info("Batch ingestion script finished.")