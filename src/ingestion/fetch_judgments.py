import asyncio
from pathlib import Path
import logging

try:
    from bharat_courts import Judgments
except ImportError:
    print("Warning: The 'bharat_courts' library is required but not found. Please install it.")
    raise

# --- DOCKER PATH FIX ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# UPDATE PATH
OUTPUT_DIR = PROJECT_ROOT / "data/raw/judgments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging for better operational feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def fetch_by_year(year: str, limit: int = 50):
    """
    Fetch and save judgments from Chhattisgarh HC for a given year.
    Includes logging and structure for robust fetching.
    """
    j = Judgments()
    logging.info(f"--- Starting fetch process for year {year} (Limit: {limit}) ---")

    try:
        # Attempt to find results
        results = await j.find(
            court="chhattisgarh",
            year=year,
            source="archive",
            limit=limit
        )
        logging.info(f"✅ Found {len(results)} judgments records for year {year}.")

        for idx, result in enumerate(results):
            try:
                # Fetch the PDF bytes
                pdf_bytes = await j.fetch_pdf(result)

                # Use CNR (unique case number) as the primary identifier
                case_id = result.cnr or result.case_id or f"unknown_{year}_{idx}"
                # Sanitize: Ensure filename compatibility
                case_id_safe = str(case_id).replace("/", "_").replace("\\", "_")

                filename = f"{case_id_safe}_{year}.pdf"
                filepath = OUTPUT_DIR / filename
                
                # Save the file bytes
                filepath.write_bytes(pdf_bytes)
                logging.info(f"  -> Successfully saved: {filename}")

            except Exception as e:
                # Catch exceptions specific to a single result fetch/save
                logging.error(f"Failed to process record #{idx} for year {year}: {e}")
        
    except Exception as e:
        # Catch broader failures (e.g., API connectivity issues)
        logging.critical(f"Fatal error fetching data for year {year}. Check connectivity or source limits: {e}")

async def main():
    """Main asynchronous loop to orchestrate the fetching of multiple years."""
    # Fetching recent years (2021-2026)
    years_to_fetch = [str(y) for y in range(2021, 2027)] 
    tasks = []
    for year in years_to_fetch:
        # 250 per year * 6 years = 1500 new judgments
        tasks.append(fetch_by_year(year, limit=250))

    # Running all tasks concurrently
    await asyncio.gather(*tasks)
    logging.info("\n✨ All fetching tasks have been attempted.")

if __name__ == "__main__":
    asyncio.run(main())