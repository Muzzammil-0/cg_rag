import asyncio
from pathlib import Path
from bharat_courts import Judgments

OUTPUT_DIR = Path("data/raw/judgments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def fetch_by_year(year: str, limit: int = 50):
    """Fetch 50 judgments from Chhattisgarh HC for a given year."""
    j = Judgments()
    try:
        results = await j.find(
            court="chhattisgarh",
            year=year,
            source="archive",
            limit=limit
        )
        print(f"Found {len(results)} judgments for year {year}")

        for idx, result in enumerate(results):
            pdf_bytes = await j.fetch_pdf(result)

            # Use CNR (unique case number) as the primary identifier
            case_id = result.cnr or result.case_id or f"unknown_{year}_{idx}"
            # Sanitize: CNR can contain slashes
            case_id = str(case_id).replace("/", "_").replace("\\", "_")

            filename = f"{case_id}_{year}.pdf"
            filepath = OUTPUT_DIR / filename
            filepath.write_bytes(pdf_bytes)
            print(f"  Saved: {filename}")
    except Exception as e:
        print(f"Error fetching year {year}: {e}")


async def main():
    # Recent years only — keeps test set manageable
    for year in range(2010, 2015):
        await fetch_by_year(str(year), limit=100)


if __name__ == "__main__":
    asyncio.run(main())