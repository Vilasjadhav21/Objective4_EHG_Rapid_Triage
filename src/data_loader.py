"""
Data Loader module for PhysioNet Term-Preterm Electrohysterogram Database (TPEHGDB).

Provides utility functions to read RECORDS list and load records using WFDB.
"""

from pathlib import Path
from typing import List, Union
import wfdb


class TPEHGDataLoader:
    """Loader for PhysioNet TPEHGDB records using the WFDB library."""

    def __init__(self, data_dir: Union[str, Path] = "data/tpehgdb"):
        self.data_dir = Path(data_dir)

    def read_records_list(self, records_filename: str = "RECORDS") -> List[str]:
        """
        Read list of record names from the RECORDS file in data_dir.

        Returns:
            List of record identifiers.
        """
        records_path = self.data_dir / records_filename
        if not records_path.exists():
            # Fallback check for root data/tpehgdb/RECORDS
            records_path = Path("data/tpehgdb/RECORDS")

        if not records_path.exists():
            raise FileNotFoundError(f"RECORDS file not found at {records_path}")

        with open(records_path, "r", encoding="utf-8") as f:
            records = [line.strip() for line in f if line.strip()]

        return records

    def get_first_record_name(self) -> str:
        """Select and return the FIRST record listed in the RECORDS file."""
        records = self.read_records_list()
        if not records:
            raise ValueError("RECORDS file is empty.")
        return records[0]

    def load_record(self, record_name: str) -> wfdb.Record:
        """
        Load a WFDB record by record_name using wfdb.rdrecord().

        Args:
            record_name: Record identifier string (e.g., 'tpehg1007' or 'tpehgdb/tpehg1007').

        Returns:
            wfdb.Record object containing header metadata and signal array.
        """
        clean_name = record_name.strip()

        # 1. Check direct path under data_dir
        local_path = self.data_dir / clean_name
        if local_path.with_suffix('.hea').exists():
            return wfdb.rdrecord(str(local_path))

        # 2. Check stem name under data_dir
        stem_name = Path(clean_name).name
        local_stem_path = self.data_dir / stem_name
        if local_stem_path.with_suffix('.hea').exists():
            return wfdb.rdrecord(str(local_stem_path))

        # 3. Check recursively within data_dir
        hea_matches = list(self.data_dir.glob(f"**/{stem_name}.hea"))
        if hea_matches:
            record_base = str(hea_matches[0].with_suffix(''))
            return wfdb.rdrecord(record_base)

        # 4. If not found in local data directory, raise FileNotFoundError immediately
        raise FileNotFoundError(f"Record '{record_name}' (.hea header file) not found in local directory {self.data_dir}")


def load_record(record_name: str, data_dir: Union[str, Path] = "data/tpehgdb") -> wfdb.Record:
    """Convenience functional wrapper to load a WFDB record."""
    loader = TPEHGDataLoader(data_dir=data_dir)
    return loader.load_record(record_name)
