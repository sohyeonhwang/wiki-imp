#!/usr/bin/env python3
"""
Wikipedia Revision Statistics Parser

Parses Wikipedia revision parquet files and computes aggregated statistics
for a given list of article IDs.
"""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

def load_article_ids(filepath: Path) -> list[int]:
    """Load article IDs from a text file (one ID per line)."""
    article_ids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                article_ids.append(int(line))
    return article_ids


def compute_revision_stats(
    namespace_dir: Path, article_ids: list[int]
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """
    Compute revision statistics for the given article IDs.

    Returns:
        - DataFrame with statistics (earliest_revision, revision_count, editorid_count)
        - List of found article IDs
        - List of not found article IDs
    """
    columns = ["articleid", "date_time", "editorid"]

    combined = pd.read_parquet(
        namespace_dir,
        columns=columns,
        filters=[("articleid", "in", article_ids)],
    )

    # Compute aggregations
    if not combined.empty:
        stats = combined.groupby("articleid").agg(
            earliest_revision=("date_time", "min"),
            revision_count=("date_time", "count"),
            editorid_count=("editorid", "nunique"),
        )
    else:
        stats = pd.DataFrame(
            columns=["earliest_revision", "revision_count", "editorid_count"]
        )
        stats.index.name = "articleid"

    # Determine found vs not found
    found_ids = [aid for aid in article_ids if aid in stats.index]
    not_found_ids = [aid for aid in article_ids if aid not in stats.index]

    # Reindex to include all requested article IDs (missing ones get NaN)
    stats = stats.reindex(article_ids)

    return stats, found_ids, not_found_ids


def compute_revision_stats_parallel(
    namespace_dir: Path, article_ids: list[int], num_workers: int
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """
    Parallel wrapper around compute_revision_stats.

    Splits article_ids into num_workers chunks and processes them in parallel
    using a ProcessPoolExecutor.
    """
    chunk_size = math.ceil(len(article_ids) / num_workers)
    chunks = [
        article_ids[i : i + chunk_size]
        for i in range(0, len(article_ids), chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(compute_revision_stats, namespace_dir, chunk)
            for chunk in chunks
        ]
        results = [f.result() for f in futures]

    all_stats = pd.concat([r[0] for r in results])
    all_found = []
    all_not_found = []
    for _, found, not_found in results:
        all_found.extend(found)
        all_not_found.extend(not_found)

    # Reindex to original article_ids order
    all_stats = all_stats.reindex(article_ids)

    return all_stats, all_found, all_not_found


def main():
    parser = argparse.ArgumentParser(
        description="Compute revision statistics from Wikipedia parquet files"
    )
    parser.add_argument(
        "-n", "--namespace-dir",
        type=Path,
        required=True,
        help="Path to the namespace directory (e.g., .../namespace=0)",
        default="/corral-tacc/utexas/DBS25003/optimized_enwiki_2025_wikiq_output.parquet/'namespace=0'"
    )
    parser.add_argument(
        "-id", "--article-ids-file",
        type=Path,
        required=True,
        help="Path to a text file with one articleid per line",
        default="pageids.txt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parpar.csv"),
        help="Path to output CSV file (default: parpar.csv)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("parpar.json"),
        help="Path to output JSON file tracking articleids (default: parpar.json)"
    )
    parser.add_argument(
        "-t","--test",
        action="store_true",
        help="Test mode: only process the first 5 article IDs"
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (uses ProcessPoolExecutor)"
    )
    args = parser.parse_args()

    # Validate inputs
    print("> Validating inputs...")
    if not args.namespace_dir.is_dir():
        parser.error(f"Namespace directory does not exist: {args.namespace_dir}")
    if not args.article_ids_file.is_file():
        parser.error(f"Article IDs file does not exist: {args.article_ids_file}")

    # Load article IDs
    print("> Loading article IDs...")
    article_ids = load_article_ids(args.article_ids_file)
    if not article_ids:
        parser.error("No article IDs found in the input file")

    if args.test:
        article_ids = article_ids[:50]
        print(f"Test mode: sliced to {len(article_ids)} article IDs")

    print(f"Loaded {len(article_ids)} article IDs")
    print(f"Scanning parquet files in {args.namespace_dir}")

    input("Start processing? (Press Enter to continue)")

    # Compute statistics
    if args.parallel:
        print(f"Using {args.parallel} parallel workers")
        stats, found_ids, not_found_ids = compute_revision_stats_parallel(
            args.namespace_dir, article_ids, args.parallel
        )
    else:
        stats, found_ids, not_found_ids = compute_revision_stats(
            args.namespace_dir, article_ids
        )

    # Write CSV output
    stats.to_csv(args.output)
    print(f"Wrote parquet-parsed statistics to {args.output}")

    # Write JSON output
    json_output = {
        "requested": article_ids,
        "found": found_ids,
        "not_found": not_found_ids,
    }
    with open(args.output_json, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Wrote article ID tracking to {args.output_json}")

    # Summary
    print(f"\nSummary:")
    print(f"  Requested: {len(article_ids)}")
    print(f"  Found: {len(found_ids)}")
    print(f"  Not found: {len(not_found_ids)}")


if __name__ == "__main__":
    main()
