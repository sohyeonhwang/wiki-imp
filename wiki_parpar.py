#!/usr/bin/env python3
"""
Wikipedia Revision Statistics Parser

Parses Wikipedia revision parquet files and computes aggregated statistics
for a given list of article IDs.
"""

import argparse
from datetime import datetime
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import time

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
    columns = ["articleid", "date_time", "revid","editor"]

    combined = pd.read_parquet(
        namespace_dir,
        columns=columns,
        filters=[("articleid", "in", article_ids)],
    )

    # Compute aggregations
    if not combined.empty:
        stats = combined.groupby("articleid").agg(
            earliest_revision=("date_time", "min"),
            revision_count=("revid", "count"),
            editor_nunique=("editor", "nunique"),
        )
    else:
        stats = pd.DataFrame(
            columns=["earliest_rev", "rev_count", "editor_nunique"]
        )
        stats.index.name = "articleid"

    # Determine found vs not found
    found_ids = [aid for aid in article_ids if aid in set(stats.index)]
    not_found_ids = [aid for aid in article_ids if aid not in set(stats.index)]

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
    print(f"> Split {len(article_ids)} article IDs into {len(chunks)} chunks of size ~{chunk_size}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(compute_revision_stats, namespace_dir, chunk)
            for chunk in chunks
        ]
        results = [f.result() for f in futures]  #parallel run gives extra columns: earliest_rev,rev_count

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
        default="/scratch/10114/nathante/global_data/enwiki_2025_wikiq_output.parquet/'namespace=0'"
    )
    parser.add_argument(
        "-id", "--article-ids-file",
        type=Path,
        required=True,
        help="Path to a text file with one articleid per line",
        default="pageids.txt"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/parpar"),
        help="Path to output CSV file (default: parpar.csv and .json)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also output a JSON version of the output."
    )
    parser.add_argument(
        "-t","--test",
        action="store_true",
        help="Test mode: only process the first 50 article IDs"
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (uses ProcessPoolExecutor)"
    )
    args = parser.parse_args()

    print("\n\n\n\n=== Wikipedia Revision Statistics Parser ===\n")

    # Validate inputs
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"> Run: {run_ts}")
    print("> Validating inputs...")
    #if not args.namespace_dir.is_dir():
    #    parser.error(f"Namespace directory does not exist: {args.namespace_dir}")
    if not args.article_ids_file.is_file():
        parser.error(f"! Article IDs file does not exist: {args.article_ids_file}")

    # Load article IDs
    print("> Loading article IDs...")
    article_ids = load_article_ids(args.article_ids_file)
    if not article_ids:
        parser.error("No article IDs found in the input file")
    print(f"> Note: overall {len(article_ids)} article IDs.")

    if args.test:
        article_ids = article_ids[:500]
        print(f"> Test mode: sliced to {len(article_ids)} article IDs")

    print(f"> Loaded {len(article_ids)} article IDs")
    print(f"> Scanning parquet files in {args.namespace_dir}")

    input("> Start processing? (Press Enter to continue)")

    # Compute statistics
    if args.parallel:
        print(f"> Using {args.parallel} parallel workers")
        start_time = time.time()
        stats, found_ids, not_found_ids = compute_revision_stats_parallel(
            args.namespace_dir, article_ids, args.parallel
        )
        end_time = time.time()
        print(f"> Parallel processing took {end_time - start_time:.2f} seconds")
    else:
        print(f"> Processing {len(article_ids)} article IDs sequentially")
        #time how long it takes to process sequentially
        start_time = time.time()
        stats, found_ids, not_found_ids = compute_revision_stats(
            args.namespace_dir, article_ids
        )
        end_time = time.time()
        print(f"> Sequential processing took {end_time - start_time:.2f} seconds")

    # Output formatting
    # timestamp
    csv_path = f"{args.output}_{run_ts}_{'parallel' if args.parallel else 'sequential'}.csv"
    json_path = f"{args.output}_{run_ts}_{'parallel' if args.parallel else 'sequential'}.json"

    # Write CSV output
    stats.to_csv(csv_path)
    print(f"> Wrote parquet-parsed output to {csv_path}")

    # Write JSON output
    if args.json:
        stats.to_json(json_path, orient="records")
    print(f"> Wrote parquet-parsed output to {json_path}")

    # Tracking what was / wasn't found
    track_file = f"output/tracking_{run_ts}_{'parallel' if args.parallel else 'sequential'}.json"
    json_output = {
        "requested": article_ids,
        "found": found_ids,
        "not_found": not_found_ids,
    }
    with open(track_file, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"> Wrote article ID tracking to {track_file}")

    # Summary
    print(f"\n> Summary:")
    print(f"  Requested: {len(article_ids)}")
    print(f"  Found: {len(found_ids)}")
    print(f"  Not found: {len(not_found_ids)}")

    print("Done.\n\n\n")

if __name__ == "__main__":
    main()
