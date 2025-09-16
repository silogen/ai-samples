### Copyright 2025 Advanced Micro Devices, Inc.  All rights reserved.
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###      http://www.apache.org/licenses/LICENSE-2.0
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.

"""
Trace analysis for the MONAI/AZ-SwinUNETR++ example.

The script prints out a summary of metrics of interest, such as, time spent in
the different "sections" (user annotations with `__sec:` prefix).

Note that this is **not** a generic script to analyse profile traces.
Pleaes adapt it to your needs if using in a different context.
"""

import argparse
import gzip
import json
import zipfile

import pandas as pd
import polars as pl


def load(path: str):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zip:
            with zip.open("trace.json") as fo:
                data = json.load(fo)
    elif path.endswith(".gz"):
        with gzip.open(path) as fo:
            data = json.load(fo)
    elif path.endswith(".json"):
        with open(path, "wt") as fo:
            data = json.load(fo)
    else:
        raise ValueError("File must be a '.json', '.zip' or '.gz'.")

    # polars struggles to figure out the column types directly, so we use
    # here pandas as an intermediary
    return pl.DataFrame(
        pd.json_normalize(data["traceEvents"])[
            ["name", "ts", "dur", "cat", "args.correlation"]
        ].reset_index()
    )


def analyse_trace(path: str) -> tuple[pl.DataFrame, ...]:
    df = load(path).select(
        "index",
        "name",
        pl.col("ts").alias("start"),
        (pl.col("ts") + pl.col("dur")).alias("end"),
        "dur",
        "cat",
        "args.correlation",
    )

    # filter for the annotated sections
    sections = (
        df.filter(pl.col("name").str.starts_with("__sec:"))
        .select(pl.exclude("cat", "args.correlation"))
        .rename({"name": "section_name"})
    )

    # get all kernels launched within the sections
    # this is done by
    # 1. getting the kernel launchers in each section that live on the CPU side
    #    (note that some sections may not have any)
    # 2. match the lauchers with the kernels based on the correlation id (args.correlation_id)
    kl = df.filter(
        pl.col("name").is_in(["hipLaunchKernel", "hipExtModuleLaunchKernel"])
    ).select(pl.col("start").alias("kl_start"), "args.correlation")
    section_kl = sections.join_where(
        kl,
        [pl.col("start") < pl.col("kl_start"), pl.col("kl_start") < pl.col("end")],
    )
    kl_kernels = section_kl.join(
        df.filter(pl.col("cat").eq("kernel")).select(
            pl.col("name").alias("kernel_name"),
            pl.col("start").alias("kernel_start"),
            pl.col("dur").alias("kernel_dur"),
            pl.col("end").alias("kernel_end"),
            "args.correlation",
        ),
        on="args.correlation",
        how="left",
    )

    # sanity-check
    assert not kl_kernels["kernel_name"].is_null().any()
    assert kl_kernels.height == section_kl.height

    # merge the kernels' information back to the complete list of sections
    # this dataframe contains 1 row per kernel, with the corresponding section information
    # sections that do not launch kernels appear as a single row with missing kernel information
    sections_and_kernels = sections.join(
        kl_kernels, on=["index", "section_name"], how="left"
    )

    # calculate metrics per section instance
    sections_timings = sections_and_kernels.group_by(pl.col("index")).agg(
        pl.col("section_name").first(),
        (pl.col("kernel_dur").sum() / 1e6).alias("kernel_total_time [s]"),
        ((pl.col("end") - pl.col("start")) / 1e6).first().alias("cpu_total_time [s]"),
        ((pl.col("kernel_end").max() - pl.col("kernel_start").min()) / 1e6).alias(
            "kernel first-to-last [s]"
        ),
    )

    # timings per section
    section_metric = {
        "__sec:step": "cpu_total_time [s]",
        "__sec:forward": "kernel first-to-last [s]",
        "__sec:backward": "kernel first-to-last [s]",
        "__sec:loader_next": "cpu_total_time [s]",
        "__sec:loader_first": "cpu_total_time [s]",
    }
    stats = []
    for sec, col in section_metric.items():
        s = sections_timings.filter(pl.col("section_name").eq(sec))[col]
        stats.append((sec.split(":")[-1], s.min(), s.mean(), s.median(), s.max()))

    section_timings = pl.from_records(
        stats, schema=["section", "min", "mean", "median", "max"], orient="row"
    )

    return (section_timings,)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the trace file.")
    parser.add_argument(
        "trace_file", type=str, help="Path to the trace file (.json, .zip, or .gz)"
    )
    args = parser.parse_args()
    return args


def print_dfs(*dfs: pl.DataFrame, headers: list[str] = None):
    headers = headers or [None] * len(dfs)
    assert len(headers) == len(dfs), (
        f"Number of headers ({len(headers)}) differs from number of dataframes ({len(dfs)})."
    )

    for df, header in zip(dfs, headers):
        print()
        print(
            "\n"
            + "=" * 88
            + (f"\n{header.center(88)}\n" if header else "\n")
            + "=" * 88
            + "\n"
        )
        print(str(df).split("\n", 1)[1])


def cli():
    args = parse_args()

    print_dfs(
        *analyse_trace(args.trace_file),
        headers=["Section Timings"],
    )


if __name__ == "__main__":
    cli()
