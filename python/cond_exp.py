#!/usr/bin/env python3

"""
from a set of samples estimate E[X - x | X > x] as a function of x
generate an estimate, say, at every transition and ignore horizontal error bars

TODO: actually reasonable meaning of error bars

output is compatible with `plot 'foo.txt' with errorbars`
"""

import argparse
import sys
import math
import json


def censored_expectations(samples):
    samples = sorted(samples)
    for i in range(1, len(samples) - 1):
        subset = samples[len(samples) - i :]
        cutoff = samples[len(samples) - i - 1]
        exp = sum(subset) / len(subset)
        # var is variance of error of our estimate of expectation
        var = sum([(x - exp) ** 2 for x in subset]) / len(subset) / len(subset)
        print(cutoff, exp - cutoff, math.sqrt(var))


def cdf(samples):
    samples = sorted(samples)
    for i, x in enumerate(samples):
        print(x, float(i) / len(samples))


vals = []

parser = argparse.ArgumentParser(prog="cond_exp")
parser.add_argument("--raw_samples", action="store_true")
parser.add_argument(
    "--output", choices=["censored_expectation", "cdf"], default="censored_expectation"
)
parser.add_argument("input_files", nargs="*")

args = parser.parse_args(sys.argv[1:])


for fname in args.input_files:
    with open(fname) as f:
        if args.raw_samples:
            vals += [float(v.strip()) for v in f if v.strip() != ""]
        else:
            last_epoch = None
            for row in f:
                row = json.loads(row)
                if row["higher_entropy"] > 3.0:
                    last_epoch = row["epoch"]
                    break
            assert last_epoch is not None

            vals.append(last_epoch)

if args.output == "censored_expectation":
    censored_expectations(vals)
elif args.output == "cdf":
    cdf(vals)
else:
    assert False
