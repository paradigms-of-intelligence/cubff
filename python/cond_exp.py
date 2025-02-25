#!/usr/bin/env python3

"""
from a set of samples estimate E[X - x | X > x] as a function of x
generate an estimate, say, at every transition and ignore horizontal error bars

TODO: actually reasonable meaning of error bars

output is compatible with `plot 'foo.txt' with errorbars`
"""

import sys
import math
import csv


def censored_expectations(samples):
    samples = sorted(samples)
    for i in range(1, len(samples) - 1):
        subset = samples[len(samples) - i :]
        cutoff = samples[len(samples) - i - 1]
        exp = sum(subset) / len(subset)
        # var is variance of error of our estimate of expectation
        var = sum([(x - exp) ** 2 for x in subset]) / len(subset) / len(subset)
        print(cutoff, exp - cutoff, math.sqrt(var))


vals = []

if len(sys.argv) > 1:
    for fname in sys.argv[1:]:
        with open(fname) as f:
            r = csv.DictReader(f)
            last_epoch = None
            for row in r:
                print(row)
                # TODO: use the higher_entropy>3 heuristic instead (or sth totally different)
                if (
                    8.0 * float(row["brotli_size"]) / float(row["soup_size"]) / 64.0
                    < 3.0
                ):
                    last_epoch = float(row["epoch"])
                    break
            assert last_epoch is not None

            vals.append(last_epoch)
else:
    vals = [float(v.strip()) for v in sys.stdin if v.strip() != ""]

censored_expectations(vals)
