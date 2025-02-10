# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import re

MAX_EPOCH = 5000


def count_loops(filename):
    R = []
    with open(filename, "rb") as file:
        B = file.read()[24:]

        for i in range(len(B) // 64):
            p = B[64 * i : 64 * (i + 1)]
            pr = filter(lambda x: x in b"{}[]-+.,<>", p)
            print(
                p[:2],
                (p[0] - p[1]) % 128,
                "".join([x.to_bytes().decode("utf-8") for x in pr]),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="file to process")
    args = parser.parse_args()

    count_loops(args.file)


if __name__ == "__main__":
    main()
