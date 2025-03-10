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


"""
Tool to analyse selfreplicators in bff soups including head position
"""

import os
import argparse
import re

MAX_EPOCH = 5000
MAX_INSTR = 8192

CEND = "\33[0m"
CSTART = "\33[31m"


def print_forth(p):
    for i, b in p:
        match b.to_bytes():
            case b"\x00":
                print(f"{CSTART+str(i)+CEND}READ0", end=" ")
            case b"\x01":
                print(f"{CSTART+str(i)+CEND}READ1", end=" ")
            case b"\x02":
                print(f"{CSTART+str(i)+CEND}WRITE0", end=" ")
            case b"\x03":
                print(f"{CSTART+str(i)+CEND}WRITE1", end=" ")
            case b"\x04":
                print(f"{CSTART+str(i)+CEND}DUP", end=" ")
            case b"\x05":
                print(f"{CSTART+str(i)+CEND}DROP", end=" ")
            case b"\x06":
                print(f"{CSTART+str(i)+CEND}SWAP", end=" ")
            case b"\x07":
                print(f"{CSTART+str(i)+CEND}IF0", end=" ")
            case b"\x08":
                print(f"{CSTART+str(i)+CEND}INC", end=" ")
            case b"\x09":
                print(f"{CSTART+str(i)+CEND}DEC", end=" ")
            case b"\x0A":
                print(f"{CSTART+str(i)+CEND}ADD", end=" ")
            case b"\x0B":
                print(f"{CSTART+str(i)+CEND}SUB", end=" ")
            case b"\x0C":
                print(f"{CSTART+str(i)+CEND}COPY0", end=" ")
            case b"\x0D":
                print(f"{CSTART+str(i)+CEND}COPY1", end=" ")
            case _:
                if b >> 7 & 1:
                    print(
                        f"{CSTART+str(i)+CEND}JUMP{'-'if b>>6&1 else '+'}{(b&63)+1}",
                        end=" ",
                    )
                else:
                    print(f"{CSTART+str(i)+CEND}PUSH{b&63}", end=" ")
    print("\n")


def print_forthcopy(p):
    for i, b in p:
        match b.to_bytes():
            case b"\x00":
                print(f"{CSTART+str(i)+CEND}READ", end=" ")
            case b"\x01":
                print(f"{CSTART+str(i)+CEND}WRITE", end=" ")
            case b"\x02":
                print(f"{CSTART+str(i)+CEND}COPY", end=" ")
            case b"\x03":
                print(f"{CSTART+str(i)+CEND}XOR", end=" ")
            case b"\x04":
                print(f"{CSTART+str(i)+CEND}DUP", end=" ")
            case b"\x05":
                print(f"{CSTART+str(i)+CEND}DROP", end=" ")
            case b"\x06":
                print(f"{CSTART+str(i)+CEND}SWAP", end=" ")
            case b"\x07":
                print(f"{CSTART+str(i)+CEND}IF0", end=" ")
            case b"\x08":
                print(f"{CSTART+str(i)+CEND}INC", end=" ")
            case b"\x09":
                print(f"{CSTART+str(i)+CEND}DEC", end=" ")
            case b"\x0A":
                print(f"{CSTART+str(i)+CEND}ADD", end=" ")
            case b"\x0B":
                print(f"{CSTART+str(i)+CEND}SUB", end=" ")
            case b"\x0C":
                print(f"{CSTART+str(i)+CEND}COPY0", end=" ")
            case b"\x0D":
                print(f"{CSTART+str(i)+CEND}COPY1", end=" ")
            case _:
                if b >> 7 & 1:
                    print(
                        f"{CSTART+str(i)+CEND}JUMP{'-'if b>>6&1 else '+'}{(b&63)+1}",
                        end=" ",
                    )
                else:
                    print(f"{CSTART+str(i)+CEND}PUSH{b&63}", end=" ")
    print("\n")


def forth_loop(p):
    i = 0
    c = 0
    vis = [0 for _ in p]

    while i < MAX_INSTR:
        if vis[c]:
            return True
        vis[c] = 1
        if p[c] == 7:
            c += 2
            i += 1
        elif p[c] < 64:
            c += 1
            i += 1
        else:
            i += 1
            c = c - (p[c] & 63) + 1 if p[c] >> 6 & 1 else c + (p[c] & 63) + 1
        if c < 0 or c > 63:
            return False
    return True


def count_forth_loops(filename):
    res = 0
    with open(filename, "rb") as file:
        B = file.read()[24:]

        for i in range(len(B) // 64):
            p = B[64 * i : 64 * (i + 1)]
            if forth_loop(p):
                res += 1
    print(res)


def analyse(filename, lang):
    R = []
    with open(filename, "rb") as file:
        B = file.read()[24:]

        for i in range(len(B) // 64):
            p = B[64 * i : 64 * (i + 1)]
            if lang == "bff":
                pr = filter(lambda x: x in b"{}[]-+.,<>", p)
                print(
                    p[:2],
                    (p[0] - p[1]) % 128,
                    "".join([x.to_bytes().decode("utf-8") for x in pr]),
                )
            if lang == "bff_noheads":
                pr = filter(lambda x: x in b"{}[]-+.,<>", p)
                print("".join([x.to_bytes().decode("utf-8") for x in pr]))
            if lang == "forth":
                pr = filter(
                    lambda x: x[1]
                    in b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B"
                    or x[1].to_bytes() >= b"\x40",
                    enumerate(p),
                )
                print_forth(pr)
            if lang == "forthcopy":
                pr = filter(
                    lambda x: x[1]
                    in b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B"
                    or x[1].to_bytes() >= b"\x40",
                    enumerate(p),
                )
                print_forthcopy(pr)
            if lang == "forthtrivial":
                pr = filter(
                    lambda x: x[1]
                    in b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D"
                    or x[1].to_bytes() >= b"\x40",  # 64
                    enumerate(p),
                )
                print_forth(pr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="file to process")
    parser.add_argument("-l", "--lang", help="language to process")
    parser.add_argument(
        "-cl",
        "--countloopforth",
        action="store_true",
        help="count the number of forth programs that loop forever",
    )
    args = parser.parse_args()

    if args.countloopforth:
        count_forth_loops(args.file)
    else:
        analyse(args.file, args.lang)


if __name__ == "__main__":
    main()
