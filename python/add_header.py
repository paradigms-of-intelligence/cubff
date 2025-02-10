"""TODO: cknierim - DO NOT SUBMIT without one-line documentation for add_header.

TODO: cknierim - DO NOT SUBMIT without a detailed description of add_header.
"""
import struct
import sys


def main() -> None:
  f = sys.argv[1]

  with open(f,"rb") as bf:
    b = bf.read()
    # header = reset index, num_programs, epoch
    header = struct.pack('=Q',0) + struct.pack('=Q',131072)+struct.pack('=Q',0)
    new = header + b

  with open(f,'wb') as wf:
    wf.write(new)

if __name__ == "__main__":
  main()
