import os 
import argparse
import re

MAX_EPOCH = 5000

def count_loops(filename):
  R = []
  with open(filename,'rb') as file:
    B = file.read()[24:]
    
    for i in range(len(B)//64):
      p = B[64*i:64*(i+1)]
      pr = filter(lambda x: x in b"{}[]-+.,<>",p)
      print(p[:2],(p[0]-p[1])%128,"".join([x.to_bytes().decode("utf-8") for x in pr]))
def main():
  parser= argparse.ArgumentParser()
  parser.add_argument("-f","--file",help="file to process")
  args =parser.parse_args()

  count_loops(args.file)

if __name__ == "__main__":
    main()
