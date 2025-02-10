#!/bin/bash

LANG=$1

if [ -z "$LANG" ]
then
  echo "Usage: $0 <language>"
  exit 1
fi

TESTDATA="testdata/$LANG.txt"
NUM_EPOCHS=256
SEED=10248

if ! [ -f "$TESTDATA" ]
then
  echo "$TESTDATA does not exist! Generate it with:"
  echo " ./bin/main --lang $LANG --max_epochs $NUM_EPOCHS --disable_output --log $TESTDATA --seed $SEED"
  exit 2
fi

TMPFILE=$(mktemp)

cleanup() {
  rm $TMPFILE
}

trap cleanup EXIT

./bin/main --lang $LANG --max_epochs $NUM_EPOCHS --disable_output --log $TMPFILE --seed $SEED

if ! diff -u $TMPFILE $TESTDATA
then
  echo Ground truth
  cat $TESTDATA
  echo
  echo Output 
  cat $TMPFILE
  exit 3
fi

exit 0
