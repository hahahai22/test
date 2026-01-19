#!/bin/bash

filename="001.txt"

result=$(grep "min_time :" $filename | sed -n 's/.*min_time :\s*$[0-9]*$.*/\1\p')

echo "$result"
