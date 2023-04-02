#!/bin/bash

find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.gif" \) | while read file
do
  size=$(identify -format "%wx%h" "$file")
  echo "$file, $size"
done

