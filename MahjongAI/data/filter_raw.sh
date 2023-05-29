#!/bin/bash

directory="raw/2021/"

# Delete files that do not start with "scc"
for file in $directory*; do
    if [[ ! $file == $directory"scc"* ]]; then
        rm "$file"
        echo "Deleted file: $file"
    fi
done

# Unzip .gz files in the directory
for file in $directory*.gz; do
    gunzip "$file"
    unzipped_file="${file%.gz}"
    echo "Unzipped file: $unzipped_file"
done
