#!/bin/sh
set -e

check_copyright_notices() {
    exitcode=0
    for file in $1; do
        first_line=$(head -n1 "$file")
        
        if echo "$first_line" | grep -q "// Copyright .* Ulvetanna Inc."; then
            echo "$file: ERROR - Copyright notice is using Ulvetanna instead of Irreducible"
            exitcode=1
        elif ! echo "$first_line" | grep -q "// Copyright "; then
            echo "$file: ERROR - Copyright notice missing on first line"
            exitcode=1
        elif ! echo "$first_line" | grep -q "2025"; then
            echo "$file: ERROR - Copyright notice does not contain the year 2025"
            exitcode=1
        fi
    done
    exit $exitcode
}

check_copyright_notices "$(find crates examples -type f -name '*.rs')"
