#!/bin/sh
set -e

check_copyright_notices() {
    exitcode=0
    for file in $1; do
        if (head -n1 "$file" | grep -q "// Copyright .* Ulvetanna Inc."); then
            echo "$file: ERROR - Copyright notice is using Ulvetanna instead of Irreducible"
            exitcode=1
        elif !(head -n1 "$file" | grep -q "// Copyright "); then
            echo "$file: ERROR - Copyright notice missing on first line"
            exitcode=1
        fi
    done
    exit $exitcode
}

check_copyright_notices "$(
    find crates -type f -name '*.rs';
    find examples -type f -name '*.rs'
)"
