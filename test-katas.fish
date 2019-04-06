#!/usr/bin/env fish

set -g error 0

function test_kata_readme -d "Check if a kata does not have a README"
    set -l matches $argv[1]/README.*
    if test -z "$matches"
        echo "✖ No README for '$argv[1]' kata"
        set -g error (math $error + 1)
    end
end

function test_kata_siblings -d "Check if a kata has non-directory, non-README files"
    set -l extra_files (find $argv[1] -maxdepth 1 -type f -not -name "README.*")
    if test -n "$extra_files"
        echo "✖ Extra files found for '$argv[1]' kata"
        for f in $extra_files
            echo "  - $f"
        end
        set -g error (math $error + 1)
    end
end

for kata in */
    test_kata_readme $kata
end

for kata in */
    test_kata_siblings $kata
end

if test $error -eq 0
    echo ""
    echo "✓ All tests passed"
else
    echo ""
    echo "$error failed case(s)"
    exit 1
end
