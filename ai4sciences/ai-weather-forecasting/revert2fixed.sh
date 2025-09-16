#!/bin/bash

# Read the original dockerfile and create a processed copy
input_file=$1
output_file=$1-fixed

# Check if input file exists
if [[ ! -f "$input_file" ]]; then
    echo "Error: $input_file not found"
    exit 1
fi

# Read all lines into an array
lines=()
while IFS= read -r line || [[ -n "$line" ]]; do
    lines+=("$line")
done < "$input_file"

# Process lines - build output array instead of modifying file repeatedly
output_lines=()

for i in "${!lines[@]}"; do
    current_line="${lines[i]}"
    
    # Check if current line starts with "#### "
    if [[ "$current_line" =~ ^####[[:space:]] ]]; then
        # Comment out the previous line if it exists
        if [[ $i -gt 0 ]]; then
            # Modify the last added line to be commented
            last_index=$((${#output_lines[@]} - 1))
            output_lines[$last_index]="# ${output_lines[$last_index]}"
        fi
        
        # Remove "#### " substring from current line
        modified_line="${current_line#*#### }"
        output_lines+=("$modified_line")
    else
        output_lines+=("$current_line")
    fi
done

# Write all output lines to file
printf '%s\n' "${output_lines[@]}" > "$output_file"

echo "Processed dockerfile saved as $output_file"
