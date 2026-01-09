#!/bin/bash

# Target directory
TARGET_DIR="Recordings"
FILE_NAME="labels.yaml"

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

echo "Choose an option for content inside $FILE_NAME:"
echo "1) Change all (...) to [...]"
echo "2) Change all [...] to (...)"
read -p "Enter choice [1 or 2]: " choice

case $choice in
    1)
        # Replaces ( with [ and ) with ]
        SED_COMMAND="s/(\([^)]*\))/[\1]/g"
        # Simple version if you just want to swap every char regardless of matching:
        # SED_COMMAND="s/(/[/g; s/)/]/g" 
        DESC="() -> []"
        ;;
    2)
        # Replaces [ with ( and ] with )
        SED_COMMAND="s/\[\([^]]*\)\]/(\1)/g"
        DESC="[] -> ()"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "Updating $FILE_NAME files in $TARGET_DIR subdirectories..."

# Find all labels.yaml files and apply sed
find "$TARGET_DIR" -type f -name "$FILE_NAME" -exec sed -i "$SED_COMMAND" {} +

echo "Successfully updated brackets ($DESC) in all $FILE_NAME files."