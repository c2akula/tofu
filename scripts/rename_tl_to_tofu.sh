#!/bin/bash
# Rename tl_ → tofu_ across the entire codebase
# This is a BREAKING CHANGE for API stabilization (Milestone 4)

set -e

echo "=== Tofu API Rename Script ==="
echo "This will rename tl_ → tofu_ and TL_ → TOFU_"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Count occurrences before
BEFORE_COUNT=$(grep -r "tl_" src test examples --include="*.c" --include="*.h" 2>/dev/null | wc -l | tr -d ' ')
echo -e "${YELLOW}Found ${BEFORE_COUNT} occurrences of 'tl_' to rename${NC}"
echo ""

# Create backup
BACKUP_DIR="/tmp/tofu_rename_backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup at: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp -r src test examples "$BACKUP_DIR/"
echo -e "${GREEN}✓ Backup created${NC}"
echo ""

# Find all C and H files
FILES=$(find src test examples -type f \( -name "*.c" -o -name "*.h" \))

echo "Renaming in $(echo "$FILES" | wc -l | tr -d ' ') files..."
echo ""

# Perform replacements
for file in $FILES; do
    echo "Processing: $file"

    # Use sed with backup to rename patterns
    # Pattern 1: tl_ → tofu_ (function/type prefix)
    # Pattern 2: TL_ → TOFU_ (macro prefix)

    sed -i.bak \
        -e 's/tl_/tofu_/g' \
        -e 's/TL_/TOFU_/g' \
        "$file"

    # Remove backup file
    rm -f "$file.bak"
done

echo ""
echo -e "${GREEN}✓ Replacements complete${NC}"
echo ""

# Count occurrences after
AFTER_COUNT=$(grep -r "tl_" src test examples --include="*.c" --include="*.h" 2>/dev/null | wc -l | tr -d ' ')
RENAMED_COUNT=$((BEFORE_COUNT - AFTER_COUNT))

echo "=== Summary ==="
echo "Before: $BEFORE_COUNT occurrences"
echo "After:  $AFTER_COUNT occurrences"
echo "Renamed: $RENAMED_COUNT occurrences"
echo ""

if [ "$AFTER_COUNT" -eq "0" ]; then
    echo -e "${GREEN}✓ All tl_ patterns renamed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Warning: ${AFTER_COUNT} tl_ patterns remain (may be in comments/strings)${NC}"
fi

echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review changes: git diff"
echo "2. Build: make clean && ./configure && make lib"
echo "3. Test: build and run validation tests"
echo "4. If everything works: rm -rf $BACKUP_DIR"
echo "5. If something breaks: cp -r $BACKUP_DIR/* ."
