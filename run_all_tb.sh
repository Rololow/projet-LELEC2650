#!/bin/bash
# =============================================================================
# LELEC2650 - OTA Cascode Miller - Run All Testbenches
# =============================================================================
# Usage: ./run_all_tb.sh
# Requires: Eldo simulator available in PATH
# =============================================================================

echo "=============================================="
echo "  LELEC2650 - OTA Cascode Miller Simulations"
echo "=============================================="
echo ""

# Directory containing testbenches
TB_DIR="."

# List of testbenches to run
TESTBENCHES=(
    "TB_AC.cir"
    "TB_FOM.cir"
    "TB_SR.cir"
    "TB_STEP.cir"
    "TB_NOISE.cir"
    "TB_CMRR.cir"
    "TB_PSRR.cir"
    # "TB_MC.cir"    # Commented out - long simulation (1000 runs)
    # "TB_PVT.cir"   # Commented out - 16 corners
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for results
PASSED=0
FAILED=0

echo "Starting simulations at $(date)"
echo ""

for TB in "${TESTBENCHES[@]}"; do
    if [ -f "${TB_DIR}/${TB}" ]; then
        echo -n "Running ${TB}... "
        
        # Run Eldo simulation
        eldo -i "${TB_DIR}/${TB}" -o "${TB%.cir}.log" > /dev/null 2>&1
        
        # Check exit status
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}OK${NC}"
            ((PASSED++))
        else
            echo -e "${RED}FAILED${NC}"
            ((FAILED++))
        fi
    else
        echo -e "${YELLOW}SKIP${NC} - ${TB} not found"
    fi
done

echo ""
echo "=============================================="
echo "  Summary"
echo "=============================================="
echo -e "  Passed: ${GREEN}${PASSED}${NC}"
echo -e "  Failed: ${RED}${FAILED}${NC}"
echo ""
echo "Finished at $(date)"

# Run Monte Carlo and PVT if requested
if [ "$1" == "--full" ]; then
    echo ""
    echo "Running full simulations (MC + PVT)..."
    echo ""
    
    echo -n "Running TB_MC.cir (1000 runs)... "
    eldo -i "${TB_DIR}/TB_MC.cir" -o "TB_MC.log" > /dev/null 2>&1
    if [ $? -eq 0 ]; then echo -e "${GREEN}OK${NC}"; else echo -e "${RED}FAILED${NC}"; fi
    
    echo -n "Running TB_PVT.cir (16 corners)... "
    eldo -i "${TB_DIR}/TB_PVT.cir" -o "TB_PVT.log" > /dev/null 2>&1
    if [ $? -eq 0 ]; then echo -e "${GREEN}OK${NC}"; else echo -e "${RED}FAILED${NC}"; fi
fi

echo ""
echo "Log files saved as TB_*.log"
