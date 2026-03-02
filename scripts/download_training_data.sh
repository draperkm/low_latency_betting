#!/bin/bash

# Download real tennis data from Jeff Sackmann's repositories

set -e

echo "=== Downloading Real Tennis Data ==="

# Create directories
mkdir -p data/training/raw
mkdir -p data/realtime

# Download point-by-point data (training)
echo "Downloading point-by-point data for training..."
cd data/training/raw
if [ ! -d "tennis_pointbypoint" ]; then
    git clone --depth 1 https://github.com/JeffSackmann/tennis_pointbypoint.git
    echo "✓ Point-by-point data downloaded"
else
    echo "✓ Point-by-point data already exists"
fi
cd ../../..

# Download Match Charting Project (for detailed examples)
echo "Downloading Match Charting Project data..."
cd data/realtime
if [ ! -d "tennis_MatchChartingProject" ]; then
    git clone --depth 1 https://github.com/JeffSackmann/tennis_MatchChartingProject.git
    echo "✓ Match Charting data downloaded"
else
    echo "✓ Match Charting data already exists"
fi
cd ../..

echo ""
echo "=== Data Download Complete ==="
echo ""
echo "Available files:"
echo "  Training: data/training/raw/tennis_pointbypoint/atp_pbp_matches_*.csv"
echo "  Training: data/training/raw/tennis_pointbypoint/wta_pbp_matches_*.csv"
echo "  Real-time: data/realtime/tennis_MatchChartingProject/charting-m-matches.csv"
echo ""
echo "Next step: Run src/tennis_predictor/notebooks/01_data_module_validation.ipynb"
