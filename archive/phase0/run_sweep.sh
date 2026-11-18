#!/bin/bash
set -e

PYTHON=".venv/bin/python"

echo "=== Running gate_sweep ==="
$PYTHON cl.py --methods gate:conflict:0.5 gate:conflict:1.0 gate:layer_cos:1.0 gate:magnitude:0.5 > logs/gate_sweep.log 2>&1
mv results.json outputs/results_gate_sweep.json
mv forgetting_curve.png outputs/forgetting_curve_gate_sweep.png

echo "=== Running ewc_1000 ==="
$PYTHON cl.py --methods ewc --ewc_lambda 1000 > logs/ewc_1000.log 2>&1
mv results.json outputs/results_ewc_1000.json
mv forgetting_curve.png outputs/forgetting_curve_ewc_1000.png

echo "=== Running ewc_10000 ==="
$PYTHON cl.py --methods ewc --ewc_lambda 10000 > logs/ewc_10000.log 2>&1
mv results.json outputs/results_ewc_10000.json
mv forgetting_curve.png outputs/forgetting_curve_ewc_10000.png

echo "=== Running gate_decay09 ==="
$PYTHON cl.py --methods gate:conflict:1.0 gate:magnitude:1.0 --decay 0.9 > logs/gate_decay09.log 2>&1
mv results.json outputs/results_gate_decay09.json
mv forgetting_curve.png outputs/forgetting_curve_gate_decay09.png

echo "=== Running epochs5 ==="
$PYTHON cl.py --methods finetune ewc gate:conflict:1.0 --epochs 5 > logs/epochs5.log 2>&1
mv results.json outputs/results_epochs5.json
mv forgetting_curve.png outputs/forgetting_curve_epochs5.png

echo "=== All sweeps finished successfully ==="
