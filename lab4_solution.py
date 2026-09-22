"""Точка запуска актуального решения лабораторной работы №4."""
import argparse
from pathlib import Path
from lab4_recalculation import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('Daily_Water_Intake.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('lab4_reworked'))
    args = parser.parse_args()
    run(args.data, args.output_dir)
