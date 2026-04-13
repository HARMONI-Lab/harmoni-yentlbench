#!/usr/bin/env python3
"""
DEPRECATED: ESI Triage Gender Bias Analysis

This script has been deprecated in favor of the unified Attention Pipeline.
"Gender Bias" is now treated as a subset of "Attention Analysis", and the 
omnibus statistical tests (Cochran’s Q, Friedman, etc.) have been moved 
to `attention_pipeline/analyze_statistical_significance.py`.

Please use `attention_pipeline/pipeline.py` instead.
"""

import sys

def main():
    print("WARNING: bias_analysis.py is deprecated.", file=sys.stderr)
    print("Please use the unified pipeline: python attention_pipeline/pipeline.py", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
