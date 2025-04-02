#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train the TrOCR model from scratch with checkpoints saved locally.
This version bypasses Google Drive mounting and is suitable for command-line/terminal use.
"""

import os
import subprocess
import sys

print("Starting fresh training with checkpoints saved locally")

# Environment variable to force local storage mode
os.environ["FORCE_LOCAL_STORAGE"] = "1"

# Run the training script with no checkpoint to start fresh
try:
    result = subprocess.run(
        ["python", "fine_tune_trocr.py", "--resume_checkpoint=None"],
        check=True,
        text=True
    )
    print("Training completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error during training: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("Training interrupted by user")
    sys.exit(0) 