#!/bin/bash
#Author: Adam Jaamour
# ----------------------------
echo "Setting Dissertation remote Jupyter environment"
source /home/eduarda/tcc/tf2/venv/bin/activate
cd  ~/Projects/Breast-Cancer-Detection-and-Segmentation
jupyter lab --no-browser --port=8888
