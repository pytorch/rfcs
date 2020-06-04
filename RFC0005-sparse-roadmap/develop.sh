#!/bin/sh
#
# Run this script to process LaTeX comments in GitHub Flavored
# Markdown files and generate the corresponding HTML file.
#
# Installing prerequisities:
#
#   conda install -c conda-forge pandoc
#
# Development hint:
#   ls *.{md,py} | entr -s "bash develop.sh"
#
# Author: Pearu Peterson
# Created: May 2020

python mdlatex.py README.md

pandoc README.md -f gfm -t html --metadata title='RFC0005' -s -o README.html


