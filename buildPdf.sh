#!/bin/bash

pandoc -i README.md -o report.pdf --latex-engine=xelatex
