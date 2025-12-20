#!/bin/bash
cd "$(dirname "$0")"
echo "Submitting fvqa jobs..."
(cd fvqa && bash run.sh)
echo "Submitting aokvqa jobs..."
(cd aokvqa && bash run.sh)

