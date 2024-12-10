#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(dirname $(dirname $(dirname $(realpath "$0"))))))

ray start --head --port=6379 --resources '{"headNode": 100000}' --dashboard-host=0.0.0.0 --disable-usage-stats