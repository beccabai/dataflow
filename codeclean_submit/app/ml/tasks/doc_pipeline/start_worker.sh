#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(dirname $(dirname $(dirname $(realpath "$0"))))))
export ip_head
echo "IP Head: $ip_head"

ray start --address $ip_head