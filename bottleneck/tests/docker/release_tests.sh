#!/bin/bash
set -e

rm -rf temp_clone
git clone ../../.. temp_clone

cases=(centos_7_min_deps centos_8_min_deps ubuntu_lts_min_deps ubuntu_devel_min_deps)
for case in ${cases[@]}; do
    echo $case
    docker build -t $case $case
    docker run --mount type=bind,source=$(pwd)/temp_clone,destination=/bottleneck_src,readonly $case
done

rm -rf temp_clone