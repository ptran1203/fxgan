#! /bin/bash
now=$(date +'%Y-%m-%d-%H:%M:%S')
git add .
git commit -m "$now"
git ps
echo $now