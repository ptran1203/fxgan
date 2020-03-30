#! /bin/bash
now=$(date +'%Y-%m-%d-%H:%M:%S')
git add .
git commit -m "$now"
git push
echo $now