#!/bin/bash

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-grass --measurements grass --matchmaker yes
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-sheep --measurements sheep --matchmaker yes
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolf --measurements wolves --matchmaker yes
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolfgrass --measurements wolves+grass --matchmaker yes
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolfsheep --measurements wolves+sheep --matchmaker yes
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-sheepgrass --measurements sheep+grass --matchmaker yes
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolfsheepgrass --measurements wolves+sheep+grass --matchmaker yes
done
