#!/bin/bash

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-grass --measurements grass
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-sheep --measurements sheep
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolf --measurements wolves
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolf-grass --measurements wolves+grass
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolf-sheep --measurements wolves+sheep
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-sheep-grass --measurements sheep+grass
done

for i in $(seq -f "%03g" 0 100)
do
  python ekf.py --prefix $i-wolf-sheep-grass --measurements wolves+sheep+grass
done
