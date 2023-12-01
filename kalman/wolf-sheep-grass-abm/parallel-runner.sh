#!/bin/bash

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix {}-grass --measurements grass --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix {}-sheep --measurements sheep --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix {}-wolf --measurements wolves --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix {}-wolfgrass --measurements wolves+grass --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix {}-wolfsheep --measurements wolves+sheep --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix {}-sheepgrass --measurements sheep+grass --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix {}-wolfsheepgrass --measurements wolves+sheep+grass --matchmaker yes
