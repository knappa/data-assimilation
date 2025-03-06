#!/bin/bash

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix g-{} --measurements grass --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix s-{} --measurements sheep --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix w-{} --measurements wolves --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix wg-{} --measurements wolves+grass --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix ws-{} --measurements wolves+sheep --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix sg-{} --measurements sheep+grass --matchmaker yes

seq -f "%03g" 0 100 | parallel --tag --linebuffer --jobs 4 python ekf.py --prefix wsg-{} --measurements wolves+sheep+grass --matchmaker yes
