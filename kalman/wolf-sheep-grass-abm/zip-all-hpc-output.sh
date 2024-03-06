#!/bin/bash

zip -rm w-data.zip ekf-wolf-array_* w-*.hdf5
zip -rm s-data.zip ekf-sheep-array_* s-*.hdf5
zip -rm g-data.zip ekf-grass-array_* g-*.hdf5
zip -rm ws-data.zip ekf-wolf-sheep-array_* ws-*.hdf5
zip -rm wg-data.zip ekf-wolf-grass-array_* wg-*.hdf5
zip -rm sg-data.zip ekf-sheep-grass-array_* sg-*.hdf5
zip -rm wsg-data.zip ekf-wolf-sheep-grass-array_* wsg-*.hdf5
