#!/bin/bash

zip -rm w-data.zip wsg-ekf-wolf-array_* w-*.hdf5
zip -rm s-data.zip wsg-ekf-sheep-array_* s-*.hdf5
zip -rm g-data.zip wsg-ekf-grass-array_* g-*.hdf5
zip -rm ws-data.zip wsg-ekf-wolf-sheep-array_* ws-*.hdf5
zip -rm wg-data.zip wsg-ekf-wolf-grass-array_* wg-*.hdf5
zip -rm sg-data.zip wsg-ekf-sheep-grass-array_* sg-*.hdf5
zip -rm wsg-data.zip wsg-ekf-wolf-sheep-grass-array_* wsg-*.hdf5
