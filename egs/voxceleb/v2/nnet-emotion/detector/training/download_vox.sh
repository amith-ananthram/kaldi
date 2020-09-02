#!/bin/bash

USERNAME="voxceleb1904"
PASSWORD="9hmp7488"

VOX1_DEV_DOWNLOAD_URLS=(
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad'
)
VOX1_TEST_DOWNLOAD_URLS=(
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip'
)
VOX2_DEV_DOWNLOAD_URLS=(
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag'
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah'
)
VOX2_TEST_DOWNLOAD_URLS=(
  'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip'
)
if [[ "$1" == "vox1" ]]; then
  if [[ "$2" == "dev" ]]; then
    to_download=("${VOX1_DEV_DOWNLOAD_URLS[@]}")
  elif [[ "$2" == "test" ]]; then
    to_download=("${VOX1_TEST_DOWNLOAD_URLS[@]}")
  fi
elif [[ "$1" == "vox2" ]]; then
  if [[ "$2" == "dev" ]]; then
    to_download=("${VOX2_DEV_DOWNLOAD_URLS[@]}")
  elif [[ "$2" == "test" ]]; then
    to_download=("${VOX2_TEST_DOWNLOAD_URLS[@]}")
  fi
fi

for i in "${to_download[@]}"; do
  echo "Downloading $i"
  wget --user $USERNAME --password $PASSWORD $i
done
