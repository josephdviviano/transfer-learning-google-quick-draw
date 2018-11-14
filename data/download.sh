#!/bin/bash

if ( ! -f test_images.npy); do
    echo "if this command fails, you need to install the kaggle API -- https://github.com/Kaggle/kaggle-api"
    kaggle competitions download -c ift3395-6390-f2018
    unzip test_images.npy.zip
    unzip train_images.npy.zip
    rm *.zip
done
