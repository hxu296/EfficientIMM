#!/bin/bash

echo 'Running dataset download script'

pip3 install gdown charset_normalizer chardet
gdown --id 1CRNC2NjSQ5B1_Jngbg_G4uCZzgWbG83Q
tar -xvf test-data.tar.gz