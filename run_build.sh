#!/bin/bash

echo 'Running setup script for `efficient IMM`'

export agile_WF=$PWD

conan create conan/waf-generator user/stable
conan create conan/trng 4.22@user/stable
conan install --install-folder build . --build 

./waf configure
./waf build_release
./waf build_release --build-ripples

echo $PWD
cd $agile_WF