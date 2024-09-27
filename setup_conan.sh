#!/bin/bash

echo 'Running the first time setup script'

conda install -y \
  -c conda-forge \
  -c esrf-bcu \
  gcc_linux-64=12.2.0 \
  gxx_linux-64=12.2.0 \
  cmake=3.20.4 \
  esrf-bcu::libnuma
  
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)

pip3 install --user "conan==1.59"

if [ ! -f ~/.conan/settings.yml ]; then
    conan config init
fi
conan profile new default --detect &> /dev/null
conan profile update settings.compiler.libcxx=libstdc++11 default

if grep riscv $HOME/.conan/settings.yml; then
    echo RISCV support already added. Skipping.
else
    echo RISCV support added.
    sed --in-place=.bkp 's/x86/x86, riscv/' $HOME/.conan/settings.yml
fi


for i in $(ls conan/); do
    if [ -d $HOME/.conan/data/$i ]; then
        rm -rf $HOME/.conan/data/$i
    fi

    conan create conan/$i user/stable
done
