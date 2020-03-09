#!/usr/bin/env bash

set -e

mkdir -p _build

cd _build
cmake ..
make -j8
mkdir -p ../python/pyicecake/lib/
cp lib/*.so ../python/pyicecake/lib/
cd ../python

version=`python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";`
ver=py${version:0:1}${version:2:1}

echo '--------------------------------- packing --------------------------------------'
python3 setup.py bdist_wheel --python-tag ${ver}


#rm -rf _build