#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(realpath $0)))

PYTHON_SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
$PACKAGE_PTH_PATH="$PYTHON_SITE_PACKAGES_DIR/code_search.pth"
echo $ROOT_DIR > $PACKAGE_PTH_PATH
