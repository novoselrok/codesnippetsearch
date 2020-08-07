#!/bin/bash

ROOT_DIR=$(dirname $(dirname $(realpath $0)))

wget -P "$ROOT_DIR/codesearchnet_data" https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,javascript,ruby}.zip
unzip "$ROOT_DIR/codesearchnet_data/*.zip" -d "$ROOT_DIR/codesearchnet_data/"
rm $ROOT_DIR/codesearchnet_data/*.zip
