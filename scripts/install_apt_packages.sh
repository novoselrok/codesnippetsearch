#!/bin/bash

sudo apt update

sudo apt install build-essential python3-pip python3-dev libpq-dev postgresql postgresql-contrib nginx curl zip unzip

sudo -H pip3 install --upgrade pip
sudo -H pip3 install setuptools wheel
sudo -H pip3 install virtualenv
