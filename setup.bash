#!/usr/bin/env bash

if [ -d "venv" ]; then
    :
else
    virtualenv venv -p python3
fi
. venv/bin/activate
pip3 install -U pip
pip3 install -U -r requirements.txt
