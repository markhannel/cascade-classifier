#!/bin/bash

echo "Populating a new directory..."

name='Features_'

# Check if there is a first instance of Features.
if [ ! -d Features_001 ]; then
    cp -r .master_file Features_001
else

    master_file='./.master_file/'
    LAST=`exec ls -d $name*[0-9][0-9][0-9] | sort -n | tail -1 | sed -e s/[^0-9]//g`
    cp -r './.master_file/' $(printf "$name%02d"$(($LAST + 1)))
fi
