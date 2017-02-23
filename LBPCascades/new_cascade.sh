#!/bin/bash

echo "Populating a new directory..."

# Find the last created file and generate a new one.
name='Features_'
master_file='./.master_file/'
LAST=`exec ls -d $name*[0-9][0-9][0-9] | sort -n | tail -1 | sed -e s/[^0-9]//g`
cp -r './.master_file/' $(printf "$name%02d"$(($LAST + 1)))
