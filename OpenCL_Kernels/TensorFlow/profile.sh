#! /bin/bash

filename=profile.mon
x=0
var1=0
timeset=0
#dir where we want to save profile.mon
dirToStore=testdir

while true; do
    if [ -f $filename ]
    then
      echo "$filename exists"
     else   
      echo "$filename doesn't exists"
    fi
    x=$(( $x + 1 ))
    sleep 1
    
    #time of last data modification, seconds since Epoch
    newtimeset=$(stat -c %Y "$filename")
    echo "$newtimeset $timeset"
    if [ "$timeset" -lt "$newtimeset" ]
    then
        timeset=$newtimeset
        var1=$(( $var1 + 1 ))
        cp $filename "profile$var1.mon" #copy file with name starting as profile1.mon
    fi
    #exit condition run for 200 sec
    if [ "$x" -eq "200" ]
    then
	echo "exit condtiont trigerred";
   exit;
    fi
done