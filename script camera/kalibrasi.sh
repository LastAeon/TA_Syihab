#!/bin/bash
host_moses="192.168.161.30"
host_jose="192.168.161.156"
host_wisnu="192.168.161.78"
host_syihab="10.178.253.93"


for i in {1..5}
do
   echo "Vtes3" | timeout 1 nc -u -b $host_wisnu 8000
done