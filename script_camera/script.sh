#!/bin/bash
host_moses="192.168.189.30"
host_jose="192.168.189.156"
host_wisnu="192.168.189.78"
host_syihab="10.178.253.93"

# start recording
echo "Vgerakan5" | timeout 0.1 nc -u -b $host_moses 8000 &
echo "Vgerakan5" | timeout 0.1 nc -u -b $host_jose 8000 
# echo "Vgerakan5" | timeout 0.1 nc -u -b $host_wisnu 8000
# echo "Vtes4" | timeout 0.1 nc -u -b $host_syihab 8000

# # time for recording
# sleep 5

# # stop recording
# echo "Vtes1" | timeout 0.1 nc -u -b $host_moses 8000 &
# echo "Vtes1" | timeout 0.1 nc -u -b $host_jose 8000 &
# echo "Vtes1" | timeout 1 nc -u -b $host_wisnu 8000
# # echo "Vtes4" | timeout 0.1 nc -u -b $host_syihab 8000

# echo "Vtes1" | timeout 1 nc -u -b $host_moses 8000 &
# echo "Vtes2" | timeout 1 nc -u -b $host_jose 8000 &
# echo "Vtes3" | timeout 1 nc -u -b $host_wisnu 8000 &
# echo "Vtes4" | timeout 1 nc -u -b $host_syihab 8000