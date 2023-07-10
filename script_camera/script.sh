#!/bin/bash
host_depan="192.168."
host_tengah="116."
host_moses=$host_depan$host_tengah"30"
host_jose=$host_depan$host_tengah"156"
host_wisnu=$host_depan$host_tengah"78"
host_syihab="10.178.253.93"
vid_name="SStereo_Take4"

# start recording
echo $vid_name | timeout 0.1 nc -u -b $host_moses 8000 &
echo $vid_name | timeout 0.1 nc -u -b $host_jose 8000 &
echo $vid_name | timeout 0.1 nc -u -b $host_wisnu 8000
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