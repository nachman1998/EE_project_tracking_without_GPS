setting up the hardware
step 1: download to phone "IP Wabcam" from google-play store
and download to phone "Sensor Server" from F-droid store
step 2: download "main.py","real_time_plot.py","classifier.py","homography_correction.py","lightglue","segformer","vitpose_local"
and put them together in same folder. 
setp 3: download "esp32_web_server" from this repository and open "CameraWebServer.ino" in Arduino IDE,
and change wifi name and pass like in photo below.
<img width="440" height="157" alt="image" src="https://github.com/user-attachments/assets/0a8b7d85-0e23-4431-bf4e-179895f3cf03" />
upload the code to the ESP32.
in the Serial Monitor you will see the follwing ip of the server that will be uesd in next steps.
<img width="782" height="72" alt="image" src="https://github.com/user-attachments/assets/c210eada-2315-4bcb-a7b1-86b80a8e1444" />
step 4: open "IP Wabcam" and "press start" butten in menu.
in "Sensor Server" in main screen press "start" butten
in both it will show the  "X.Y.Z.W:P" ip+port for those servers, save those adrresses.

setting up the software
step 5: In "main.py" put the ip+port from "Sensor Server" in adrress variable in code.
and in "homography_correction.py" put the ip+port from "IP Wabcam" in adrress variable in code.
and in "classifier.py" put the ip from "step 3" in adrress variable in code.
step 6: open four treminals. first on one terminal run "real_time_plot.py".
then on two others run "classifier.py","homography_correction.py".
then run on last treminal "main.py".

using the devices
step 7: put phone on foot, the esp32 camera on chest with clear view of the lower half body.
step 8: start walking.
step 9: for calculating yaw correction. in "main.py" terminal press "c" then as you rotate take overlapping photos in diraction of rotation by pressing "z".
then perss "x" for calculating full rotation and then "n" for correcting in practice


