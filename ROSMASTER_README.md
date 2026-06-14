# ROSMASTER X3

[Official Documentation](http://www.yahboom.net/study/ROSMASTER-X3)

## Connecting to ROSMASTER X3

### SSH
1. Connect to ROSMASTER X3 via Wi-Fi.
2. Open terminal.
3. Run the following command:
   ```bash
   ssh jetson@192.168.1.11
Password: yahboom

### Jupyter Notebook
1. Connect to ROSMASTER X3 via Wi-Fi.
2. Go to http://192.168.1.11:8888/
Password: yahboom

### VNC Viewer (You will have access to the car's display)
1. Connect to ROSMASTER X3 via Wi-Fi.
2. Go to 192.168.1.11
Password: yahboom


## Setting up controller
1. Make sure USB receiver for controller is connected
2. Connect to ROSMASTER X3
3. Run the following commands:
    ```bash
        cd Rosmaster/rosmaster
        python3 joystick_rosmaster.py
4. Turn on controller
5. Hold the mode button until green light flashing
6. Press start button and then there should be a solid green light

## Testing functions (Make sure the correct sensor is selected, for us its S2)

### Gmapping
1. Run this command:
    ```bash
        roslaunch yahboomcar_nav laser_astrapro_bringup.launch
2. In a different terminal run the following:
    ```bash
        roslaunch yahboomcar_nav yahboomcar_map.launch use_rviz:=false map_type:=gmapping
3. In another terminal to control car with keyboard run: 
    ```bash
        rosrun teleop_twist_keyboard teleop_twist_keyboard.py


### Lidar Follow
1. Run this command:
    ```bash
        roslaunch yahboomcar_nav laser_astrapro_bringup.launch
2. In a different terminal run the following:
    ```bash
        roslaunch yahboomcar_laser laser_Tracker.launch
