# Hardware Setup Guide

This guide walks you through setting up the hardware and software components for the gait analysis system.

## Table of Contents
- [Hardware Setup](#hardware-setup)
- [Software Setup](#software-setup)
- [Using the System](#using-the-system)

---

## Hardware Setup

### Step 1: Install Mobile Applications

Download and install the following applications on your phone:

- **IP Webcam** - Download from [Google Play Store](https://play.google.com/store)
- **Sensor Server** - Download from [F-Droid Store](https://f-droid.org/)

### Step 2: Download Project Files

Download the following files and place them in the same folder:

- `main.py`
- `real_time_plot.py`
- `classifier.py`
- `homography_correction.py`
- `lightglue`
- `segformer`
- `vitpose_local`

### Step 3: Configure ESP32 Camera

1. Download `esp32_web_server` from this repository
2. Open `CameraWebServer.ino` in Arduino IDE
3. Update the WiFi credentials in the code:

   ```cpp
   const char* ssid = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```

   ![WiFi Configuration](https://github.com/user-attachments/assets/0a8b7d85-0e23-4431-bf4e-179895f3cf03)

4. Upload the code to the ESP32
5. Open the Serial Monitor to view the server IP address (save this for later use):

   ![ESP32 IP Address](https://github.com/user-attachments/assets/c210eada-2315-4bcb-a7b1-86b80a8e1444)

   Example output: `192.168.1.100`

### Step 4: Start Mobile Applications

1. Open **IP Webcam** app and press the **Start** button in the menu
2. Open **Sensor Server** app and press the **Start** button on the main screen
3. Both apps will display their server addresses in the format `X.Y.Z.W:P` (IP:Port)
4. **Save both addresses** for the next step

---

## Software Setup

### Step 5: Configure IP Addresses

Update the IP addresses in the following Python files:

1. **`main.py`**
   - Set the `address` variable to the IP:Port from **Sensor Server**
   
   ```python
   address = "192.168.1.XXX:XXXX"  # From Sensor Server
   ```

2. **`homography_correction.py`**
   - Set the `address` variable to the IP:Port from **IP Webcam**
   
   ```python
   address = "192.168.1.XXX:XXXX"  # From IP Webcam
   ```

3. **`classifier.py`**
   - Set the `address` variable to the IP from **ESP32** (Step 3)
   
   ```python
   address = "192.168.1.XXX"  # From ESP32 Serial Monitor
   ```

### Step 6: Launch the System

Open **four separate terminals** and run the scripts in the following order:

1. **Terminal 1:**
   ```bash
   python real_time_plot.py
   ```

2. **Terminal 2:**
   ```bash
   python classifier.py
   ```

3. **Terminal 3:**
   ```bash
   python homography_correction.py
   ```

4. **Terminal 4:**
   ```bash
   python main.py
   ```

---

## Using the System

### Step 7: Position the Devices

- **Phone**: Attach to your foot
- **ESP32 Camera**: Mount on your chest with a clear view of the lower half of your body

### Step 8: Begin Walking

Start walking to collect gait data.

### Step 9: Yaw Correction Calibration

To calculate and apply yaw correction:

1. In the `main.py` terminal, press **`c`** to start calibration mode
2. Rotate your body and take overlapping photos by pressing **`z`** during rotation
3. Press **`x`** to calculate the full rotation
4. Press **`n`** to apply the correction in practice

