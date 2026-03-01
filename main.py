import socket
import threading
import json
import cv2
import multiprocessing
import websocket
import atexit
import math
from scipy import constants as const
from collections import deque
import torch
import numpy as np
from PIL import Image
import time
import select
import sys
import os
import msvcrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import datetime
from multiprocessing.connection import Client
import traceback
import os
import psutil

#set as real time Process
p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)



formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
const_dt=0.0025





# Global prams for calibration of roll and pitch
calib_with_g=1200
strong_calib_with_g=5
# Global prams for calibration of gyro bias
calib_ang=1200
offset=np.array([[0.0],[0.0],[0.0]])
#for delta t between mesurments
last_gyr=None
last_acc=None

# Global prams for Identifying the beginning and end of a step
non_move_after=40
non_move_before=40
before_num_non_move=non_move_before
curr_and_after_num_non_move=0
end_of_step=True
yaw_at_end_of_step=0
flag_yaw_at_end_of_step=True
acc_delay_que=deque()
acc_for_calc=deque()
acc_dt=deque()


start_time=time.time()
curr_time=time.time()

#data  queue for incoming data from websocket
t_acc_q=deque()
t_gyr_q=deque()

end_ondata_loop=False
end_plot_loop=False
# roll,pitch,yaw vector
X_n=np.array([0,0,0], dtype=float).reshape(-1,1)
# veriance matrix
P_n=(0.0001)*np.eye(3,3)
#current position vector
Pos_n=np.array([0,0,0], dtype=float).reshape(-1,1)


x=deque()
y=deque()
z=deque()
label=deque()




def pitch_from_g_321(x_acc,y_acc,z_acc):
    return math.asin((-x_acc)/(((x_acc)**2+(y_acc)**2+(z_acc)**2)**0.5))

def roll_from_g_321(x_acc,y_acc,z_acc,mu=0.005):
    return math.atan2((y_acc),(z_acc))

def norm_pi_2_neg_pi(ang):
    if ang>math.pi:
        return ang-(2*math.pi)
    if ang<=-math.pi:
        return ang + (2 * math.pi)
    return ang

def A(dt=const_dt):
    return np.array([[1,0,0]
                    ,[0,1,0]
                    ,[0,0,1]])
def B(dt=const_dt):
    return np.array([[dt,0,0],
                     [0,dt,0],
                     [0,0,dt]])
def C(mag=False):
    if mag==False:
        return np.array([ [1, 0, 0]
                        , [0, 1, 0]
                        , [0, 0, 0]])
    else:
        return np.array([ [1, 0, 0]
                        , [0, 1, 0]
                        , [0, 0, 1]])

def E_R(roll,pitch):
    return np.array([[1, math.sin(roll)*math.tan(pitch), math.cos(roll)*math.tan(pitch)]
                    ,[0,math.cos(roll),-1*math.sin(roll)]
                    ,[0,math.sin(roll)*(1/math.cos(pitch)),math.cos(roll)*(1/math.cos(pitch))]])


def Q():
    return np.array([ [0.0001, 0,0]
                    , [0, 0.0001, 0]
                    , [0, 0, 0.0001]])

def R_moving(mag=False):
    if mag==False:
        return np.array([ [200000, 0, 0]
                        , [0, 200000, 0]
                        , [0, 0, 0]])
    else:
        return np.array([ [200000, 0, 0]
                        , [0,200000, 0]
                        , [0, 0,0.2]])

def R_not_moving(mag=False):
    if mag==False:
        return np.array([ [0.1, 0, 0]
                        , [0, 0.1, 0]
                        , [0, 0, 0]])
    else:
        return np.array([ [0.1, 0, 0]
                        , [0,0.1, 0]
                        , [0, 0,0.2]])

def K(P_hat_n_plus1,C,R):
    return P_hat_n_plus1@C.T@np.linalg.pinv(C@P_hat_n_plus1@C.T+R)

# calculates rotaion matrix from roll,pitch,yaw in xyz oreder
def Rot(roll,pitch,yaw):

    s_r,s_p,s_y=math.sin(roll),math.sin(pitch),math.sin(yaw)
    c_r,c_p,c_y=math.cos(roll),math.cos(pitch),math.cos(yaw)
    return np.array([[c_y*c_p,s_y*c_p,-s_p],
                     [(c_y*s_p*s_r)-(c_r*s_y),(s_y*s_p*s_r)+(c_r*c_y),c_p*s_r],
                     [(c_r*s_p*c_y)+(s_r*s_y),(s_y*s_p*c_r)-(s_r*c_y),c_p*c_r]])


# calculates roll,pitch,yaw instantaneous change from gyro
#updates X_n,P_n
def orantaion_update(gyr_vec,X_n,P_n,dt=const_dt):

    # perdict case 1
    X_hat_n_plus1 = A() @ X_n + B(dt) @ E_R(X_n[0, 0], X_n[1, 0]) @ gyr_vec #
    P_hat_n_plus1 = A() @ P_n @ (A().T) + Q()
    X_hat_n_plus1[2] = norm_pi_2_neg_pi(X_hat_n_plus1[2])

    return (X_hat_n_plus1, P_hat_n_plus1)

# Kalman filter correcting roll and pitch and updates X_n,P_n
def kalman(acc_vec,X_n,P_n,R,dt=const_dt):
    roll_g_321 = roll_from_g_321(acc_vec[0, 0], acc_vec[1, 0], acc_vec[2, 0])
    pitch_g_321 = pitch_from_g_321(acc_vec[0, 0], acc_vec[1, 0], acc_vec[2, 0])

    X_n_plus1 = X_n + K(P_n, C(), R()) @ (np.array([[roll_g_321], [pitch_g_321], [0]]) - C() @ X_n)
    P_n_plus1 = P_n - K(P_n, C(), R()) @ C() @ P_n

    return (X_n_plus1, P_n_plus1)

# change accelerometer coordinates to world coordinates and cancel gravity
def acc_earth(X_n,acc_vec):
    return (Rot(X_n[0, 0], X_n[1, 0], X_n[2, 0]).T @ acc_vec) + np.array([[0], [0], [const.g]])


# websocket listiner for accelerometer
def on_accelerometer_event(values, timestamp):
    global t_acc_q
    t_acc_q.append([values, timestamp])

# websocket listiner for gyro
def on_gyroscope_event(values, timestamp):
    global t_gyr_q
    t_gyr_q.append([values, timestamp])

# class for websocket lisiner
class Sensor:

    def __init__(self, address, sensor_type, on_sensor_event):
        self.address = address
        self.sensor_type = sensor_type
        self.on_sensor_event = on_sensor_event
        self.ws=None
    def on_message(self, ws, message):
        values = json.loads(message)['values']
        timestamp = json.loads(message)['timestamp']

        self.on_sensor_event(values=values, timestamp=timestamp)

    def on_error(self, ws, error):
        print("error occurred")
        print(error)

    def on_close(self, ws, close_code, reason):
        print(f"close {self.sensor_type}")


    def on_open(self, ws):
        print(f"connected to : {self.address}")

    def disconnect(self):
        if self.ws:
            self.ws.close()



    def make_websocket_connection(self):
        self.ws = websocket.WebSocketApp(f"ws://{self.address}/sensor/connect?type={self.sensor_type}",
                                    on_open=self.on_open,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)

        # blocking call
        self.ws.run_forever()

    # make connection and start recieving data on sperate thread
    def connect(self):
        thread = threading.Thread(target=self.make_websocket_connection)
        thread.start()


# classifier for sitting,Stairs,Crab-walk,standing in place that we defined in book projerct
def classifier(yaw,position,siting_state):

    start_index=0
    stop_index=-1
    dir_vec=np.array([[math.cos(yaw),math.sin(yaw),0],[-math.sin(yaw),math.cos(yaw),0],[0,0,1]])@((position[stop_index] - position[start_index]).reshape((-1,1)))

    if abs((math.sqrt((position[stop_index, 0] - position[start_index, 0]) ** 2 + (position[stop_index, 1] - position[start_index, 1]) ** 2 )))<0.05 and max(position[:,2])>0.08:
        if siting_state=='siting':
            return 'siting'
        return 'standing_still'
    if abs((math.sqrt((position[stop_index, 0] - position[start_index, 0]) ** 2 + (position[stop_index, 1] - position[start_index, 1]) ** 2 )))<0.05:
        if siting_state=='siting':
            return 'siting'
        return 'no_movement'
    if position[stop_index, 2] - position[start_index, 2]<-0.13:
        return 'upstairs'
    if position[stop_index, 2] - position[start_index, 2]>0.13:
        return 'downstairs'

    par_ang=30
    ang1=par_ang;ang2=180-par_ang;ang4=-par_ang;ang3=-180+par_ang
    ang_y_dir=math.degrees(math.atan2(dir_vec[1,0],dir_vec[0,0]))
    if ang_y_dir<ang2 and ang_y_dir>ang1:
        return 'forward '
    elif ang_y_dir<ang4 and ang_y_dir>ang3:
        return 'backward '
    elif  ang_y_dir<ang1 and ang_y_dir>ang4:
        return 'left '
    else:
        return 'right '

#calculates step end coordinates and class
def step_calc(acc_vec_arr,acc_dt,yaw_at_start,siting_state):
    global Pos_n
    global const_dt
    global log_pos
    global start_time
    global curr_time
    global x, y, z, label
    global conn

    velocity = np.zeros((len(acc_vec_arr)+1, 3))

    for index in range(1,len(acc_vec_arr)+1):
        velocity[index] = velocity[index - 1] + (acc_dt[index-1] * acc_vec_arr[index-1].flatten())
    #print(velocity)
    drift_vel=velocity[-1]/(len(acc_vec_arr)+1)
    #print(drift_vel)
    position = np.zeros((len(acc_vec_arr)+1, 3))

    for index in range(1,len(acc_vec_arr)+1):
        position[index] = position[index - 1] + (acc_dt[index-1] * (velocity[index]-(drift_vel*index)))

    Pos_n=Pos_n+position[-1].reshape((-1,1))
    step_class=classifier(yaw_at_start,position,siting_state)
    if step_class!='no_movement':
        try:
            conn.send((Pos_n[0,0], Pos_n[1,0], Pos_n[2,0], step_class, time.time()-start_time))
            print(f"{Pos_n[0, 0]},{Pos_n[1, 0]},{Pos_n[2, 0]},{step_class},{time.time() - start_time}\n")
        except Exception as e:
            print("3d map send problem")
    log_pos.write(f"{Pos_n[0,0]},{Pos_n[1,0]},{Pos_n[2,0]},{step_class},{time.time()-start_time}\n")




# main loop sort incoming data in to diffrent cases
def ondata():
    global X_n
    global P_n
    global Pos_n
    global before_num_non_move
    global curr_and_after_num_non_move
    global end_of_step
    global acc_delay_que
    global acc_for_calc
    global non_move_after
    global non_move_before
    global flag_yaw_at_end_of_step
    global yaw_at_end_of_step
    global log_pos
    global log_orination
    global calib_with_g
    global strong_calib_with_g
    global calib_ang
    global offset
    global last_gyr
    global last_acc
    global acc_dt
    global end_ondata_loop

    sit_server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    host=''
    port=6009
    sit_server.setblocking(0)
    sit_server.bind((host,port))
    siting_state='no_siting'

    hom_server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    host='localhost'
    port=6023
    hom_server.setblocking(0)
    hom_server.bind((host,port))

    hom_send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    timer=time.time()
    orant_save=X_n.copy()
    tmp_orant_save = None
    pos_sav=Pos_n.copy()
    a=0.5

    while True:

        if len(t_acc_q) > 1000 or len(t_gyr_q) > 1000:
            print("len(t_acc_q)", len(t_acc_q), "len(t_gyr_q)", len(t_gyr_q))
        #reciving flag if siting or not form the sitting identifier program running in parallel
        try:
            s = time.perf_counter()
            ready, _, _ = select.select([sit_server], [], [], 0)
            for s in ready:
                siting_state=s.recv(30).decode()
                print(siting_state)
        except Exception as e:
            print("udp prob")


        if end_ondata_loop==True:
            log_pos.flush()
            log_pos.close()
            log_orination.flush()
            log_orination.close()
            print("did it2")
            break


        sensorType =""
        timestamp = 0
        values = []
        if len(t_acc_q)==0 or len(t_gyr_q)==0:
            time.sleep(0.001)
            continue
        if t_gyr_q[0][1]<= t_acc_q[0][1]:
            data=t_gyr_q.popleft()
            sensorType = "android.sensor.gyroscope"
            timestamp = data[1]
            values = data[0]
        else:
            data=t_acc_q.popleft()
            sensorType = "android.sensor.accelerometer"
            timestamp = data[1]
            values = data[0]

        # if accelerometer data arrives in start of walk for calibration
        if sensorType == "android.sensor.accelerometer" and calib_with_g > 0:

            x, y, z = values
            acc_vec = np.array([[y], [-z], [-x]])
            acc_vec_g = np.array([[-y], [z], [x]])
            if strong_calib_with_g > 0:
                X_n[0, 0] = roll_from_g_321(acc_vec_g[0, 0], acc_vec_g[1, 0], acc_vec_g[2, 0])
                X_n[1, 0] = pitch_from_g_321(acc_vec_g[0, 0], acc_vec_g[1, 0], acc_vec_g[2, 0])
                strong_calib_with_g = strong_calib_with_g - 1
            X_n, P_n = kalman(acc_vec_g, X_n, P_n, R_not_moving, dt=const_dt)
            calib_with_g = calib_with_g - 1
            last_acc = timestamp

        # if accelerometer data arrives after calibration apply the logic for identifying start and end of step from book project
        # and apply roll pitch corraction if foot is at rest
        if sensorType == "android.sensor.accelerometer" and calib_with_g <= 0:
            x, y, z = values
            acc_vec = np.array([[y], [-z], [-x]])
            acc_vec_g = np.array([[-y], [z], [x]])

            if end_of_step == True:
                kalman(acc_vec_g, X_n, P_n, R_not_moving, dt=const_dt)

                if flag_yaw_at_end_of_step == False:
                    yaw_at_end_of_step = X_n[2, 0]
                    flag_yaw_at_end_of_step = True

            acc_earth_vec = acc_earth(X_n, acc_vec)
            log_abs_a.write(f"{(((acc_earth_vec[0, 0]) ** 2 + (acc_earth_vec[1, 0]) ** 2 + (acc_earth_vec[2, 0]) ** 2) ** 0.5)},{(((x) ** 2 + (y) ** 2 + (z) ** 2) ** 0.5)},{time.time() - start_time},{int(end_of_step)}\n")

            moving_Q = (((acc_earth_vec[0, 0]) ** 2 + (acc_earth_vec[1, 0]) ** 2 + (acc_earth_vec[2, 0]) ** 2) ** 0.5) > 1.5

            acc_delay_que.append([acc_earth_vec, moving_Q, (timestamp - last_acc) * 1e-9])
            last_acc = timestamp
            if moving_Q:
                curr_and_after_num_non_move = max(curr_and_after_num_non_move - 1, 0)
            else:
                curr_and_after_num_non_move = min(curr_and_after_num_non_move + 1, non_move_after + 1)


            if len(acc_delay_que) == non_move_after + 1:
                delay_earth_vec = acc_delay_que.popleft()
                moving_Q_delay_vec = delay_earth_vec[1]
                if before_num_non_move + curr_and_after_num_non_move == non_move_before + 1 + non_move_after:
                    if end_of_step == False:
                        end_of_step = True
                        acc_for_calc.append(delay_earth_vec[0])
                        acc_dt.append(delay_earth_vec[2])

                        step_calc(acc_for_calc, acc_dt, yaw_at_end_of_step,siting_state)

                        acc_for_calc = deque()
                        acc_dt = deque()

                else:
                    if end_of_step == True:
                        end_of_step = False
                        flag_yaw_at_end_of_step = False
                    acc_for_calc.append(delay_earth_vec[0])
                    acc_dt.append(delay_earth_vec[2])
                if moving_Q_delay_vec:
                    before_num_non_move = max(before_num_non_move - 1, 0)
                else:
                    before_num_non_move = min(before_num_non_move + 1, non_move_after)

        # if gyroscope data arrives update orantaion from gyroscope
        if sensorType == "android.sensor.gyroscope" and calib_ang <= 0:
            x, y, z = values
            gyr_vec = np.array([[y], [-z], [-x]]) #-((1/1200)*offset)
            X_n, P_n = orantaion_update(gyr_vec, X_n, P_n, (timestamp - last_gyr) * 1e-9)
            last_gyr = timestamp
            #print(X_n.flatten(),Pos_n.flatten(),time.time()-start_time)
            log_orination.write(f"{X_n[0, 0]},{X_n[1, 0]},{X_n[2, 0]},{timestamp*1e-9},{int(end_of_step)}\n")

        if sensorType == "android.sensor.gyroscope" and calib_ang > 0:
            x, y, z = values
            gyr_vec = np.array([[y], [-z], [-x]])
            last_gyr = timestamp
            offset = offset + gyr_vec
            calib_ang = calib_ang - 1

        # this part In charge for communicating with camera yaw correction program
        # listening to keyboard for c,z,x,n keys
        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8").lower()
            # for 'c' we start recording new set of pics
            if key == 'c':
                print("c")
                orant_save = X_n.copy()
                pos_sav = Pos_n.copy()
                hom_send_socket.sendto(b'c', ('localhost', 6044))
            # for 'z' we save current camera view
            if key == 'z':
                payload = json.dumps({
                    "r": float(X_n[0,0]),
                    "p": float(X_n[1,0]),
                    "y": float(X_n[2,0])
                }).encode('utf-8')
                hom_send_socket.sendto(payload, ('localhost', 6983))
                hom_send_socket.sendto(b'z', ('localhost', 6044))
            # for 'x' we calcutate photos camera orientations
            if key == 'x':
                hom_send_socket.sendto(b'x', ('localhost', 6044))
            # for 'n' we clac/correct our current yaw by taking new pic and finding nearest photo in same direction and clac rotatin from that photo
            if key == 'n':
                tmp_orant_save =X_n
                payload = json.dumps({
                    "r": float(X_n[0,0]),
                    "p": float(X_n[1,0]),
                    "y": float(X_n[2,0])
                }).encode('utf-8')
                hom_send_socket.sendto(payload, ('localhost', 6983))
                hom_send_socket.sendto(b'n', ('localhost', 6044))
            if key == 'w':
                hom_send_socket.sendto(b'w', ('localhost', 6044))
            if key == 'q':
                end_ondata_loop=True
                end_plot_loop=True
                print("did it")
        try:
            ready, _, _ = select.select([hom_server], [], [], 0)
            for s in ready:
                option_vec = json.loads(s.recv(4096).decode('utf-8'))
                option_vec = np.array(list(option_vec.values()))
                diff_vec=(X_n - tmp_orant_save).flatten()
                est_vec=option_vec+diff_vec
                est_vec[2]=norm_pi_2_neg_pi(est_vec[2])
                print("x_n", X_n)
                print("--------option_vec------------")
                print(option_vec)
                print("--------error_no_normaleizion------------")
                print(est_vec[2]-X_n[2,0])
                print("--------error_with_normaleizion------------")
                print(norm_pi_2_neg_pi((est_vec[2]-X_n[2,0])))
                print("time of fix",time.time() - start_time)
                print("--------end------------")
                X_n[2,0]=X_n[2]+(1*norm_pi_2_neg_pi((est_vec[2]-X_n[2,0])))
                X_n[2,0]=norm_pi_2_neg_pi(X_n[2,0])
        except Exception as e:
            print(e)
            traceback.print_exc()






if __name__ == '__main__':
    '''
    to connect phone imu enter address that shown in "sensor server" app after pressing start  
    '''
    address = "localhost:8080"


    folder_path = "logs"+formatted_datetime
    os.makedirs(folder_path, exist_ok=True)
    start_pos_time = time.time()
    log_pos = open(folder_path + "/" + "log_pos.txt", "w")
    log_pos.write(f"{0.0},{0.0},{0.0},{'standing_still'},{0}\n")
    log_pos.flush()
    log_orination = open(folder_path + "/" + "log_orination.txt", "w")
    log_abs_a = open(folder_path + "/" + "log_abs_a.txt", "w")


    #connect to real time ploter program
    address2 = ('localhost', 6000)
    conn = Client(address2, authkey=b'secret password')

    acc_lsin=Sensor(address=address, sensor_type="android.sensor.accelerometer", on_sensor_event=on_accelerometer_event)
    gyr_lsin=Sensor(address=address, sensor_type="android.sensor.gyroscope", on_sensor_event=on_gyroscope_event)
    gyr_lsin_proc=threading.Thread(target=gyr_lsin.make_websocket_connection)
    acc_lsin_proc=threading.Thread(target=acc_lsin.make_websocket_connection)
    t = threading.Thread(target=ondata)


    gyr_lsin_proc.start()
    acc_lsin_proc.start()
    t.start()
    t.join()
    acc_lsin.disconnect()
    gyr_lsin.disconnect()
    gyr_lsin_proc.join()
    acc_lsin_proc.join()

    conn.close()


