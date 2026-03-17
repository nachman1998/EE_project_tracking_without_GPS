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
from transformers import AutoImageProcessor, LightGlueForKeypointMatching, SegformerImageProcessorFast, SegformerForSemanticSegmentation
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

start_time=time.time()

'''
!!!!!!!!!!! AFTER CALIBRATION ENTER THE RESULT K MATRIX HERE  !!!!!!!!!!!!!!!!!!!!!!!
'''
K= np.array([[-511.53544277, 0., 354.38189319], [0, -515.91600933, 253.04608242], [0., 0., 1.]])
K_inv=np.linalg.inv(K)
K_cam= np.array([[-511.53544277, 0., 354.38189319], [0, -515.91600933, 253.04608242], [0., 0., 1.]])
K_cam_inv=np.linalg.inv(K_cam)

def Rot(roll,pitch,yaw):

    s_r,s_p,s_y=math.sin(roll),math.sin(pitch),math.sin(yaw)
    c_r,c_p,c_y=math.cos(roll),math.cos(pitch),math.cos(yaw)
    return np.array([[c_y*c_p,s_y*c_p,-s_p],
                     [(c_y*s_p*s_r)-(c_r*s_y),(s_y*s_p*s_r)+(c_r*c_y),c_p*s_r],
                     [(c_r*s_p*c_y)+(s_r*s_y),(s_y*s_p*c_r)-(s_r*c_y),c_p*c_r]])

def ang_from_matrix(mat):
    Rtag = mat
    yaw = math.atan2(Rtag[0, 1], Rtag[0, 0])
    pitch = math.asin(-Rtag[0, 2])
    roll = math.atan2(Rtag[1, 2], Rtag[2, 2])
    return [roll, pitch,yaw]

def norm_pi_2_neg_pi(ang):
    if ang>math.pi:
        return ang-(2*math.pi)
    if ang<=-math.pi:
        return ang + (2 * math.pi)
    return ang

#find matching features between photos and calculate homograpy and retuen decomposition options
def calc_angle_diff_homograpy_no_distance(old_frame,new_frame,y_level,erosion=11,iter=4):

    global X_n
    global model_seg
    global model_lightglue
    global processor_seg
    global processor_lightglue
    global K_cam
    global K_cam_inv

    image1 =Image.fromarray(old_frame).convert("RGB")#Image.open('out11.jpeg') #
    image2 =Image.fromarray(new_frame).convert("RGB")#Image.open('out1.jpeg') #
    images = [image1, image2]

    inputs_lightglue = processor_lightglue(images=images, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs_lightglue = model_lightglue(**inputs_lightglue)

    image_sizes = [[(image.height, image.width) for image in images]]
    processed_outputs_lightglue = processor_lightglue.post_process_keypoint_matching(outputs_lightglue, image_sizes,
                                                                                     threshold=0.8)

    matches = processed_outputs_lightglue[0]
    kpts1 = matches["keypoints0"].cpu().numpy()
    kpts2 = matches["keypoints1"].cpu().numpy()
    scores = matches["matching_scores"].cpu().numpy()

    inputs = processor_seg(images=image1, return_tensors="pt").to(device)
    outputs_seg = model_seg(**inputs)
    # --- inference ---
    logits = outputs_seg.logits  # shape (batch_size, num_labels, height/4, width/4)

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image1.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )

    pred = upsampled_logits.argmax(dim=1)[0]  # shape [H, W]
    pred = pred.cpu().numpy()
    wanted_ids = [2]
    bin_mask = np.zeros((image1.height, image1.width), dtype=np.uint8)
    for id_ in wanted_ids:
        bin_mask[pred == id_] = 255
    kernel = np.ones((erosion, erosion), np.uint8)
    eroded = cv2.erode(bin_mask, kernel, iterations=iter)
    h, w = image1.height, image1.width
    mask_ground = np.zeros((h, w), dtype=np.uint8)
    mask_ground[int(h-(h*0.001)):-1, :] = 255
    # --- Filter Keypoints ---

    kpts1_int = np.round(kpts1).astype(int)
    kpts2_int = np.round(kpts1).astype(int)

    kpts1_x_clipped = np.clip(kpts1_int[:, 0], 0, w - 1)
    kpts1_y_clipped = np.clip(kpts1_int[:, 1], 0, h - 1)
    kpts2_x_clipped = np.clip(kpts2_int[:, 0], 0, w - 1)
    kpts2_y_clipped = np.clip(kpts2_int[:, 1], 0, h - 1)

    inside_seg_mask1 = eroded[kpts1_y_clipped, kpts1_x_clipped] != 0
    inside_ground_mask2 = mask_ground[kpts2_y_clipped, kpts2_x_clipped] != 0
    invalid_pair_mask = inside_seg_mask1 | inside_ground_mask2
    valid_pair_mask = ~invalid_pair_mask

    filtered_kpts1 = kpts1[valid_pair_mask]
    filtered_kpts2 = kpts2[valid_pair_mask]

    print(f"Original keypoints: {len(kpts1)}, Keypoints after filtering: {len(filtered_kpts1)}")


    if len(filtered_kpts1) >= 4 and len(filtered_kpts2) >= 4:
        H, mask = cv2.findHomography(filtered_kpts1, filtered_kpts2, cv2.RANSAC, 5.0)

        if H is None:
            print("Homography computation failed - not enough inliers")
            return None  # or return 0, or handle appropriately
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K_cam)
        # Now safe to use H
        R = K_cam_inv @ H @ K_cam
    else:
        print(f"Not enough keypoints for homography: {len(filtered_kpts1)} points")
        return None




    vec_option={}
    R90=Rot((-const.pi)/2,0,0)
    Rtag=(R90.T)@R@R90
    yaw = math.atan2(Rtag[0, 1], Rtag[0, 0])
    #pitch = math.asin(-Rtag[0, 2])
    roll = math.atan2(Rtag[1, 2], Rtag[2, 2])
    pitch2 = math.atan2(-Rtag[0, 2] * math.cos(roll), Rtag[2, 2])

    vec_option["vec1"]=([roll, pitch2,  yaw])

    ve=2
    for Rq in Rs:
        Rtag = (R90.T) @ Rq @ R90
        yaw = math.atan2(Rtag[0, 1], Rtag[0, 0])
        pitch = math.asin(-Rtag[0, 2])
        roll = math.atan2(Rtag[1, 2], Rtag[2, 2])
        vec_option[f"vec{ve}"]=([roll, pitch, yaw])
        ve=ve+1

    print("vec_option ---->",vec_option)
    return vec_option


def find_closest_vector(query_vec, vec_list):
    vec_list = np.atleast_2d(vec_list)
    print("vec_list ---->",vec_list)
    print("query_vec ---->", query_vec)
    distances = np.linalg.norm(vec_list - query_vec, axis=1)
    idx = np.argmin(distances)
    return vec_list[idx]


def delta_rotation(ang1,ang2):
    """All angles in radians. Returns R_delta and its (roll, pitch, yaw) in rad."""
    R1 = Rot(ang1[0], ang1[1], ang1[2])
    R2 = Rot(ang2[0], ang2[1], ang2[2])

    Rtag = R2 @ R1.T
    yaw = math.atan2(Rtag[0, 1], Rtag[0, 0])
    pitch = math.asin(-Rtag[0, 2])
    roll = math.atan2(Rtag[1, 2], Rtag[2, 2])

    return np.array([roll,pitch,yaw])

#main loop that run in background receives commands from main progrms with keys c,x,z,n
def h264_stream():
    global url
    global server
    global X_n
    global P_n
    global Pos_n
    global end_ondata_loop
    global frame
    global endlp

    hom_server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    host='localhost'
    port=6044
    hom_server.setblocking(0)
    hom_server.bind((host,port))

    pic=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    host='localhost'
    port=6983
    pic.setblocking(0)
    pic.bind((host,port))


    hom_send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cam_mat = np.array([[-511.53544277, 0., 354.38189319], [0, -515.91600933, 253.04608242], [0., 0., 1.]])
    dis_coff = np.array([[-0.05361873, 0.00417398, 0.00351033, -0.00343848, 0.02826234]])
    #im_src = cv2.undistort(im_src, cam_mat, dis_coff, None, None)


    flag_first_Frame = True
    X = np.empty((0, 3))


    timer=time.time()

    calib_frames=[]
    tot_frame_ang_post=[]
    tot_frame_ang_neg = []
    tot_frame_ang=[]
    calibrate=True
    loop=True
    a=0.5




    while loop:


        if msvcrt.kbhit():
            key2 = msvcrt.getch().decode("utf-8").lower()
            if key2 == 'w':
                loop = False
                end_plot_loop = True
                endlp=True
                print("did it")
                break

        try:

            ready, _, _ = select.select([hom_server], [], [], 0)
            for s in ready:
                key=s.recv(600).decode()
                if key == 'c':
                    print("c")
                    calib_frames = []
                    tot_frame_ang_post = []
                    tot_frame_ang_neg = []
                    tot_frame_ang = []
                if key=='z':
                    try:
                        ready, _, _ = select.select([pic], [], [], 0)
                        for s in ready:
                            ang_vec = json.loads(s.recv(4096).decode('utf-8'))
                            ang_vec = list(ang_vec.values())
                            calib_frames.append([frame,ang_vec])
                            print("saved frame at" ,ang_vec)
                    except Exception as e:
                        print(e)
                if key=='x':


                    tot_frame_ang = calib_frames
                    print("end")
                    '''for i in range(len(tot_frame_ang)):
                        cv2.imshow(f"Frame {i} with {tot_frame_ang[i][1][0]},{tot_frame_ang[i][1][1]},{tot_frame_ang[i][1][2]}", tot_frame_ang[i][0])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''
                if key == 'n':
                    print("in n")

                    try:
                        new_frame = frame
                        ready, _, _ = select.select([pic], [], [], 0)
                        for s in ready:
                            ang_vec = json.loads(s.recv(4096).decode('utf-8'))
                            ang_vec = list(ang_vec.values())
                            closest = min(tot_frame_ang, key=lambda x: abs(norm_pi_2_neg_pi(x[1][2]-ang_vec[2])))


                            option_vec = calc_angle_diff_homograpy_no_distance(closest[0], new_frame, y_level=280)
                            if option_vec == None:
                                print("problm at X part no 4 points")
                                break
                            option_vec = list(option_vec.values())
                            option_vec = np.array(option_vec)
                            pic_vec = find_closest_vector((delta_rotation(closest[1],ang_vec )).flatten(), option_vec)
                            pic_vec_mat = Rot(pic_vec[0], pic_vec[1], pic_vec[2])
                            closest_mat = Rot(closest[1][0], closest[1][1], closest[1][2])
                            post = ang_from_matrix(pic_vec_mat@closest_mat)
                            post[2]=norm_pi_2_neg_pi(post[2])
                            mtchframe=[closest,[new_frame,post],[new_frame,ang_vec]]
                            '''for i in range(len(mtchframe)):
                                cv2.imshow(
                                    f"Frame {i} with {mtchframe[i][1][0]},{mtchframe[i][1][1]},{mtchframe[i][1][2]}",mtchframe[i][0])
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()'''
                            print(f"Frame choose colssest with {closest[1][0]},{closest[1][1]},{closest[1][2]}")
                            print("post ---->",post)
                            payload = json.dumps({
                                "r": float(post[0]),
                                "p": float(post[1]),
                                "y": float(post[2])
                            }).encode('utf-8')
                            hom_send_socket.sendto(payload, ('localhost', 6023))
                    except Exception as e:
                        print(e)
                        traceback.print_exc()


        except Exception as e:
            print(e)
            traceback.print_exc()





if __name__ == '__main__':

    '''
    to connect phone camera enter address that shown in "ipwabcam" app after pressing start  
    '''
    address="localhost:5000"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor_seg = SegformerImageProcessorFast.from_pretrained(r"segformer")
    model_seg = SegformerForSemanticSegmentation.from_pretrained(r"segformer").to(device)
    model_seg.eval()

    processor_lightglue = AutoImageProcessor.from_pretrained(r"lightglue")
    model_lightglue = LightGlueForKeypointMatching.from_pretrained(r"lightglue").to(device)
    model_lightglue.eval()
    vido_t=threading.Thread(target=h264_stream)
    vido_t.start()

    cap = cv2.VideoCapture(f"http://{address}/video")
    frame=None
    endlp=False
    while endlp==False:
        try:

            ret, frame = cap.read()
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            traceback.print_exc()
    cap.release()
    cv2.destroyAllWindows()

    vido_t.join()
