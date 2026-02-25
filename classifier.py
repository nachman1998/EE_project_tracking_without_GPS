import torch
import requests
import numpy as np
import cv2
import time
import socket
from PIL import Image
from io import BytesIO


from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

device ="cpu"

curr_time=time.time()

# ------------------------------------------------------------------------
# Stage 1. Detect humans on the image
# ------------------------------------------------------------------------

#huggingface-cli download PekingU/rtdetr_r50vd_coco_o365 --local-dir rtdetr_local
#huggingface-cli download usyd-community/vitpose-plus-base --local-dir vitpose_local

# You can choose detector by your choice
person_image_processor = AutoProcessor.from_pretrained("rtdetr_local")
person_model = RTDetrForObjectDetection.from_pretrained("rtdetr_local",device_map=device,local_files_only=True)
image_processor = AutoProcessor.from_pretrained("vitpose_local")
model = VitPoseForPoseEstimation.from_pretrained("vitpose_local",device_map=device,local_files_only=True)
sit_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
Adrr=('localhost',6009)
#Adrr=('10.232.156.44',6009)
#url = "http://10.0.0.4:5000/video"
image_url="http://10.21.229.81/capture"
#image_url="http://10.21.229.81:5000/photo.jpg"
'''cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot connect to IP webcam stream")
    exit()'''
session = requests.Session()
while True:

    #ret, frame = cap.read()
    #if not ret:
     #   print("Failed to grab frame")
      #  break
    try:
        # timeout=(2, 5) means: 2 seconds to connect, 5 seconds to download the image
        response = session.get(image_url, timeout=(2, 5))
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        frame = np.array(image)

        # Show the frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('IP Webcam Stream', frame_bgr)

    except requests.exceptions.Timeout:
        print("Camera took too long to respond. Skipping frame...")
        continue  # Skips the rest of the loop and tries again
    except requests.exceptions.ConnectionError:
        print("Connection dropped by phone. Retrying in 1 second...")
        time.sleep(1)
        continue
    except Exception as e:
        print(f"Network error: {e}")
        continue

    if cv2.waitKey(1) == 27:  # Press Esc to quit
        break

    if (time.time()-curr_time)>5:
        curr_time=time.time()
        inputs = person_image_processor(images=image, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                outputs = person_model(**inputs)

            results = person_image_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
            )
            result = results[0]  # take first image results
            print(result)
            # Human label refers 0 index in COCO dataset
            person_boxes = result["boxes"][result["labels"] == 0]
            person_boxes = person_boxes.cpu().numpy()

            # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
            person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
            person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

            # ------------------------------------------------------------------------
            # Stage 2. Detect keypoints for each person found
            # ------------------------------------------------------------------------


            inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

            # This is MOE architecture, we should specify dataset indexes for each image in range 0..5
            inputs["dataset_index"] = torch.tensor([0], device=device)

            with torch.no_grad():
                outputs = model(**inputs)

            pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes], threshold=0.3)
            image_pose_result = pose_results[0]  # results for first image
            r_hip=None
            r_knee=None
            l_hip = None
            l_knee = None
            for i, person_pose in enumerate(image_pose_result):
                if i>0:
                    break
                print(f"Person #{i}")
                for keypoint, label, score in zip(
                    person_pose["keypoints"], person_pose["labels"], person_pose["scores"]
                ):
                    keypoint_name = model.config.id2label[label.item()]
                    x, y = keypoint
                    print(f" - {keypoint_name}: x={x.item():.2f}, y={y.item():.2f}, score={score.item():.2f}")

                    if keypoint_name=='R_Hip' :
                        r_hip=np.array([x.item(),y.item()])
                    if keypoint_name=='R_Knee':
                        r_knee = np.array([x.item(), y.item()])
                    if keypoint_name=='L_Hip' :
                        l_hip=np.array([x.item(),y.item()])
                    if keypoint_name=='L_Knee':
                        l_knee = np.array([x.item(), y.item()])
            print(f"r_hip pos is: {r_hip}")
            print(f"r_knee pos is: {r_knee}")

            print(f"l_hip pos is: {l_hip}")
            print(f"l_knee pos is: {l_knee}")

            r_check = False
            l_check = False
            coount=0
            if type(r_hip)!= type(None) and type(r_knee)!= type(None):
                coount+=1
                r_check = np.linalg.norm(r_hip - r_knee) > 150
                print(f"r_dis is:{np.linalg.norm(r_hip - r_knee)}")
            if type(l_hip)!= type(None) and type(l_knee)!= type(None):
                coount += 1
                l_check = np.linalg.norm(l_hip - l_knee) > 150
                print(f"l_dis is:{np.linalg.norm(l_hip - l_knee)}")
            if coount==0:
                sit_socket.sendto(b"no_siting",Adrr)
                continue
            if (r_check or l_check):
                sit_socket.sendto(b"siting", Adrr)
                print("sitting mode")
                continue
            sit_socket.sendto(b"no_siting", Adrr)
        except Exception as e:
            print("none person")
            continue

cv2.destroyAllWindows()