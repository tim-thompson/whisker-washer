import datetime
import logging
import os
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.core import Detections
from supervision.draw.color import Color
from ultralytics import YOLO

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_log_file_path():
    now = datetime.datetime.now()
    return os.path.join(LOG_DIR, f'log_{now.strftime("%Y%m%d")}.log')


log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(get_log_file_path())
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected to MQTT Broker!")
    else:
        logging.warning(f"Failed to connect, return code {rc}")


def on_disconnect(client, userdata, rc):
    logging.info("Disconnected from MQTT Broker")


def on_publish(client, userdata, mid):
    msg_info = client.user_data_set
    logging.info(
        f"MQTT message sent. Topic: {msg_info['topic']}, Payload: {msg_info['payload']}, ID: {mid}"
    )


client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_publish = on_publish
client.username_pw_set("zigbee", "zigbee")
client.connect("192.168.178.18", 1883, 60)
client.loop_start()


def publish(topic, payload):
    client.user_data_set = {"topic": topic, "payload": payload}
    client.publish(topic, payload=payload, qos=1, retain=False)


class ObjectDetection:
    def __init__(self, video_stream):
        self.video_stream = video_stream
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using Device: {self.device}")
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(
            color=Color.red(), thickness=1, text_thickness=1, text_scale=0.5
        )

    def load_model(self):
        model = YOLO("yolov8x.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model.predict(frame, conf=0.4, classes=[15, 16], verbose=False)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results[0]:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        self.labels = [
            f"{class_id}: {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for xyxy, mask, confidence, class_id, tracker_id in detections
        ]

        frame = self.box_annotator.annotate(
            scene=frame, detections=detections, labels=self.labels
        )

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.video_stream)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        video_writer = None
        start_recording_time = None
        last_detection_time = None

        while True:
            if not client.is_connected():
                logging.warning("Attempting to reconnect to MQTT Broker...")
                try:
                    client.reconnect()
                except:
                    logging.error("Reconnection failed, trying again in 5 seconds")
                    time.sleep(5)
                    continue

            if not cap.isOpened():
                logging.warning(
                    "Video stream connection lost, attempting to reconnect..."
                )
                cap.release()
                cap = cv2.VideoCapture(self.video_stream)

            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from video stream")
                continue

            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

            results = self.predict(frame)

            frame = self.plot_bboxes(results, frame)

            if any([len(res.boxes.xyxy) > 0 for res in results]):
                logging.info("Detection made, sending MQTT message")
                publish("cat_washer/hose", "ON")
                last_detection_time = time.time()
                if video_writer is None:
                    logging.info("Starting video recording")
                    start_recording_time = time.time()
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    now = datetime.datetime.now()
                    filename = f'output_{now.strftime("%Y%m%d%H%M%S")}.mp4'
                    video_writer = cv2.VideoWriter(filename, fourcc, 20, (1280, 720))

            if video_writer is not None and (time.time() - last_detection_time) >= 10:
                logging.info("Detection ended, stopping video recording")
                publish("cat_washer/hose", "OFF")
                video_writer.release()
                video_writer = None

            if video_writer is not None:
                video_writer.write(frame)

            frame = self.plot_bboxes(results, frame)
            cv2.imshow("CAT Detector", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        if video_writer is not None:
            logging.info("Stopping video recording")
            video_writer.release()

        cap.release()
        cv2.destroyAllWindows()
        client.loop_stop()


detector = ObjectDetection("rtsp://192.168.178.72/11")
detector()