import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {str(rc)}")

def on_publish(client, userdata, result):
    print(f"Data published with result code {result}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_publish = on_publish

client.username_pw_set("zigbee", "zigbee")

client.connect("192.168.178.18", 1883, 60)

while True:
    client.publish("cat_washer/hose", payload="ON", qos=0, retain=False)
    print('tried to send message')
    time.sleep(2)
    client.publish("cat_washer/hose", payload="OFF", qos=0, retain=False)
    print('tried to send message')
    time.sleep(2)