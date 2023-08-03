#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "FRITZ!Box 7530 JR";
const char* password = "41828145138821269583";

// Add MQTT server settings
const char* mqttServer = "192.168.178.18";
const int mqttPort = 1883;
const char* mqttUser = "zigbee";
const char* mqttPassword = "zigbee";

WiFiClient espClient;
PubSubClient client(espClient);

// Assign output variables to GPIO pins
const int output26 = 26;

void setup() {
  Serial.begin(115200);

  pinMode(output26, OUTPUT);
  digitalWrite(output26, LOW);

  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  client.setServer(mqttServer, mqttPort);
  client.setCallback(callback);

  while (!client.connected()) {
    Serial.println("Connecting to MQTT...");

    if (client.connect("ESP32Client", mqttUser, mqttPassword )) {
      Serial.println("connected");  
    } else {
      Serial.print("failed with state ");
      Serial.print(client.state());
      delay(2000);
    }
  }

  client.publish("esp32/debug", "ESP32 online");
  client.subscribe("cat_washer/hose");
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived in topic: ");
  Serial.println(topic);

  Serial.print("Message:");
  for (unsigned int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }

  Serial.println();
  Serial.println("-----------------------");

  String msgString;
  for(unsigned int i=0; i < length; i++) {
    msgString += (char)payload[i];
  }

  if (msgString == "ON") {
    digitalWrite(output26, HIGH);
    Serial.println("GPIO 26 on");
  } else if (msgString == "OFF") {
    digitalWrite(output26, LOW);
    Serial.println("GPIO 26 off");
  }
}

void loop() {
  client.loop();
}
