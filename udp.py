import json
import os
import socket
import threading
import wave

import paho.mqtt.client as mqtt
import requests

# --- CONFIGURATION ---
MQTT_BROKER = "mff41cf7.ala.asia-southeast1.emqxsl.com"
MQTT_PORT = 8883
MQTT_USER = "toy"
MQTT_PASS = "Lalit@123"
TOPIC_PUB_RESPONSE = "pc/response"

UDP_PORT = 5005
WEB_SERVER_PORT = 8080  # Changed from 8000 to avoid Permission Error
SAMPLE_RATE = 16000

# VM Configuration
EC2_IP = "13.235.246.113"
VM_BASE_URL = f"http://{EC2_IP}:8000"

audio_chunks = []


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "127.0.0.1"
    finally:
        s.close()
    return ip_address


MY_IP = EC2_IP


# --- AUDIO PROCESSING ---
def save_audio_and_replay(buffer_data, mqtt_client):
    filename = "recording.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(buffer_data)

    print(f"\nAudio saved: {filename} ({len(buffer_data)} bytes)")

    with open(filename, "rb") as f:
        files = {"file": f}
        params = {"lang": "hi"}
        stt_res = requests.post(f"{VM_BASE_URL}/stt", files=files, params=params)

    if stt_res.status_code != 200:
        print(f"STT failed with status {stt_res.status_code}")
        return None

    print("STT successful.")
    print(f"Full STT Response: {stt_res.json()}")

    print("Send to LLM")
    ask_url = f"{VM_BASE_URL}/ask"
    text = stt_res.text

    print("Request the LLM")
    params = {"question": text, "lang": "en"}

    print("Sending question for ASK...")
    response = requests.post(ask_url, params=params)

    if response.status_code == 200:
        print("ASK success")
        print("Response:", response.json())
    else:
        print("ASK failed")
        print(response.text)
        return None

    print("Requesting TTS for LLM response...")
    ask_res = response.json()
    print(ask_res)

    tts_url = f"{VM_BASE_URL}/tts"
    text = ask_res.get("answer", "")

    print(f"Requesting TTS for text: {text}")
    params = {"text": text, "lang": "hi"}

    print("Sending text for TTS...")
    response = requests.post(tts_url, params=params)

    if response.status_code != 200:
        print("TTS failed")
        print(response.text)
        return None

    print("TTS success")
    tts_res = response.json()
    response_audio_url = tts_res.get("audio_url")

    if not response_audio_url:
        print("TTS response missing audio_url")
        print(tts_res)
        return None

    play_command = {
        "identifier": "audioplay",
        "inputParams": {"url": response_audio_url},
    }
    print(f"MQTT -> ESP32: Play this file: {response_audio_url}")
    mqtt_client.publish(TOPIC_PUB_RESPONSE, json.dumps(play_command))
    return response_audio_url


# --- UDP RECEIVER ---
def start_udp_receiver(mqtt_client):
    global audio_chunks
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"[UDP] Listening for audio on {MY_IP}:{UDP_PORT}...")

    while True:
        data, addr = sock.recvfrom(2048)
        try:
            token_len = int.from_bytes(data[0:4], byteorder="big")
            payload = data[4 + token_len :]
            if b"STOP" in payload:
                print(f"\nSTOP signal from {addr}")
                if audio_chunks:
                    save_audio_and_replay(b"".join(audio_chunks), mqtt_client)
                    audio_chunks = []
            else:
                audio_chunks.append(payload)
                print(f"Recording chunks... {len(audio_chunks)}", end="\r")
        except Exception:
            pass


# --- MQTT SETUP ---
# Updated to Callback API version 2 to remove DeprecationWarning
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.tls_set()


def on_connect(client, userdata, flags, rc, properties=None):
    print("MQTT connected.")
    client.subscribe([("esp32/data", 0), ("user/cheekotoy/e4b063b90be8/thing/data/post", 0)])


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        if data.get("identifier") == "login":
            client.publish(
                TOPIC_PUB_RESPONSE,
                json.dumps({"identifier": "updatetoken", "inputParams": {"token": "SECRET"}}),
            )
        elif data.get("identifier") == "data_config":
            client.publish(
                TOPIC_PUB_RESPONSE,
                json.dumps(
                    {
                        "identifier": "updateconfig",
                        "inputParams": {
                            "udp_host": MY_IP,
                            "udp_port": UDP_PORT,
                            "http_host": MY_IP,
                            "http_port": WEB_SERVER_PORT,
                            "sample_rate": 16000,
                            "channels": 1,
                        },
                    }
                ),
            )
    except Exception:
        pass


mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)

# --- START ALL SERVICES ---
threading.Thread(target=mqtt_client.loop_forever, daemon=True).start()
start_udp_receiver(mqtt_client)