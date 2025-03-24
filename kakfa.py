import sys
sys.path.append(r"C:\Users\isuru\Desktop\atha\blender_venv\Lib\site-packages")

import os
import bpy
import math
import time
import json
import threading
from queue import Queue
from confluent_kafka import Consumer, KafkaException, KafkaError
from dotenv import load_dotenv

# load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)


# Read the environment variables
KAFKA_BROKER_URL = os.getenv('KAFKA_BROKER_URL', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'streaming')

# Configuration
finger_data_queue = Queue()
running = True

# Function to apply the finger rotations directly
def apply_finger_rotations(data):
    try:
        pose_bones = bpy.data.objects["metarig"].pose.bones
        f_index_1_r = pose_bones["f_index.01.R"]
        f_middle_1_r = pose_bones["f_middle.01.R"]
        f_ring_1_r = pose_bones["f_ring.01.R"]
        f_pinky_1_r = pose_bones["f_pinky.01.R"]
        
        rotations = data["finger_rotations"]
        
        print("Applying rotations:", rotations)
        
        if len(rotations) >= 5:
            f_index_1_r.rotation_euler[0] = math.radians(rotations[1])
            f_middle_1_r.rotation_euler[0] = math.radians(rotations[2])
            f_ring_1_r.rotation_euler[0] = math.radians(rotations[3])
            f_pinky_1_r.rotation_euler[0] = math.radians(rotations[4])
        
        # Force update view
        bpy.context.view_layer.update()
    except Exception as e:
        print(f"Error applying finger rotations: {e}")

# Kafka consumer function
def consume_kafka_messages():
    global running
    conf = {
        "bootstrap.servers": KAFKA_BROKER_URL,
        'group.id': "mygroup",
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])

    try:
        while running:
            msg = consumer.poll(0.1)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Kafka error: {msg.error()}")
                    continue
                
            try:
                data = msg.value().decode("utf-8")
                print(f"Received message: {data}")
                data_dict = json.loads(data)
                finger_data_queue.put(data_dict)
            except Exception as e:
                print(f"Error processing message: {e}")
                
    except Exception as e:
        print(f"Kafka consumer error: {e}")
    finally:
        consumer.close()
        print("Kafka consumer stopped")

# Modal timer function
def modal_timer():
    if not finger_data_queue.empty():
        data = finger_data_queue.get()
        apply_finger_rotations(data)
    
    if running:
        return 0.01  # Call again in 0.01 seconds
    else:
        return None  # Stop calling

# Start function
def start_kafka_real_time():
    global running
    running = True
    
    # Start the Kafka consumer in a background thread
    consumer_thread = threading.Thread(target=consume_kafka_messages)
    consumer_thread.daemon = True
    consumer_thread.start()
    print("Kafka consumer thread started")
    
    # Register a timer to check for messages
    if not bpy.app.timers.is_registered(modal_timer):
        bpy.app.timers.register(modal_timer)
    
    print("Real-time finger control started")

# Stop function
def stop_kafka_real_time():
    global running
    running = False
    
    # Unregister the timer
    if bpy.app.timers.is_registered(modal_timer):
        bpy.app.timers.unregister(modal_timer)
    
    print("Real-time finger control stopped")

# Start the real-time control
start_kafka_real_time()

# To stop it later, you can run:
# stop_kafka_real_time()