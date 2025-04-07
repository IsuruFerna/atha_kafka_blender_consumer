import os
from dotenv import load_dotenv
from confluent_kafka import Producer
import json
import time

# load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Read the environment variables
KAFKA_BROKER_URL = os.getenv('KAFKA_BROKER_URL', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'streaming')



conf = {
    'bootstrap.servers': KAFKA_BROKER_URL,
    'socket.timeout.ms': 10000, # Increase connection timeout
    'message.timeout.ms': 5000, # Max time to retry delivery
    'retries': 3, # Number of retries on failure
}
producer = Producer(**conf)

def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}]")

def produce_message(finger_output):
    try:
        producer.produce(
            topic=KAFKA_TOPIC, 
            key='key', # Key for partitioning
            value=json.dumps(finger_output), 
            callback=delivery_report
        )
    except Exception as e:
        print(f"Producer error: {e}")

if __name__ == "__main__":
    print("this is host: ", KAFKA_BROKER_URL)
    print("path: ", dotenv_path)

    FINGER_ROTATE = 0
    for i in range(10):
        FINGER_ROTATE += 10
        finger_output = {"finger_rotations": [FINGER_ROTATE, FINGER_ROTATE, FINGER_ROTATE, FINGER_ROTATE, FINGER_ROTATE]} 
        produce_message(finger_output)
        time.sleep(0.03)  # Simulate real-time intervals
        producer.flush()