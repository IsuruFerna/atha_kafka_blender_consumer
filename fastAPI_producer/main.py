from fastapi import FastAPI
from aiokafka import AIOKafkaProducer
from contextlib import asynccontextmanager

producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup code
    print("Starting up...")
    await producer.start()
    
    yield
    #shoutdown code
    print("Shutting down...")
    await producer.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/streaming/")
async def send_messages(topic: str, message: str):
    await producer.send_and_wait(topic, message.encode())
    return {"status": "message sent"}


