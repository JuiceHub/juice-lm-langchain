import logging

from fastapi import APIRouter
from langchain.memory import RedisChatMessageHistory
from langchain.schema import _message_to_dict

from memory.database.my_redis import my_redis as rd

session = APIRouter()
logger = logging.getLogger(__name__)

# 消息记录
message_history: RedisChatMessageHistory


def get_message_history():
    return message_history


@session.post('/create')
def create(data: dict):
    title = data.get('title')
    session_id = data.get('session_id')
    chain_type = data.get('chain_type')
    rd.hset('session:' + chain_type, session_id, title)


@session.delete('/{chain_type}/{session_id}')
def delete(chain_type: str, session_id: str):
    rd.hdel('session:' + chain_type, session_id)
    rd.delete('messages:' + session_id)


@session.get('/read_all/{chain_type}')
def read_all(chain_type: str):
    res = rd.hgetall('session:' + chain_type)
    sessions = []
    for uuid, title in res.items():
        sessions.append({
            'title': title,
            'sessionID': uuid
        })
    return sessions


@session.post('/read_messages')
def read_messages(data: dict):
    global message_history
    chain_type = data.get('chain_type')
    session_id = data.get('session_id')
    message_history = RedisChatMessageHistory(
        url="redis://192.168.1.119:6380/0", session_id=session_id, key_prefix=chain_type + ":"
    )

    messages = [_message_to_dict(m) for m in message_history.messages]
    return messages
