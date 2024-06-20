import json

import requests

if __name__ == '__main__':
    data = {
        'history': [
            {"role": "system", "content": "您好，有什么可以帮您的"},
            {"role": "user", "content": "没有"}
        ],
        "prompt": "农历和阳历的区别"
    }
    res = requests.post('http://192.168.1.119:5000/model/chat', data=json.dumps(data))
    print(res.content)
