from langchain import SerpAPIWrapper

params = {
    "hl": "zh-cn",
    "gl": "cn"
}
serp_api = SerpAPIWrapper(serpapi_api_key='b1c6f89c612b9fba33e06b87ab00d475b2535f235950234d706ef7ffc4ffba98',
                          params=params)
