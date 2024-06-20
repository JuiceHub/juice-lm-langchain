import os

os.environ["BING_SUBSCRIPTION_KEY"] = "<key>"
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

from langchain.utilities import BingSearchAPIWrapper

search = BingSearchAPIWrapper()

search.run("python")