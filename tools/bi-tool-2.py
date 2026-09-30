
#DuckDuckGoSearch tool
'''
from langchain_community.tools import DuckDuckGoSearchRun

search_tools = DuckDuckGoSearchRun()


result = search_tools.invoke('Dehradun news')

print(result)
'''


# Shell tool

from langchain_community.tools import ShellTool

shell = ShellTool()

result = shell.invoke('whoami')

print(result)
