with open("app/services/rest_api.py", "r") as f:
    code = f.read()
import re
code = re.sub(r'return re.sub\(.*replace, template\)', "return re.sub(r'\\\\{([a-zA-Z0-9_]+)\\\\}', replace, template)", code)
with open("app/services/rest_api.py", "w") as f:
    f.write(code)
