import re
template = "https://api.example.com/users/{user_id}/posts/{post_id}"
params = {"user_id": 123, "post_id": 456, "extra": "data"}

def replace(match):
    key = match.group(1)
    if key in params:
        return str(params[key])
    return match.group(0)

print(re.sub(r'\{([a-zA-Z0-9_]+)\}', replace, template))
