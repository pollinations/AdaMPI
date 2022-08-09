import json


with open("inspect.json", "r") as f:
    data = json.load(f)[0]

with open("openapi.json", "w") as f:
   f.write(data["ContainerConfig"]["Labels"]["org.cogmodel.openapi_schema"])