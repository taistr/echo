import json

path = "/home/tyson/echo/resources/example_dataset/dataset.json"

with open(path, "r") as f:
    data = json.load(f)

for record in data["data"]:
    record["id"] = int(record["id"]) >> 64

with open(path, "w") as f:
    json.dump(data, f, indent=4)