from roboflow import Roboflow
rf = Roboflow(api_key="WnIEFjHcUwDt2iJb66KQ")
project = rf.workspace("10014rebellion").project("rebuilt")
version = project.version(1)
dataset = version.download("yolov7")

