import os, pickle
PATH = "/mnt/k/OpenSG_ScanNet"

scan = "scene0011_00"
split = "0"
pth = os.path.join(PATH, "preprocessed", scan, f"data_dict_{split}.pkl")
d = pickle.load(open(pth, "rb"))

obj2frame = d.get("object2frame", {})
obj_ids = d["objects_id"]
print("max obj_id:", max(obj_ids))
print("max obj2frame key:", max(map(int, obj2frame.keys())))
print("matched:", sum(1 for i in obj_ids if int(i) in obj2frame or str(int(i)) in obj2frame), "/", len(obj_ids))