"""Load a .dotcfg's saved jobs without touching the GUI or the program source."""
import io, json, zipfile
import numpy as np, cv2
from dotgen.core.models import Job, DotModel

def load(path):
    zf = zipfile.ZipFile(path, "r")
    meta = json.loads(zf.read("config.json"))
    names = set(zf.namelist())

    model = None
    if "dot_model.npz" in names:
        npz = np.load(io.BytesIO(zf.read("dot_model.npz")), allow_pickle=False)
        model = DotModel(
            patch_radius=int(npz["patch_radius"][0]),
            mean=npz["mean"], components=npz["components"],
            score_std=npz["score_std"], n_samples=int(npz["n_samples"][0]),
        )

    bg_arrays = []
    for i in range(len(meta.get("backgrounds", []))):
        f = f"images/bg_{i}.png"
        if f in names:
            bg_arrays.append(cv2.imdecode(np.frombuffer(zf.read(f), np.uint8), cv2.IMREAD_COLOR))
    samples = [cv2.imdecode(np.frombuffer(zf.read(e["file"]), np.uint8), cv2.IMREAD_COLOR)
               for e in meta.get("sample_images", []) if e["file"] in names]

    jobs = [Job.from_dict(j) for j in meta.get("jobs", [])]
    for j in jobs:
        j.dot_model = model
        for k, bg in enumerate(j.backgrounds):
            if k < len(bg_arrays):
                bg.array = bg_arrays[k]
    return meta, jobs, model, bg_arrays, samples
