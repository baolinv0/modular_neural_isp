"""Strict domain-aware scene manifests and deterministic HDR/motion fixtures."""
import json
import hashlib
from pathlib import Path
import re
import numpy as np
import torch

from .types import SCENE_DOMAIN, Scene, SensorProfile


def _float_array(path: Path, key: str = "frames") -> np.ndarray:
    if path.suffix.lower() not in {".npy",".npz"}:
        raise ValueError("manifest arrays must use .npy/.npz float storage")
    loaded = np.load(path,allow_pickle=False)
    if isinstance(loaded,np.lib.npyio.NpzFile):
        try:
            if key not in loaded.files:
                raise ValueError(f"npz must contain {key!r}")
            value = loaded[key]
        finally:
            loaded.close()
    else:
        value = loaded
    if not np.issubdtype(value.dtype,np.floating):
        raise ValueError(f"{path.name} must contain float arrays")
    if not np.isfinite(value).all():
        raise ValueError(f"{path.name} contains nonfinite values")
    return np.array(value,dtype=np.float32,copy=True)


def _payload_hash(array: np.ndarray) -> str:
    """System-computed identity of decoded float32 shape/dtype/bytes.

    Canonical little-endian C order makes identity independent of npy/npz
    packaging and source float dtype after the manifest's float32 conversion.
    A caller-supplied provenance hash is never used as this identity.
    """
    canonical = np.asarray(array,dtype="<f4",order="C")
    descriptor = json.dumps({"dtype":"<f4","shape":list(canonical.shape)},
        sort_keys=True,separators=(",",":")).encode("ascii")
    digest = hashlib.sha256(descriptor+b"\0")
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _content_provenance(provenance: dict, frames: np.ndarray, mask: np.ndarray | None) -> dict:
    value = dict(provenance)
    value["frames_sha256"] = _payload_hash(frames)
    value["payload_hash_schema"] = "sha256_float32_le_shape_bytes_v1"
    # A claimed mask identity cannot survive when no actual mask is present.
    value.pop("mask_sha256",None)
    if mask is not None:
        value["mask_sha256"] = _payload_hash(mask)
    return value


def load_manifest(path, *, splits=None) -> tuple[SensorProfile,list[Scene]]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("unsupported manifest version")
    if "sensor" not in payload or not isinstance(payload.get("scenes"),list) or not payload["scenes"]:
        raise ValueError("manifest requires sensor and nonempty scenes")
    sensor = SensorProfile.from_dict(payload["sensor"])
    scenes, identities, sources = [], set(), {}
    records = []
    for record in payload["scenes"]:
        required = {"scene_id","split","frames_path","frame_times_s","source_kind","domain","source_id"}
        if not isinstance(record,dict) or not required <= record.keys():
            missing = required-set(record) if isinstance(record,dict) else required
            raise ValueError(f"scene missing fields/domain: {sorted(missing)}")
        if record["domain"] != SCENE_DOMAIN:
            raise ValueError(f"unknown scene domain {record['domain']!r}")
        for field in ("scene_id","source_id","frames_path","source_kind"):
            if not isinstance(record[field],str) or not record[field]:
                raise ValueError(f"{field} must be a nonempty string")
        provenance = record.get("provenance",{})
        if not isinstance(provenance,dict):
            raise ValueError("provenance must be an object")
        if record["scene_id"] in identities:
            raise ValueError("duplicate scene identity")
        if record['split'] not in {'train', 'val', 'test'}:
            raise ValueError('split must be train, val, or test')
        source = record["source_id"]
        frame_path = (path.parent/record["frames_path"]).resolve()
        source_keys = [('source_id', source), ('frames_path', str(frame_path))]
        if provenance.get('input_sha256'):
            source_keys.append(('input_sha256', str(provenance['input_sha256'])))
        for source_key in source_keys:
            if source_key in sources and sources[source_key] != record['split']:
                raise ValueError(f'source identity {source_key!r} crosses scene splits')
            sources[source_key] = record['split']
        identities.add(record['scene_id'])
        records.append((record, frame_path, provenance))
    # Validate declared identities even for excluded splits, without opening
    # their pixels. Decoded-content identities can only cover included splits.
    for record, frame_path, provenance in records:
        if splits is not None and record['split'] not in splits:
            continue
        source = record['source_id']
        frame_array = _float_array(frame_path)
        mask_array = None
        if record.get("subject_mask_path") is not None:
            mask_array = _float_array(path.parent/record["subject_mask_path"],"mask")
        provenance = _content_provenance(provenance,frame_array,mask_array)
        source_keys = [("source_id",source),("frames_path",str(frame_path)),
            ("frames_sha256",provenance["frames_sha256"])]
        if provenance.get("input_sha256"):
            source_keys.append(("input_sha256",str(provenance["input_sha256"])))
        for source_key in source_keys:
            if source_key in sources and sources[source_key] != record["split"]:
                raise ValueError(f"source identity {source_key!r} crosses scene splits")
            sources[source_key] = record["split"]
        frames = torch.from_numpy(frame_array)
        mask = None if mask_array is None else torch.from_numpy(mask_array)
        scenes.append(Scene(record["scene_id"],record["split"],frames,
            torch.as_tensor(record["frame_times_s"],dtype=torch.float64),mask,record["source_kind"],
            record["domain"],source,provenance))
    return sensor,scenes


def _write_scene(directory: Path, scene: Scene, sensor: SensorProfile) -> Path:
    directory.mkdir(parents=True,exist_ok=True)
    token = re.sub(r"[^A-Za-z0-9_.-]","_",scene.scene_id)
    if token in {"", ".", ".."}:
        raise ValueError("scene_id cannot form a safe filename")
    token = token[:80]+"_"+hashlib.sha256(scene.scene_id.encode()).hexdigest()[:10]
    frames_name, mask_name = f"{token}_frames.npy",f"{token}_mask.npy"
    frame_array = scene.frames.detach().cpu().numpy()
    mask_array = None if scene.subject_mask is None else scene.subject_mask.detach().cpu().numpy()
    scene.provenance = _content_provenance(scene.provenance,frame_array,mask_array)
    record = {"scene_id":scene.scene_id,"source_id":scene.source_id,"split":scene.split,
        "frames_path":frames_name,"frame_times_s":scene.frame_times_s.cpu().tolist(),
        "source_kind":scene.source_kind,"domain":scene.domain,"provenance":scene.provenance}
    if scene.subject_mask is not None:
        record["subject_mask_path"] = mask_name
    path = directory/"manifest.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("sensor") != json.loads(json.dumps(sensor.to_dict())):
            raise ValueError("cannot append scene using a different sensor profile")
        if any(r["scene_id"] == scene.scene_id for r in payload["scenes"]):
            raise ValueError("scene_id already exists in manifest")
        if any(r.get("source_id",r["scene_id"]) == scene.source_id and r["split"] != scene.split for r in payload["scenes"]):
            raise ValueError("source crosses scene splits")
        source_hash = scene.provenance.get("input_sha256")
        if source_hash and any(r.get("provenance",{}).get("input_sha256") == source_hash and r["split"] != scene.split for r in payload["scenes"]):
            raise ValueError("input source hash crosses scene splits")
        # Recompute prior arrays: metadata in an existing manifest may have
        # been supplied or modified externally and is not a trusted content ID.
        for prior in payload["scenes"]:
            if prior["split"] != scene.split and _payload_hash(_float_array(directory/prior["frames_path"])) == scene.provenance["frames_sha256"]:
                raise ValueError("decoded frame content crosses scene splits")
        payload["scenes"].append(record)
    else:
        payload = {"version":1,"sensor":sensor.to_dict(),"scenes":[record]}
    np.save(directory/frames_name,frame_array)
    if scene.subject_mask is not None:
        np.save(directory/mask_name,mask_array)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return path


def write_demo_dataset(directory, *, seed=0, size=64, scenes=12) -> Path:
    if size < 8 or scenes < 3:
        raise ValueError("demo requires size>=8 and scenes>=3 to provide all splits")
    directory = Path(directory).expanduser().resolve()
    if (directory/"manifest.json").exists():
        raise FileExistsError("demo manifest already exists")
    rng = np.random.default_rng(int(seed))
    sensor = SensorProfile(provenance={"source":"synthetic demo engineering assumptions","seed":int(seed)})
    yy,xx = np.mgrid[-1:1:complex(size),-1:1:complex(size)].astype(np.float32)
    last = None
    train_count = max(1,int(np.floor(.6*scenes)))
    val_count = max(1,int(np.floor(.2*scenes)))
    if scenes == 3:
        train_count,val_count = 1,1
    # Reserve one test scene even for the smallest valid integer dataset.
    val_count = min(val_count,scenes-train_count-1)
    train_count = min(train_count,scenes-val_count-1)
    for index in range(scenes):
        split = "train" if index < train_count else "val" if index < train_count+val_count else "test"
        scene_id = f"demo_{index:03d}"
        color = rng.uniform(.4,1.,size=(3,1,1)).astype(np.float32)
        base = (.025+.06*(xx+1)+.025*(np.sin(xx*16)*np.sin(yy*16)+1))[None]*color
        times = np.linspace(-.25,.25,33,dtype=np.float32) if index%2 else np.array([0.],np.float32)
        speed = 3.0 if index%2 else 0.
        rendered = []
        for t in times:
            blob = np.exp(-((xx-speed*t)**2+(yy+.15)**2)/.045).astype(np.float32)
            highlight = 2.3*np.exp(-((xx+.55)**2+(yy-.5)**2)/.015)
            rendered.append(base+blob[None]*color*.65+highlight[None])
        frames = torch.from_numpy(np.stack(rendered).astype(np.float32))
        mask = torch.from_numpy((((xx)**2+(yy+.15)**2)<.13).astype(np.float32)[None])
        scene = Scene(scene_id,split,frames,torch.from_numpy(times),mask,"synthetic",
            provenance={"generator":"analytic_HDR_moving_subject_v1","seed":int(seed),"absolute_radiance_calibrated":False,"clean_reference_verified":True})
        last = _write_scene(directory,scene,sensor)
    return last
