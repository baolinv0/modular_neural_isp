"""Explicit source importers; never infer radiometric domains from appearance.

RawGen generates bounded OETF-encoded XYZ. RL-3A provides demosaiced linear
RGB proxies, usually clipped and denoised/noisy. Neither recovers absolute
radiance, real shutter/gain labels or unseen HDR highlights by inversion.
"""
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import torch

from .data import _float_array, _write_scene
from .types import CaptureAction, SCENE_DOMAIN, Scene, SensorProfile


def _read_source(path: Path):
    """Return float HWC/explicit-layout array and storage provenance."""
    suffix = path.suffix.lower()
    if suffix in {".npy",".npz"}:
        return _float_array(path), {"storage":"float_numpy"}
    if suffix == ".png":
        try:
            import png
        except ImportError as error:
            raise ImportError("PNG16 RGB import requires pypng; Pillow RGB decoding can lose 16-bit precision") from error
        width,height,rows,info = png.Reader(filename=str(path)).read()
        if info["bitdepth"] != 16 or info["planes"] != 3 or info.get("greyscale") or info.get("alpha") or "palette" in info:
            raise ValueError("linear/RawGen PNG sources must be RGB 16-bit, without palette/alpha")
        rgb = np.array(list(rows),dtype=np.uint16).reshape(height,width,3)
        return rgb.astype(np.float32)/65535., {"storage":"RGB_uint16_PNG","normalization":"code/65535","native_bayer":False}
    if suffix == ".dng":
        try:
            import rawpy
        except ImportError as error:
            raise ImportError("DNG import requires optional rawpy") from error
        # LibRaw subtracts black and scales the camera range to output_bps.
        # Explicit unity user_wb prevents its otherwise-default daylight WB.
        # This is demosaiced camera RGB; it is not native Bayer ground truth.
        try:
            with rawpy.imread(str(path)) as raw:
                provenance = {"storage":"DNG_rawpy_camera_RGB","native_bayer":False,
                    "demosaiced":True,"black_level_per_channel":list(raw.black_level_per_channel),
                    "white_level":float(raw.white_level),
                    "normalization":"LibRaw black subtraction and camera white scaling to uint16; code/65535",
                    "decoder":{"gamma":[1,1],"no_auto_bright":True,"no_auto_scale":False,
                        "unity_user_wb":True,"output_color":"raw","adjust_maximum_thr":0}}
                rgb = raw.postprocess(gamma=(1,1),no_auto_bright=True,no_auto_scale=False,
                    bright=1,user_wb=[1,1,1,1],use_camera_wb=False,use_auto_wb=False,
                    output_color=rawpy.ColorSpace.raw,output_bps=16,adjust_maximum_thr=0,
                    user_flip=0)
        except Exception as error:
            raise ValueError("DNG cannot be decoded into linear camera RGB; compressed ProRAW may require another decoder, no preview fallback") from error
        return rgb.astype(np.float32)/65535., provenance
    raise ValueError("source must be explicit float .npy/.npz, RGB16 PNG, or supported DNG")


def _frames(array, *, input_layout):
    if input_layout == "HWC" and array.ndim == 3 and array.shape[-1] == 3:
        array = np.moveaxis(array,-1,0)[None]
    elif input_layout == "CHW" and array.ndim == 3 and array.shape[0] == 3:
        array = array[None]
    elif input_layout == "THWC" and array.ndim == 4 and array.shape[-1] == 3:
        array = np.moveaxis(array,-1,1)
    elif input_layout == "TCHW" and array.ndim == 4 and array.shape[1] == 3:
        pass
    else:
        raise ValueError(f"source shape {array.shape} does not match explicit input_layout={input_layout}")
    if not np.isfinite(array).all() or (array<0).any():
        raise ValueError("source radiance must be finite and nonnegative")
    return np.array(array,dtype=np.float32,copy=True)


def _matrix(value):
    matrix = np.asarray(value,dtype=np.float32)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
        raise ValueError("xyz_to_sensor must be a finite nonsingular 3x3 matrix")
    return matrix


def _provenance(path, input_domain, storage):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return {"input_path":str(path),"input_sha256":digest.hexdigest(),"input_domain":input_domain,
        "absolute_radiance_calibrated":False,"relative_source":True,"hdr_recovery_claim":False,
        "clean_reference_verified":False,**storage}


def _save_import(array, output_dir, *, scene_id, split, source_kind, provenance,
        reference_exposure_s, source_id=None, frame_times_s=None, sensor=None):
    profile = SensorProfile(reference_exposure_s=reference_exposure_s) if sensor is None else sensor
    if abs(profile.reference_exposure_s-reference_exposure_s)>1e-12:
        raise ValueError("sensor reference_exposure_s disagrees with import normalization")
    if frame_times_s is None:
        if len(array)>1:
            raise ValueError("multiple source frames require explicit frame_times_s")
        frame_times_s = [0.]
    scene = Scene(scene_id,split,torch.from_numpy(array),torch.as_tensor(frame_times_s,dtype=torch.float64),
        source_kind=source_kind,source_id=source_id or provenance["input_sha256"],provenance=provenance)
    return _write_scene(Path(output_dir).expanduser().resolve(),scene,profile)


def import_rawgen_xyz(input_path, output_dir, *, scene_id, split, xyz_to_sensor,
        reference_exposure_s=1/120, input_domain="rawgen_oetf_xyz", input_layout="HWC",
        source_id=None, frame_times_s=None, sensor=None) -> Path:
    """Import RawGen's XYZ file, inverse OETF once, then explicit camera matrix.

    xyz_to_sensor includes the caller's selected profile/illuminant transform.
    No upper camera clip is introduced. RawGen's bounded source has no verified
    high-light headroom and does not supply a real reference exposure.
    """
    if input_domain != "rawgen_oetf_xyz":
        raise ValueError("RawGen importer input_domain must be rawgen_oetf_xyz; camera previews are not linear truth")
    path = Path(input_path).expanduser().resolve()
    if path.suffix.lower()==".dng":
        raise ValueError("RawGen XYZ importer accepts encoded XYZ, not camera DNG")
    array,storage = _read_source(path)
    encoded = _frames(array,input_layout=input_layout)
    if encoded.max()>1:
        raise ValueError("RawGen OETF-encoded XYZ must lie within [0,1]")
    linear = np.where(encoded<=.04045,encoded/12.92,((encoded+.055)/1.055)**2.4).astype(np.float32)
    matrix = _matrix(xyz_to_sensor)
    camera = np.einsum("ij,tjhw->tihw",matrix,linear)
    provenance = _provenance(path,input_domain,storage)
    provenance.update({"source":"SamsungLabs/RawGen","upstream_commit":"6ca22da610891f9c16d0327d50edeac32f38c988",
        "inverse_oetf_count":1,"xyz_to_sensor":matrix.tolist(),
        "negative_matrix_output_clip_fraction":float(np.mean(camera<0)),
        "source_encoded_upper_endpoint_fraction":float(np.mean(encoded>=1)),
        "limitations":["bounded generative XYZ; cannot recover source clipping or unseen HDR",
            "reference exposure is a simulation anchor, not generated capture EXIF"]})
    return _save_import(np.maximum(camera,0).astype(np.float32),output_dir,scene_id=scene_id,split=split,
        source_kind="rawgen",provenance=provenance,reference_exposure_s=reference_exposure_s,
        source_id=source_id,frame_times_s=frame_times_s,sensor=sensor)


def import_linear_scene(input_path, output_dir, *, scene_id, split, input_domain,
        reference_exposure_s=1/120, xyz_to_sensor=None, input_layout="HWC",
        source_id=None, frame_times_s=None, sensor=None) -> Path:
    """Import explicit relative sensor RGB or linear XYZ; no tone/auto exposure."""
    if input_domain not in {SCENE_DOMAIN,"linear_xyz"}:
        raise ValueError("linear importer requires an explicit supported input_domain")
    path = Path(input_path).expanduser().resolve()
    if input_domain == "linear_xyz" and path.suffix.lower()==".dng":
        raise ValueError("DNG decoder output domain is camera RGB, not linear_xyz")
    array,storage = _read_source(path)
    frames = _frames(array,input_layout=input_layout)
    provenance = _provenance(path,input_domain,storage)
    if input_domain == "linear_xyz":
        if xyz_to_sensor is None:
            raise ValueError("linear XYZ requires xyz_to_sensor matrix")
        matrix = _matrix(xyz_to_sensor)
        frames = np.einsum("ij,tjhw->tihw",matrix,frames)
        provenance.update({"xyz_to_sensor":matrix.tolist(),"negative_matrix_output_clip_fraction":float(np.mean(frames<0))})
        frames = np.maximum(frames,0)
    provenance["limitations"] = ["declared relative linear source; no verified absolute radiometric or sensor calibration"]
    return _save_import(frames.astype(np.float32),output_dir,scene_id=scene_id,split=split,
        source_kind="linear_import",provenance=provenance,reference_exposure_s=reference_exposure_s,
        source_id=source_id,frame_times_s=frame_times_s,sensor=sensor)


def import_rl3a_proxy(input_path, output_dir, *, scene_id, split, exposure_s,
        analog_gain, digital_gain=1., reference_exposure_s=1/120,
        input_domain="rl3a_linear_rgb", input_layout="HWC",source_id=None,
        frame_times_s=None,sensor=None,clean_reference_verified=False) -> Path:
    """Invert supplied capture brightness into an explicitly relative proxy.

    ISO alone does not identify analog/digital gain. Those gains must be supplied
    as measured values or a declared engineering approximation by the caller.
    Input clipping, residual denoising and native noise are not repaired.
    """
    if input_domain != "rl3a_linear_rgb":
        raise ValueError("RL-3A importer requires input_domain=rl3a_linear_rgb")
    reference = CaptureAction(reference_exposure_s)
    capture_action = CaptureAction(exposure_s,analog_gain,digital_gain)
    path = Path(input_path).expanduser().resolve()
    array,storage = _read_source(path)
    frames = _frames(array,input_layout=input_layout)
    provenance = _provenance(path,input_domain,storage)
    provenance.update({"source":"RL-3A relative proxy",
        "upstream_commit":"7b7e031969522dd7e96836bb2507c5c87a4e0b5a",
        "provided_capture_action":capture_action.to_dict(),
        "reference_exposure_s":reference_exposure_s,
        "input_upper_endpoint_fraction":float(np.mean(frames>=1)),
        "input_dark_fraction":float(np.mean(frames<=.003)),
        "clean_reference_verified":bool(clean_reference_verified),
        "gain_split_source":"caller supplied; ISO is not sufficient to identify analog and digital gain",
        "limitations":["relative exposure proxy, not absolute radiance",
            "clipped highlights and dark details are not recovered",
            "demosaicing/denoising/residual native noise can bias additional synthetic noise"]})
    proxy = frames/capture_action.scale(reference)
    return _save_import(proxy.astype(np.float32),output_dir,scene_id=scene_id,split=split,
        source_kind="rl3a_proxy",provenance=provenance,reference_exposure_s=reference_exposure_s,
        source_id=source_id,frame_times_s=frame_times_s,sensor=sensor)


def load_rl3a_noise_profile(path) -> dict:
    """Bridge the natural-residual fit as a named RGB variance model, not e- calibration."""
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("model") != "iso_shot_read_v1":
        raise ValueError("expected RL-3A iso_shot_read_v1 model")
    parameters = value.get("parameters",{})
    try:
        numbers = np.array([parameters.get(k,np.nan) for k in ("a","b","c")],dtype=np.float64)
        iso_ref = float(value.get("iso_ref",np.nan))
    except (TypeError,ValueError,AttributeError) as error:
        raise ValueError("malformed RL-3A noise profile") from error
    if not np.isfinite(numbers).all() or (numbers<0).any() or not np.isfinite(iso_ref) or iso_ref<=0:
        raise ValueError("malformed RL-3A noise profile")
    warnings.warn("RL-3A profile is a natural-image denoising residual fit in normalized RGB, not calibrated electron-domain sensor noise",UserWarning,stacklevel=2)
    return {"model":value["model"],"iso_ref":iso_ref,"parameters":dict(parameters),
        "variance_domain":"normalized_demosaiced_RGB",
        "formula":"a*(ISO/iso_ref)*signal+b*(ISO/iso_ref)^2+c",
        "calibrated_sensor":False,"calibration":value.get("calibration",{}),
        "assumptions":value.get("assumptions",[]),"input_path":str(path)}
