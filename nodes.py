import os
import sys
import gc
import cv2
import shutil
import tempfile
import traceback
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = THIS_DIR / "third_party"
MATANYONE2_REPO_DIR = THIRD_PARTY_DIR / "MatAnyone2"

COMFYUI_ROOT = THIS_DIR.parent.parent
COMFY_MODELS_DIR = COMFYUI_ROOT / "models"
MATANYONE2_MODELS_DIR = COMFY_MODELS_DIR / "MatAnyone2"

if MATANYONE2_REPO_DIR.exists():
    sys.path.insert(0, str(MATANYONE2_REPO_DIR))


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_rel(path: Path, base: Path):
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _resolve_model_candidates():
    return [
        MATANYONE2_MODELS_DIR / "matanyone2.pth",
        COMFY_MODELS_DIR / "matanyone2.pth",
        MATANYONE2_REPO_DIR / "pretrained_models" / "matanyone2.pth",
    ]


def _ensure_official_model_visible():
    """
    优先使用:
      ComfyUI/models/MatAnyone2/matanyone2.pth

    然后链接/复制到:
      third_party/MatAnyone2/pretrained_models/matanyone2.pth
    """
    src_model = None
    for p in _resolve_model_candidates():
        if p.exists():
            src_model = p
            break

    if src_model is None:
        return None, None

    official_dir = MATANYONE2_REPO_DIR / "pretrained_models"
    _ensure_dir(official_dir)
    official_model = official_dir / "matanyone2.pth"

    try:
        if official_model.exists() and official_model.resolve() == src_model.resolve():
            return src_model, official_model
    except Exception:
        pass

    if not official_model.exists():
        try:
            os.symlink(str(src_model), str(official_model))
        except Exception:
            try:
                shutil.copy2(str(src_model), str(official_model))
            except Exception:
                pass

    return src_model, official_model


def _to_numpy_uint8_image(img_tensor) -> np.ndarray:
    """
    Comfy 单帧 IMAGE: [H, W, C], float32 0..1
    -> uint8 RGB
    """
    if img_tensor is None:
        raise ValueError("Image tensor is None")

    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.asarray(img_tensor)

    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected image shape [H, W, C], got {img.shape}")

    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def _batch_to_frame_list(images):
    """
    Comfy IMAGE batch: [N, H, W, C]
    """
    if images is None:
        raise ValueError("src_video is None")

    if isinstance(images, torch.Tensor):
        arr = images
    else:
        arr = torch.tensor(images)

    if arr.ndim != 4:
        raise ValueError(f"src_video must be [N,H,W,C], got {tuple(arr.shape)}")

    frames = []
    for i in range(arr.shape[0]):
        frames.append(_to_numpy_uint8_image(arr[i]))
    return frames


def _extract_single_image_from_batch(image_batch, frame_index=0):
    """
    从 IMAGE batch 中取一帧，返回 [H,W,C] uint8 RGB
    """
    if image_batch is None:
        return None

    if isinstance(image_batch, torch.Tensor):
        arr = image_batch.detach().cpu()
    else:
        arr = torch.tensor(image_batch)

    if arr.ndim == 4:
        frame_index = max(0, min(int(frame_index), arr.shape[0] - 1))
        return _to_numpy_uint8_image(arr[frame_index])
    elif arr.ndim == 3:
        return _to_numpy_uint8_image(arr)
    else:
        raise ValueError(f"Unsupported IMAGE shape: {tuple(arr.shape)}")


def _mask_tensor_to_gray(mask_tensor, target_h, target_w, frame_index=0):
    """
    MASK -> uint8 gray [H,W]
    """
    if mask_tensor is None:
        return None

    if isinstance(mask_tensor, torch.Tensor):
        arr = mask_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(mask_tensor)

    # [N,H,W] or [H,W]
    if arr.ndim == 3:
        frame_index = max(0, min(int(frame_index), arr.shape[0] - 1))
        arr = arr[frame_index]

    if arr.ndim != 2:
        raise ValueError(f"Unsupported MASK shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)

    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    _, arr = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    return arr


def _image_to_gray_mask(image_tensor, target_h, target_w, frame_index=0):
    """
    IMAGE -> 取某一帧 -> 转单通道 mask
    """
    if image_tensor is None:
        return None

    rgb = _extract_single_image_from_batch(image_tensor, frame_index=frame_index)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if gray.shape[:2] != (target_h, target_w):
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return gray


def _resolve_first_frame_mask(src_video_frames, foreground_mask=None, foreground_MASK=None, mask_frame=0):
    """
    优先:
      1) foreground_MASK
      2) foreground_mask (IMAGE)
    """
    h, w = src_video_frames[0].shape[:2]

    if foreground_MASK is not None:
        m = _mask_tensor_to_gray(foreground_MASK, h, w, frame_index=mask_frame)
        if m is not None:
            return m

    if foreground_mask is not None:
        m = _image_to_gray_mask(foreground_mask, h, w, frame_index=mask_frame)
        if m is not None:
            return m

    raise ValueError("Need foreground_MASK or foreground_mask")


def _save_frames_as_folder(frames, out_dir: Path):
    _ensure_dir(out_dir)
    for i, frame_rgb in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), frame_bgr)


def _save_mask(mask_gray: np.ndarray, out_path: Path):
    _ensure_dir(out_path.parent)
    Image.fromarray(mask_gray, mode="L").save(str(out_path))


def _frames_to_comfy_image(frames):
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _gray_frames_to_rgb_frames(frames_gray):
    out = []
    for g in frames_gray:
        if g.ndim == 2:
            out.append(np.stack([g, g, g], axis=-1))
        else:
            out.append(g)
    return out


def _make_green_background_batch(num_frames, h, w, solid_color=None):
    """
    solid_color 有就用它，没有就用纯绿
    """
    if solid_color is not None:
        try:
            img = _extract_single_image_from_batch(solid_color, frame_index=0)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            return [img.copy() for _ in range(num_frames)]
        except Exception:
            pass

    green = np.zeros((h, w, 3), dtype=np.uint8)
    green[:, :, 1] = 255
    return [green.copy() for _ in range(num_frames)]


def _composite_on_background(src_frames, alpha_frames, bg_frames):
    out = []
    for src, a, bg in zip(src_frames, alpha_frames, bg_frames):
        if a.ndim == 2:
            alpha = a.astype(np.float32) / 255.0
            alpha = alpha[..., None]
        else:
            alpha = a[..., :1].astype(np.float32) / 255.0

        src_f = src.astype(np.float32)
        bg_f = bg.astype(np.float32)
        comp = src_f * alpha + bg_f * (1.0 - alpha)
        out.append(np.clip(comp, 0, 255).astype(np.uint8))
    return out


def _clear_results_dir():
    results_root = MATANYONE2_REPO_DIR / "results"
    if results_root.exists():
        try:
            shutil.rmtree(results_root, ignore_errors=True)
        except Exception:
            pass


def _run_official_with_subprocess(input_dir: Path, mask_path: Path, max_internal_size: int = -1):
    script_path = MATANYONE2_REPO_DIR / "inference_matanyone2.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Official inference script not found: {script_path}")

    src_model, official_model = _ensure_official_model_visible()
    _clear_results_dir()

    cmd = [
        sys.executable,
        str(script_path),
        "-i", str(input_dir),
        "-m", str(mask_path),
        "--save_image",
    ]

    if isinstance(max_internal_size, int) and max_internal_size > 0:
        cmd.extend(["--max_size", str(max_internal_size)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(MATANYONE2_REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    if src_model is not None:
        env["MATANYONE2_MODEL_PATH"] = str(src_model)

    process = subprocess.run(
        cmd,
        cwd=str(MATANYONE2_REPO_DIR),
        env=env,
        capture_output=True,
        text=True
    )

    return {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "src_model": str(src_model) if src_model else "",
        "official_model": str(official_model) if official_model else "",
        "cmd": cmd,
    }


def _find_results_root():
    results_root = MATANYONE2_REPO_DIR / "results"
    if not results_root.exists():
        raise FileNotFoundError(f"Results folder not found: {results_root}")
    return results_root


def _read_video_frames(video_path: Path, force_gray=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if force_gray:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray)
        else:
            if frame.ndim == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frames.append(rgb)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    return frames


def _results_tree_text(results_root: Path):
    all_files = sorted(results_root.glob("**/*"))
    lines = []
    for p in all_files:
        if p.is_file():
            try:
                size = p.stat().st_size
            except Exception:
                size = -1
            lines.append(f"{_safe_rel(p, results_root)} | {size} bytes")
    return "\n".join(lines[:500])


def _collect_alpha_from_results(results_root: Path):
    """
    优先级：
    1) alpha/pha/matte/mask 相关逐帧图片
    2) alpha/pha/matte/mask 相关视频
    3) RGBA 逐帧图片
    4) 只有一个视频时，按 alpha 读取
    """
    img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    alpha_keywords = ("alpha", "pha", "matte", "mask")

    all_files = [p for p in results_root.glob("**/*") if p.is_file()]
    img_files = sorted([p for p in all_files if p.suffix.lower() in img_exts], key=lambda x: x.stat().st_mtime, reverse=True)
    video_files = sorted([p for p in all_files if p.suffix.lower() in video_exts], key=lambda x: x.stat().st_mtime, reverse=True)

    # 1) alpha 图片文件夹
    by_parent = {}
    for p in img_files:
        by_parent.setdefault(p.parent, []).append(p)

    alpha_parent_candidates = []
    for parent, paths in by_parent.items():
        lname = parent.name.lower()
        if any(k in lname for k in alpha_keywords):
            try:
                newest = max(x.stat().st_mtime for x in paths)
            except Exception:
                newest = 0
            alpha_parent_candidates.append((newest, parent, paths))

    alpha_parent_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, parent, paths in alpha_parent_candidates:
        try:
            alpha_frames = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in sorted(paths)]
            if alpha_frames:
                return alpha_frames
        except Exception:
            pass

    # 2) alpha 视频
    for p in video_files:
        lname = p.name.lower()
        if any(k in lname for k in alpha_keywords):
            try:
                alpha_frames = _read_video_frames(p, force_gray=True)
                if alpha_frames:
                    return alpha_frames
            except Exception:
                pass

    # 3) RGBA 图片序列
    rgba_parent_candidates = []
    for parent, paths in by_parent.items():
        try:
            sample = Image.open(sorted(paths)[0])
            if sample.mode == "RGBA":
                newest = max(x.stat().st_mtime for x in paths)
                rgba_parent_candidates.append((newest, parent, paths))
        except Exception:
            pass

    rgba_parent_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, parent, paths in rgba_parent_candidates:
        try:
            rgba_frames = [np.array(Image.open(p).convert("RGBA"), dtype=np.uint8) for p in sorted(paths)]
            return [f[:, :, 3] for f in rgba_frames]
        except Exception:
            pass

    # 4) 只有一个视频时兜底
    if len(video_files) == 1:
        return _read_video_frames(video_files[0], force_gray=True)

    raise RuntimeError(
        "Could not infer alpha outputs from results folder.\n"
        "Results tree:\n" + _results_tree_text(results_root)
    )


class SolidColorBatched:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 999999, "step": 1}),
                "height": ("INT", {"default": 720, "min": 1, "max": 8192, "step": 1}),
                "width": ("INT", {"default": 1280, "min": 1, "max": 8192, "step": 1}),
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("solid",)
    FUNCTION = "generate"
    CATEGORY = "image/generate"

    def generate(self, batch_size, height, width, red, green, blue):
        img = np.zeros((batch_size, height, width, 3), dtype=np.float32)
        img[..., 0] = float(red) / 255.0
        img[..., 1] = float(green) / 255.0
        img[..., 2] = float(blue) / 255.0
        return (torch.from_numpy(img),)


class MatAnyone2Compatible:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_video": ("IMAGE",),
                "mask_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "n_warmup": ("INT", {"default": 10, "min": 0, "max": 256, "step": 1}),
                "max_internal_size": ("INT", {"default": -1, "min": -1, "max": 8192, "step": 16}),
                "max_mem_frames": ("INT", {"default": 5, "min": 1, "max": 1024, "step": 1}),
                "use_long_term": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "foreground_mask": ("IMAGE",),
                "foreground_MASK": ("MASK",),
                "solid_color": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("matte", "green_screen")
    FUNCTION = "run"
    CATEGORY = "video/matting"

    def run(
        self,
        src_video,
        mask_frame=0,
        n_warmup=10,
        max_internal_size=-1,
        max_mem_frames=5,
        use_long_term=False,
        foreground_mask=None,
        foreground_MASK=None,
        solid_color=None,
    ):
        temp_dir = None

        try:
            if not MATANYONE2_REPO_DIR.exists():
                raise FileNotFoundError(f"Official repo missing:\n{MATANYONE2_REPO_DIR}")

            src_frames = _batch_to_frame_list(src_video)
            if len(src_frames) == 0:
                raise ValueError("src_video is empty")

            first_mask = _resolve_first_frame_mask(
                src_video_frames=src_frames,
                foreground_mask=foreground_mask,
                foreground_MASK=foreground_MASK,
                mask_frame=mask_frame,
            )

            temp_dir = Path(tempfile.mkdtemp(prefix="comfy_matanyone2_"))
            input_dir = temp_dir / "video_frames"
            mask_path = temp_dir / "first_frame_mask.png"

            _save_frames_as_folder(src_frames, input_dir)
            _save_mask(first_mask, mask_path)

            result = _run_official_with_subprocess(
                input_dir=input_dir,
                mask_path=mask_path,
                max_internal_size=max_internal_size,
            )

            if result["returncode"] != 0:
                raise RuntimeError(
                    "MatAnyone2 inference failed\n"
                    f"CMD: {' '.join(result['cmd'])}\n"
                    f"MODEL(src): {result['src_model']}\n"
                    f"MODEL(official): {result['official_model']}\n\n"
                    f"STDOUT:\n{result['stdout']}\n\n"
                    f"STDERR:\n{result['stderr']}"
                )

            results_root = _find_results_root()
            alpha_frames = _collect_alpha_from_results(results_root)

            if not alpha_frames:
                raise RuntimeError("No valid alpha output found")

            frame_count = min(len(alpha_frames), len(src_frames))
            src_frames = src_frames[:frame_count]
            alpha_frames = alpha_frames[:frame_count]

            h, w = src_frames[0].shape[:2]

            fixed_alpha = []
            for a in alpha_frames:
                if a.shape[:2] != (h, w):
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                fixed_alpha.append(a)
            alpha_frames = fixed_alpha

            bg_frames = _make_green_background_batch(frame_count, h, w, solid_color=solid_color)
            green_frames = _composite_on_background(src_frames, alpha_frames, bg_frames)

            matte_rgb_frames = _gray_frames_to_rgb_frames(alpha_frames)

            matte_tensor = _frames_to_comfy_image(matte_rgb_frames)
            green_tensor = _frames_to_comfy_image(green_frames)

            return (matte_tensor, green_tensor)

        except Exception as e:
            print("\n[ComfyUI-MatAnyone2] ERROR:")
            print(str(e))
            print(traceback.format_exc())

            empty = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty, empty)

        finally:
            try:
                if temp_dir is not None and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass


NODE_CLASS_MAPPINGS = {
    "MatAnyone": MatAnyone2Compatible,
    "MatAnyone2": MatAnyone2Compatible,
    "MatAnyone2Compatible": MatAnyone2Compatible,
    "SolidColorBatched": SolidColorBatched,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyone": "MatAnyone2",
    "MatAnyone2": "MatAnyone2",
    "MatAnyone2Compatible": "MatAnyone2",
    "SolidColorBatched": "SolidColorBatched",
}