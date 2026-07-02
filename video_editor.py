import torch, cv2


def histogram_stretch_to_uint8(
    video: torch.Tensor,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> torch.Tensor:
    """
    Histogram-stretch an integer video tensor to uint8 using global percentile limits.

    Parameters
    ----------
    video : torch.Tensor
        Input tensor, e.g. shape (frames, height, width), dtype torch.int32.
        May be on CPU or GPU.
    low_percentile : float
        Values below this percentile map to 0.
    high_percentile : float
        Values above this percentile map to 255.

    Returns
    -------
    torch.Tensor
        Same shape/device as `video`, dtype torch.uint8.
    """
    if not video.dtype in (
        torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64
    ):
        raise TypeError(f"Expected an integer tensor, got {video.dtype}")

    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Require 0 <= low_percentile < high_percentile <= 100")

    video_float = video.to(torch.float32)

    low = torch.quantile(video_float[video_float.shape[0]//2], low_percentile / 100.0)
    high = torch.quantile(video_float[video_float.shape[0]//2], high_percentile / 100.0)

    # Constant-valued video: avoid division by zero.
    if high <= low:
        return torch.zeros_like(video, dtype=torch.uint8)

    stretched = (video_float - low) * 255.0 / (high - low)
    stretched = stretched.clamp(0, 255)

    return stretched.to(torch.uint8)

def save_tensor_as_video(tensor: torch.Tensor, output_path: str, fps: int = 30) -> None:
    T, H, W = tensor.shape


    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=False)

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for i in range(T):
        writer.write(tensor[i].numpy())

    writer.release()