import torch
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt

from stabilization import register_stack, align_video, register_stack_2
from tensor_ops import quantile_per_frame
from video_editor import save_tensor_as_video, histogram_stretch_to_uint8

FPS=800

def save_background_difference(
    video: torch.Tensor,
    background: torch.Tensor,
    output_path: str):
    """
    video: torch tensor of shape (T, H, W)
    background: torch tensor of shape (H, W)
    output_path: path ending in .avi
    """

    if video.ndim != 3:
        raise ValueError(f"Expected video shape (T,H,W), got {video.shape}")

    if background.shape != video.shape[1:]:
        raise ValueError(
            f"Background shape {background.shape} must match frame shape {video.shape[1:]}"
        )

    video = video.detach().cpu().float()
    background = background.detach().cpu().float()
    diff = torch.abs(video - background)
    diff = torch.clamp(diff, 0, 255).byte()
    save_tensor_as_video(diff, output_path)



def show_frame(frame: Union[torch.Tensor, np.ndarray]):
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    plt.imshow(frame)
    plt.show()


def plot_lines(values: List[torch.Tensor]):
    for v in values:
        plt.plot(v.detach().cpu().numpy())
    plt.show()

def plot_differences(y_values: torch.Tensor):
    y_values = y_values.detach().cpu().numpy()

    plt.plot(y_values)
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Abs Difference")
    plt.title("Mean Abs Difference from Middle Frame")

    plt.show()

def load_raw_frames(filepath, num_frames=FPS, height=512, width=640):
    expected_elements = num_frames * height * width

    # Load raw binary data
    data = np.fromfile(filepath, dtype=np.uint16, count=expected_elements)

    if data.size != expected_elements:
        raise ValueError(
            f"Unexpected file size. "
            f"Expected {expected_elements} uint16 values, got {data.size}"
        )

    data = data.reshape(num_frames, height, width)
    tensor = torch.from_numpy(data.astype(np.int32))# .cuda() ran out of GPU memory
    return tensor

def fft_time_shift(x: torch.Tensor):
    x_fft = torch.fft.fft(x, dim=0)
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=0)
    return x_fft_shifted

def plot_fft_spectrum(x: torch.Tensor, fps=FPS):
    """
    Compute FFT along dim 0 and plot mean spectrum.

    Args:
        x: Tensor of shape (frames, H, W)
    """
    n_frames = x.shape[0]

    # FFT + shift
    X = torch.fft.fft(x, dim=0)
    X = torch.fft.fftshift(X, dim=0)

    # Average magnitude over spatial dimensions
    spectrum = torch.abs(X).mean(dim=(1, 2))

    # Frequency axis
    freqs = torch.fft.fftfreq(n_frames, d=1/fps)
    freqs = torch.fft.fftshift(freqs)

    # Move to CPU
    spectrum = spectrum.detach().cpu().numpy()
    freqs = freqs.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean DFT Magnitude")
    plt.title("Temporal DFT Spectrum")
    plt.grid(alpha=0.3)
    plt.show()

def save_histogram_equalized_video(video: torch.Tensor, output_path: str):
    video = histogram_stretch_to_uint8(video)
    save_tensor_as_video(video.cpu(), output_path)

def find_x_shifts_and_y_shifts(video_tensor: torch.Tensor, background: torch.Tensor):
    movements = register_stack_2(video_tensor, background, subpixel=True, phase=False, plot_frame=1)
    return movements

def see_differences(frames, middle_frame):
    frames_minus_median = torch.abs(frames - middle_frame).float()  # differences from median
    # differences = frames_minus_median.mean(dim=[1, 2])
    differences = quantile_per_frame(frames_minus_median, 0.9)
    plot_differences(torch.Tensor(differences))

def main():
    frames = load_raw_frames("example.Bin", FPS)
    middle_frame = frames[FPS//2]

    # save_histogram_equalized_video(frames, "all_frames.avi") # save all frames
    # save_background_difference(frames, middle_frame, "differences.avi") # save difference from middle frame
    see_differences(frames, middle_frame) # difference from middle frame

    shifts = find_x_shifts_and_y_shifts(frames, middle_frame)
    aligned_video = align_video(frames, shifts)
    see_differences(aligned_video, middle_frame) # difference from middle frame

    # save_background_difference(aligned_video, middle_frame, "aligned_differences.avi") # save difference from middle frame

    # plot_lines([shifts[:, 0], shifts[:, 1]])
    # exit()
    # show_frame(middle_frame)

    # save_background_difference_avi(frames, middle_frame, "differences.avi") # save difference from middle frame
    # see_differences(aligned_video, middle_frame)
    # differences_dft = fft_time_shift(differences)
    # plot_fft_spectrum(frames_minus_median)


if __name__ == "__main__":
    main()