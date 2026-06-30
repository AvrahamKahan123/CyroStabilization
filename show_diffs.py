import torch, cv2
import numpy as np
from typing import Union
import matplotlib.pyplot as plt

FPS=800

def save_background_difference_avi(
    video: torch.Tensor,
    background: torch.Tensor,
    output_path: str,
    fps: int = 30,
):
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


    diff = torch.clamp(diff, 0, 255).byte().numpy()

    T, H, W = diff.shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=False)

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for i in range(T):
        writer.write(diff[i])

    writer.release()

def show_frame(frame: Union[torch.Tensor, np.ndarray]):
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    plt.imshow(frame)
    plt.show()

def plot_line(values: Union[torch.Tensor, np.ndarray]):
    values = values.detach().cpu().numpy()
    plt.plot(values)
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

def main():
    frames = load_raw_frames("example.Bin", FPS)
    middle_frame = frames[FPS//2]
    # median = frames.median(dim=0)[0]
    # show_frame(middle_frame)
    save_background_difference_avi(frames, middle_frame, "differences.avi")
    exit()
    frames_minus_median = torch.abs(frames - middle_frame).float() # differences from median
    differences = frames_minus_median.mean(dim=[1,2])
    # plot_differences(differences)
    # differences_dft = fft_time_shift(differences)
    plot_fft_spectrum(frames_minus_median)
    # plot_line(differences_dft.abs())
    # print(type(differences))

if __name__ == "__main__":
    main()