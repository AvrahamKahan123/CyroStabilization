"""
Sub-pixel image registration by FFT cross-correlation + parabolic peak fit.
PyTorch implementation — runs on whatever device the input tensors live on
(CPU or CUDA); no host<->device copies are introduced.

Conventions
-----------
- A frame has shape (H, W); dim 0 = rows = y (vertical), dim 1 = cols = x (horizontal).
- A returned shift (delta_x, delta_y) is the *displacement of the frame relative
  to the background*:  frame  ~=  shift_frame(background, delta_x, delta_y).
  To align a frame back onto the background, apply the negated shift:
      aligned = shift_frame(frame, -delta_x, -delta_y)
"""

import torch
import torch.nn.functional as F


def _to_float(t):
    """Cast to a floating dtype (needed for FFT / grid_sample) without moving device."""
    return t if t.is_floating_point() else t.float()


def register_stack_2(stack, background, subpixel=True, phase=False, plot_frame=None):
    """
    Estimate the x/y shift of every frame in a (T, H, W) stack relative to an
    (H, W) background using FFT cross-correlation.

    Parameters
    ----------
    stack : torch.Tensor, shape (T, H, W)
    background : torch.Tensor, shape (H, W)
        Must be on the same device as `stack`.
    subpixel : bool
        If True, refine the integer peak with a 3-point parabolic fit.
    phase : bool
        If True, use normalized (phase) correlation instead of plain
        cross-correlation — a sharper, more robust peak for images with strong
        low-frequency / illumination content.
    plot_frame : int or None
        If given, plot the correlation profiles through the peak of that frame
        (correlation vs Δx and vs Δy) with matplotlib.

    Returns
    -------
    shifts : torch.Tensor, shape (T, 2)
        Rows of (delta_x, delta_y), on the input's device.
    """
    if stack.ndim != 3:
        raise ValueError("stack must be (T, H, W)")
    T, H, W = stack.shape
    if background.shape != (H, W):
        raise ValueError("background must match the (H, W) of the stack")
    if background.device != stack.device:
        raise ValueError("stack and background must be on the same device")

    stack = _to_float(stack)
    background = _to_float(background)

    # Remove the mean so the DC component doesn't dominate the correlation.
    s = stack - stack.mean(dim=(-2, -1), keepdim=True)
    b = background - background.mean()

    Fs = torch.fft.fft2(s, dim=(-2, -1))  # (T, H, W)
    Fb = torch.fft.fft2(b, dim=(-2, -1))  # (H, W), broadcasts over T
    prod = Fs * torch.conj(Fb)

    if phase:
        prod = prod / (prod.abs() + 1e-12)

    corr = torch.fft.ifft2(prod, dim=(-2, -1)).real  # (T, H, W)

    # --- integer peak (vectorised over T) ---
    flat = corr.reshape(T, -1)
    peak = flat.argmax(dim=1)  # (T,)
    peak_row = peak // W  # (T,)
    peak_col = peak % W  # (T,)

    ti = torch.arange(T, device=corr.device)

    dy = torch.where(peak_row > H // 2, peak_row - H, peak_row).to(corr.dtype)
    dx = torch.where(peak_col > W // 2, peak_col - W, peak_col).to(corr.dtype)

    if subpixel:
        y0 = corr[ti, peak_row, peak_col]

        rm = (peak_row - 1) % H
        rp = (peak_row + 1) % H
        y_lo = corr[ti, rm, peak_col]
        y_hi = corr[ti, rp, peak_col]
        den_y = y_lo - 2.0 * y0 + y_hi
        off_y = torch.where(den_y != 0, 0.5 * (y_lo - y_hi) / den_y,
                            torch.zeros_like(den_y))

        cm = (peak_col - 1) % W
        cp = (peak_col + 1) % W
        x_lo = corr[ti, peak_row, cm]
        x_hi = corr[ti, peak_row, cp]
        den_x = x_lo - 2.0 * y0 + x_hi
        off_x = torch.where(den_x != 0, 0.5 * (x_lo - x_hi) / den_x,
                            torch.zeros_like(den_x))

        dy = dy + off_y
        dx = dx + off_x

    if plot_frame is not None:
        import matplotlib.pyplot as plt
        f = plot_frame
        c = corr[f].detach().cpu()
        pr, pc = int(peak_row[f]), int(peak_col[f])
        sx = torch.arange(W);
        sx = torch.where(sx > W // 2, sx - W, sx)  # signed Δx
        sy = torch.arange(H);
        sy = torch.where(sy > H // 2, sy - H, sy)  # signed Δy
        ox, oy = torch.argsort(sx), torch.argsort(sy)
        plt.figure()
        plt.plot(sx[ox].numpy(), c[pr, :][ox].numpy(), label="corr vs Δx (peak row)")
        plt.plot(sy[oy].numpy(), c[:, pc][oy].numpy(), label="corr vs Δy (peak col)")
        plt.axvline(0, color="gray", lw=0.5)
        plt.xlabel("shift (pixels)");
        plt.ylabel("correlation")
        plt.title(f"frame {f}");
        plt.legend();
        plt.tight_layout();
        plt.show()

    return torch.stack((dx, dy), dim=1)  # (T, 2)



def register_stack(stack, background, subpixel=True, phase=False) -> torch.Tensor:
    """
    Estimate the x/y shift of every frame in a (T, H, W) stack relative to an
    (H, W) background using FFT cross-correlation.

    Parameters
    ----------
    stack : torch.Tensor, shape (T, H, W)
    background : torch.Tensor, shape (H, W)
        Must be on the same device as `stack`.
    subpixel : bool
        If True, refine the integer peak with a 3-point parabolic fit.
    phase : bool
        If True, use normalized (phase) correlation instead of plain
        cross-correlation — a sharper, more robust peak for images with strong
        low-frequency / illumination content.

    Returns
    -------
    shifts : torch.Tensor, shape (T, 2)
        Rows of (delta_x, delta_y), on the input's device.
    """
    if stack.ndim != 3:
        raise ValueError("stack must be (T, H, W)")
    T, H, W = stack.shape
    if background.shape != (H, W):
        raise ValueError("background must match the (H, W) of the stack")
    if background.device != stack.device:
        raise ValueError("stack and background must be on the same device")

    stack = _to_float(stack)
    background = _to_float(background)

    # Remove the mean so the DC component doesn't dominate the correlation.
    s = stack - stack.mean(dim=(-2, -1), keepdim=True)
    b = background - background.mean()

    Fs = torch.fft.fft2(s, dim=(-2, -1))          # (T, H, W)
    Fb = torch.fft.fft2(b, dim=(-2, -1))          # (H, W), broadcasts over T
    prod = Fs * torch.conj(Fb) # cross correlation

    if phase:
        prod = prod / (prod.abs() + 1e-12)

    corr = torch.fft.ifft2(prod, dim=(-2, -1)).real   # (T, H, W)

    # --- integer peak (vectorised over T) ---
    flat = corr.reshape(T, -1)
    peak = flat.argmax(dim=1)                     # (T,)
    peak_row = peak // W                          # (T,)
    peak_col = peak % W                           # (T,)

    ti = torch.arange(T, device=corr.device)

    dy = torch.where(peak_row > H // 2, peak_row - H, peak_row).to(corr.dtype)
    dx = torch.where(peak_col > W // 2, peak_col - W, peak_col).to(corr.dtype)

    if subpixel:
        y0 = corr[ti, peak_row, peak_col]

        rm = (peak_row - 1) % H
        rp = (peak_row + 1) % H
        y_lo = corr[ti, rm, peak_col]
        y_hi = corr[ti, rp, peak_col]
        den_y = y_lo - 2.0 * y0 + y_hi
        off_y = torch.where(den_y != 0, 0.5 * (y_lo - y_hi) / den_y,
                            torch.zeros_like(den_y))

        cm = (peak_col - 1) % W
        cp = (peak_col + 1) % W
        x_lo = corr[ti, peak_row, cm]
        x_hi = corr[ti, peak_row, cp]
        den_x = x_lo - 2.0 * y0 + x_hi
        off_x = torch.where(den_x != 0, 0.5 * (x_lo - x_hi) / den_x,
                            torch.zeros_like(den_x))

        dy = dy + off_y
        dx = dx + off_x

    return torch.stack((dx, dy), dim=1)           # (T, 2)


def shift_frame(frame, delta_x, delta_y, mode="bilinear", padding_mode="zeros"):
    """
    Shift a single (H, W) frame by (delta_x, delta_y) using grid_sample, so the
    op runs on the frame's device and is differentiable. Positive delta_x moves
    content toward higher column indices (right); positive delta_y toward higher
    row indices (down). Non-integer shifts are interpolated.

    Parameters
    ----------
    frame : torch.Tensor, shape (H, W)  (also accepts (N, H, W) with scalar shifts)
    delta_x, delta_y : float or 0-d tensor
    mode : {"bilinear", "nearest", "bicubic"}
    padding_mode : {"zeros", "border", "reflection"}
        How to fill samples that fall outside the input.

    Returns
    -------
    torch.Tensor, same shape and device as `frame`.
    """
    squeeze = frame.ndim == 2
    if squeeze:
        frame = frame[None]                       # (1, H, W)
    if frame.ndim != 3:
        raise ValueError("frame must be (H, W) or (N, H, W)")

    x = _to_float(frame)
    N, H, W = x.shape
    dev, dt = x.device, x.dtype

    # Base pixel coordinates; sample the input at (x - dx, y - dy) so that
    # content is displaced by (+dx, +dy).
    ys = torch.arange(H, device=dev, dtype=dt)
    xs = torch.arange(W, device=dev, dtype=dt)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")     # (H, W)

    src_x = gx - float(delta_x)
    src_y = gy - float(delta_y)

    # Normalize to [-1, 1] for grid_sample with align_corners=True.
    norm_x = 2.0 * src_x / max(W - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(H - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)[None]  # (1, H, W, 2)
    grid = grid.expand(N, H, W, 2)

    out = F.grid_sample(x[:, None], grid, mode=mode,
                        padding_mode=padding_mode, align_corners=True)
    out = out[:, 0]                               # (N, H, W)
    return out[0] if squeeze else out


def align_video(video, shifts):
    x = video.float()
    T, H, W = x.shape
    dev, dt = x.device, x.dtype
    dx = -shifts[:, 0].to(dt)          # (T,)  negate = inverse shift
    dy = -shifts[:, 1].to(dt)
    ys = torch.arange(H, device=dev, dtype=dt)
    xs = torch.arange(W, device=dev, dtype=dt)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")          # (H, W)
    src_x = gx[None] - dx[:, None, None]                    # (T, H, W)
    src_y = gy[None] - dy[:, None, None]
    nx = 2.0 * src_x / max(W - 1, 1) - 1.0
    ny = 2.0 * src_y / max(H - 1, 1) - 1.0
    grid = torch.stack((nx, ny), dim=-1)                    # (T, H, W, 2)
    return F.grid_sample(x[:, None], grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)[:, 0]
