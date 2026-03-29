import os
import sys

import torch
import torchvision

from wan.vae2_1 import Wan2_1_VAE


def main() -> None:
    print("python:", sys.executable)
    print("conda env:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("conda prefix:", os.environ.get("CONDA_PREFIX"))

    ckpt_dir = "/private/home/francoisporcher/Wan2.2/Wan2.2-I2V-A14B"
    vae_path = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")

    samples_dir = "/private/home/francoisporcher/detrex/samples"
    os.makedirs(samples_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path = "/private/home/francoisporcher/data/waymococo_f0/training_video_tensors/38692.pt"

    video_tensor_raw = torch.load(video_path)
    if video_tensor_raw.dtype != torch.uint8:
        video_tensor_raw = video_tensor_raw.clamp(0, 255).round().to(torch.uint8)

    frames_dir_before = os.path.join(samples_dir, "frames_before")
    frames_dir_after = os.path.join(samples_dir, "frames_after")
    os.makedirs(frames_dir_before, exist_ok=True)
    os.makedirs(frames_dir_after, exist_ok=True)

    for i, frame in enumerate(video_tensor_raw):
        frame_path = os.path.join(frames_dir_before, f"frame_{i:05d}.png")
        torchvision.utils.save_image(frame.float().div(255.0), frame_path)

    frames = video_tensor_raw.float().div(255.0).mul(2.0).sub(1.0)
    frames = frames.unsqueeze(2).to(device)  # [T, C, 1, H, W]
    
    
    # # take first 5 frames for testing
    frames = frames[:5]
    # print(f"Using {frames.shape[0]} frames for testing.")
    

    vae = Wan2_1_VAE(vae_pth=vae_path, device=device)
    vae.model.eval()
    print("VAE loaded on", device)

    total_weights = sum(p.numel() for p in vae.model.parameters())
    encoder_weights = sum(p.numel() for p in vae.model.encoder.parameters())
    decoder_weights = sum(p.numel() for p in vae.model.decoder.parameters())
    print(f"Total weights: {total_weights}")
    print(f"Encoder weights: {encoder_weights}")
    print(f"Decoder weights: {decoder_weights}")
    
    
    

    with torch.no_grad():
        encoded = vae.model.encode(frames, vae.scale).float()
        decoded = vae.model.decode(encoded, vae.scale).float().clamp_(-1, 1)

    decoded_frames = decoded.squeeze(2).detach().cpu()
    for i, frame in enumerate(decoded_frames):
        frame_path = os.path.join(frames_dir_after, f"frame_{i:05d}.png")
        torchvision.utils.save_image(frame.add(1).div(2).clamp(0, 1), frame_path)
    
    print(f"Done.")
    
    
if __name__ == "__main__":
    main()
