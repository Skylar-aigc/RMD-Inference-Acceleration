"""RMD: Few-step Wan video diffusion distillation.

RMD distills a large Wan2.1 text-to-video diffusion model into a student that
generates high-quality videos in K sampling steps (K=4/6/8), with built-in
multi-resolution (480p -> 720p) latent upsampling.
"""

__version__ = "0.1.0"
