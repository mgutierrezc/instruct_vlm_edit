# ----------------------------------------------------------------------------
# Benchmark generation for image_generality
# use the following pipelines.
# generate images given prompts. 

# ------ StableDiffusion2.1Pipeline ---------------------------------------------
# import torch
# from diffusers import StableDiffusionPipeline
# model_id = "stabilityai/stable-diffusion-2-1"
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# prompt = "a medieval knight in shining armor standing in a castle hall, cinematic lighting"
# image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
# image.save("sd21_knight.png")


# ------- StableDiffusion3Pipeline ---------------------------------------------
# import torch
# from diffusers import StableDiffusion3Pipeline
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3-medium-diffusers",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# image = pipe(
#     "a high-quality photo of a snow leopard in the mountains",
#     num_inference_steps=28,
#     guidance_scale=7.0,
# ).images[0]
# image.save("sd3_leopard.png")

# ------- FluxPipeline ---------------------------------------------
# import torch
# from diffusers import FluxPipeline  # name may differ depending on version
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell",
#     torch_dtype=torch.float16,
# )
# pipe.to("cuda")
# image = pipe("a cozy living room with a sleeping cat, cinematic lighting").images[0]
# image.save("flux_cat.png")

# ----------------------------------------------------------------------------
import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline
from huggingface_hub import login


class ImageGenerator:
    def __init__(self, model_name, device="cuda", token=None):
        """
        Initialize image generator with a model.
        
        Args:
            model_name: Model identifier (e.g., "sd2.1", "sd3", "flux", or full model path)
            device: Device to run on (default: "cuda")
            token: Hugging Face token for gated models (default: None, will try to use HF_TOKEN env var)
        """
        self.device = device
        self.model_name = model_name.lower()
        
        # Get token from parameter or environment variable
        hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)
        
        # Load appropriate pipeline based on model name
        if "flux" in self.model_name:
            model_id = "black-forest-labs/FLUX.1-schnell"
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                token=hf_token,
            )
        elif "sd3" in self.model_name or "stable-diffusion-3" in self.model_name:
            model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                token=hf_token,
            )
        elif "sd2" in self.model_name or "stable-diffusion-2" in self.model_name:
            model_id = "stabilityai/stable-diffusion-2-1"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                token=hf_token,
            )
        else:
            # Try to load as a custom model path
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token=hf_token,
            )
        
        self.pipe.to(device)
    
    def generate(self, prompt, num_inference_steps=30, guidance_scale=7.5, save_path=None):
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            num_inference_steps: Number of denoising steps (default: 30)
            guidance_scale: Guidance scale (default: 7.5)
            save_path: Optional path to save the image
        
        Returns:
            PIL Image object
        """
        # Flux doesn't need num_inference_steps or guidance_scale
        if "flux" in self.model_name:
            image = self.pipe(prompt).images[0]
        else:
            # Adjust defaults for SD3
            if "sd3" in self.model_name or "stable-diffusion-3" in self.model_name:
                if num_inference_steps == 30:  # Use default if not specified
                    num_inference_steps = 28
                if guidance_scale == 7.5:  # Use default if not specified
                    guidance_scale = 7.0
            
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        
        if save_path:
            image.save(save_path)
        
        return image