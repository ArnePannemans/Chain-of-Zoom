"""
Chain-of-Zoom Service: User-friendly wrapper for extreme super-resolution

This service provides a simple interface for the Chain-of-Zoom inference pipeline,
handling model initialization and providing easy-to-use methods for image processing.

All core logic is preserved from inference_coz.py with references to original line numbers.
"""

# ==================== IMPORTS ====================
# Copied from inference_coz.py lines 12-18
import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

# ==================== TRANSFORMS & GLOBALS ====================
# From inference_coz.py lines 20-24
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

class ChainOfZoomService:
    """
    Chain-of-Zoom Service for extreme super-resolution.
    
    This service encapsulates all model loading and configuration,
    providing a simple interface for processing images.
    """
    
    def __init__(self, 
                 pretrained_model_name_or_path='stabilityai/stable-diffusion-3-medium-diffusers',
                 lora_path='ckpt/SR_LoRA/model_20001.pkl',
                 vae_path='ckpt/SR_VAE/vae_encoder_20001.pt',
                 rec_num=4,
                 upscale=4,
                 process_size=512,
                 seed=42,
                 mixed_precision='fp16',
                 efficient_memory=False,
                 merge_and_unload_lora=False,
                 lora_rank=4,
                 vae_decoder_tiled_size=224,
                 vae_encoder_tiled_size=1024,
                 latent_tiled_size=96,
                 latent_tiled_overlap=32,
                 save_prompts=True):
        """
        Initialize Chain-of-Zoom service with model loading and configuration.
        
        Args:
            pretrained_model_name_or_path: Stable Diffusion model path or name (default: 'stabilityai/stable-diffusion-3-medium-diffusers')
            lora_path: Path to LoRA weights for SR model (default: 'ckpt/SR_LoRA/model_20001.pkl')
            vae_path: Path to VAE weights (default: 'ckpt/SR_VAE/vae_encoder_20001.pt')
            rec_num: Number of recursion steps (default: 4)
            upscale: Upscaling factor per step (default: 4) 
            process_size: Processing size in pixels (square, default: 512)
            seed: Random seed for reproducibility (default: 42)
            mixed_precision: Mixed precision mode ['fp16', 'fp32'] (default: 'fp16')
            efficient_memory: Enable memory-efficient mode (default: False)
            merge_and_unload_lora: Merge LoRA weights before inference (default: False)
            lora_rank: LoRA rank (default: 4)
            vae_decoder_tiled_size: VAE decoder tiling size (default: 224)
            vae_encoder_tiled_size: VAE encoder tiling size (default: 1024)
            latent_tiled_size: Latent space tiling size (default: 96)
            latent_tiled_overlap: Latent space tiling overlap (default: 32)
            save_prompts: Save generated prompts to text files (default: True)
        """
        
        # Store configuration - based on inference_coz.py lines 118-140
        self.config = {
            'pretrained_model_name_or_path': pretrained_model_name_or_path,
            'lora_path': lora_path,
            'vae_path': vae_path,
            'rec_num': rec_num,
            'upscale': upscale,
            'process_size': process_size,
            'seed': seed,
            'mixed_precision': mixed_precision,
            'efficient_memory': efficient_memory,
            'merge_and_unload_lora': merge_and_unload_lora,
            'lora_rank': lora_rank,
            'vae_decoder_tiled_size': vae_decoder_tiled_size,
            'vae_encoder_tiled_size': vae_encoder_tiled_size,
            'latent_tiled_size': latent_tiled_size,
            'latent_tiled_overlap': latent_tiled_overlap,
            'save_prompts': save_prompts,
            # Fixed values from inference_coz.py lines 142-145
            'rec_type': 'recursive_multiscale',
            'prompt_type': 'vlm',
            'align_method': 'nofix',
            'prompt': ''  # Additional user prompt text
        }
        
        print("🚀 Initializing Chain-of-Zoom Service...")
        
        # Validate checkpoint files
        self._validate_checkpoints()
        
        # Initialize models
        self._init_precision()      # From inference_coz.py lines 147-150
        self._init_sr_models()      # From inference_coz.py lines 152-194
        self._init_vlm_models()     # From inference_coz.py lines 212-227
        
        print("✅ Chain-of-Zoom Service initialized successfully!")
    
    def _validate_checkpoints(self):
        """Validate that required checkpoint files exist"""
        missing_files = []
        
        # Check LoRA checkpoint
        if not os.path.exists(self.config['lora_path']):
            missing_files.append(f"LoRA checkpoint: {self.config['lora_path']}")
            
        # Check VAE checkpoint  
        if not os.path.exists(self.config['vae_path']):
            missing_files.append(f"VAE checkpoint: {self.config['vae_path']}")
            
        if missing_files:
            error_msg = "❌ Missing required checkpoint files:\n"
            for file in missing_files:
                error_msg += f"   • {file}\n"
            error_msg += "\n📋 Please ensure you have downloaded the Chain-of-Zoom checkpoints.\n"
            error_msg += "💡 You can download them from the project repository or train your own models.\n"
            error_msg += "🔗 Refer to the README for checkpoint download instructions.\n"
            error_msg += "\nExpected directory structure:\n"
            error_msg += "ckpt/\n"
            error_msg += "├── SR_LoRA/\n" 
            error_msg += "│   └── model_20001.pkl\n"
            error_msg += "└── SR_VAE/\n"
            error_msg += "    └── vae_encoder_20001.pt\n"
            
            raise FileNotFoundError(error_msg)
        
        print("✅ Checkpoint files validated")
    
    def _init_precision(self):
        """Initialize precision mode - from inference_coz.py lines 147-150"""
        global weight_dtype
        weight_dtype = torch.float32
        if self.config['mixed_precision'] == "fp16":
            weight_dtype = torch.float16
            
    def _init_sr_models(self):
        """Initialize Super-Resolution models - from inference_coz.py lines 152-194"""
        # Check available GPUs - from inference_coz.py lines 155-156
        num_gpus = torch.cuda.device_count()
        print(f"🔍 Detected {num_gpus} GPU(s)")
        
        if not self.config['efficient_memory'] and num_gpus >= 2:
            # Multi-GPU setup - from inference_coz.py lines 158-170
            print("📡 Using multi-GPU setup (text encoders on GPU 0, transformer/VAE on GPU 1)")
            from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler
            self.model = SD3Euler()
            self.model.text_enc_1.to('cuda:0')
            self.model.text_enc_2.to('cuda:0')
            self.model.text_enc_3.to('cuda:0')
            self.model.transformer.to('cuda:1', dtype=torch.float32)
            self.model.vae.to('cuda:1', dtype=torch.float32)
            for p in [self.model.text_enc_1, self.model.text_enc_2, self.model.text_enc_3, 
                     self.model.transformer, self.model.vae]:
                p.requires_grad_(False)
            self.model_test = OSEDiff_SD3_TEST(self._create_args_object(), self.model)
            print("✅ SR model loaded (multi-GPU mode)")
        else:
            # Single-GPU or efficient memory setup - from inference_coz.py lines 172-194
            if self.config['efficient_memory']:
                print("💾 Using memory-efficient mode (components moved CPU/GPU on demand)")
            else:
                print("🔧 Falling back to single-GPU mode (all components on GPU 0)")
            
            from osediff_sd3 import OSEDiff_SD3_TEST_efficient, SD3Euler
            self.model = SD3Euler()
            
            if self.config['efficient_memory']:
                # For efficient memory - from inference_coz.py lines 180-182
                self.model.transformer.to('cuda', dtype=torch.float32)
                self.model.vae.to('cuda', dtype=torch.float32)
            else:
                # For single GPU - from inference_coz.py lines 184-189
                self.model.text_enc_1.to('cuda:0')
                self.model.text_enc_2.to('cuda:0') 
                self.model.text_enc_3.to('cuda:0')
                self.model.transformer.to('cuda:0', dtype=torch.float32)
                self.model.vae.to('cuda:0', dtype=torch.float32)
                
            for p in [self.model.text_enc_1, self.model.text_enc_2, self.model.text_enc_3, 
                     self.model.transformer, self.model.vae]:
                p.requires_grad_(False)
            self.model_test = OSEDiff_SD3_TEST_efficient(self._create_args_object(), self.model)
            print("✅ SR model loaded (single-GPU/efficient mode)")
    
    def _init_vlm_models(self):
        """Initialize VLM models - from inference_coz.py lines 212-227"""
        print("🧠 Loading VLM model for contextual understanding...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info as pvi

        # Make these instance variables - from inference_coz.py lines 218-219
        self.process_vision_info = pvi
        
        vlm_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
        print("✅ VLM model loaded")
    
    def _create_args_object(self):
        """Create args object compatible with original inference code"""
        class Args:
            pass
        
        args = Args()
        for key, value in self.config.items():
            setattr(args, key, value)
        return args
        
    def resize_and_center_crop(self, img: Image.Image, size: int) -> Image.Image:
        """
        Resize and center crop an image - from inference_coz.py lines 26-38
        
        Args:
            img: Input PIL Image
            size: Target size (square)
            
        Returns:
            Resized and center-cropped PIL Image
        """
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def get_validation_prompt(self, image, prompt_image_path, device='cuda'):
        """
        Generate validation prompts - adapted from inference_coz.py lines 40-112
        
        Args:
            image: Input PIL Image for SR
            prompt_image_path: List of paths [previous_output, current_zoom]
            device: Device to run inference on
            
        Returns:
            tuple: (prompt_text, low_res_tensor)
        """
        # Prepare low-res tensor - from inference_coz.py line 49
        lq = tensor_transforms(image).unsqueeze(0).to(device)
        
        # VLM processing - from inference_coz.py lines 51-57
        start_image_path = prompt_image_path[0]
        input_image_path = prompt_image_path[1]
        message_text = "The second image is a zoom-in of the first image. Based on this knowledge, what is in the second image? Give me a set of words."
        print(f'START IMAGE PATH: {start_image_path}\nINPUT IMAGE PATH: {input_image_path}\nMESSAGE TEXT: {message_text}')
        
        messages = [
            {"role": "system", "content": f"{message_text}"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": start_image_path},
                    {"type": "image", "image": input_image_path}
                ]
            }
        ]
        print(f'MESSAGES\n{messages}')

        # Process VLM input - from inference_coz.py lines 65-73
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Memory management - from inference_coz.py lines 75-87
        original_sr_devices = {}
        if self.config['efficient_memory'] and hasattr(self.model, 'text_enc_1'):
            print("Moving SR model components to CPU for VLM inference.")
            original_sr_devices['text_enc_1'] = self.model.text_enc_1.device
            original_sr_devices['text_enc_2'] = self.model.text_enc_2.device
            original_sr_devices['text_enc_3'] = self.model.text_enc_3.device
            original_sr_devices['transformer'] = self.model.transformer.device
            original_sr_devices['vae'] = self.model.vae.device
            
            self.model.text_enc_1.to('cpu')
            self.model.text_enc_2.to('cpu')
            self.model.text_enc_3.to('cpu')
            self.model.transformer.to('cpu')
            self.model.vae.to('cpu')

        # Generate text with VLM - from inference_coz.py lines 89-97
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        prompt_text = f"{output_text[0]}, {self.config['prompt']}," if self.config['prompt'] else output_text[0]

        # Restore SR model components - from inference_coz.py lines 101-109
        if self.config['efficient_memory'] and hasattr(self.model, 'text_enc_1'):
            print("Restoring SR model components to original devices.")
            self.model.text_enc_1.to(original_sr_devices['text_enc_1'])
            self.model.text_enc_2.to(original_sr_devices['text_enc_2'])
            self.model.text_enc_3.to(original_sr_devices['text_enc_3'])
            self.model.transformer.to(original_sr_devices['transformer'])
            self.model.vae.to(original_sr_devices['vae'])
            
        return prompt_text, lq

    def process_image(self, input_path, output_dir, prompt=""):
        """
        Process a single image through the Chain-of-Zoom pipeline.
        
        Args:
            input_path: Path to input image file
            output_dir: Directory to save output images
            prompt: Additional prompt text (optional)
            
        Returns:
            str: Path to the final concatenated result
        """
        # Update prompt if provided
        if prompt:
            self.config['prompt'] = prompt
            
        # Setup output directories - from inference_coz.py lines 229-234
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'per-sample'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'per-scale'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'recursive'), exist_ok=True)
        
        return self._process_single_image(input_path, output_dir)
    
    def process_directory(self, input_dir, output_dir, prompt=""):
        """
        Process all PNG images in a directory through the Chain-of-Zoom pipeline.
        
        Args:
            input_dir: Directory containing input images (PNG files)
            output_dir: Directory to save output images
            prompt: Additional prompt text (optional)
            
        Returns:
            list: Paths to final concatenated results for each image
        """
        # Update prompt if provided
        if prompt:
            self.config['prompt'] = prompt
            
        # Gather input images - from inference_coz.py lines 196-201
        image_names = sorted(glob.glob(f'{input_dir}/*.png'))
        print(f"📂 Found {len(image_names)} images in directory: {input_dir}")
        
        if not image_names:
            raise ValueError(f"No PNG images found in directory: {input_dir}")
        
        # Setup output directories - from inference_coz.py lines 229-234
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'per-sample'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'per-scale'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'recursive'), exist_ok=True)
        
        # Print configuration summary - from inference_coz.py lines 236-247
        print(f"\n📊 Configuration Summary:")
        print(f"   • Images to process: {len(image_names)}")
        print(f"   • Recursion type: {self.config['rec_type']} (fixed)")
        print(f"   • Recursion steps: {self.config['rec_num']}")
        print(f"   • Upscale factor: {self.config['upscale']}")
        print(f"   • Prompt type: {self.config['prompt_type']} (fixed)")
        print(f"   • Color alignment: {self.config['align_method']} (fixed)")
        print(f"   • Mixed precision: {self.config['mixed_precision']}")
        print(f"   • Efficient memory: {self.config['efficient_memory']}")
        print(f"   • Output directory: {output_dir}")
        print(f"\n🔥 Starting Chain-of-Zoom inference...")
        print("=" * 60)
        
        results = []
        for idx, image_name in enumerate(image_names):
            print(f"\n🖼️  Processing image {idx+1}/{len(image_names)}: {os.path.basename(image_name)}")
            result = self._process_single_image(image_name, output_dir)
            results.append(result)
            
        print(f"\n🎉 Chain-of-Zoom inference completed!")
        print(f"📁 Results saved to: {output_dir}")
        print("=" * 60)
        
        return results
    
    def _process_single_image(self, image_path, output_dir):
        """
        Process a single image - adapted from inference_coz.py lines 249-404
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            
        Returns:
            str: Path to final concatenated result
        """
        bname = os.path.basename(image_path)
        
        # Setup per-image directories - from inference_coz.py lines 252-257
        rec_dir = os.path.join(output_dir, 'per-sample', bname[:-4])
        os.makedirs(rec_dir, exist_ok=True)
        if self.config['save_prompts']:
            txt_path = os.path.join(rec_dir, 'txt')
            os.makedirs(txt_path, exist_ok=True)

        # Initial image processing - from inference_coz.py lines 259-266
        os.makedirs(os.path.join(output_dir, 'per-scale', 'scale0'), exist_ok=True)
        first_image = Image.open(image_path).convert('RGB')
        first_image = self.resize_and_center_crop(first_image, self.config['process_size'])
        first_image.save(f'{rec_dir}/0.png')
        first_image.save(os.path.join(output_dir, 'per-scale', 'scale0', bname))
        print(f"   📐 Initial image prepared: {self.config['process_size']}x{self.config['process_size']}")

        # Recursive zoom loop - from inference_coz.py lines 268-363
        for rec in range(self.config['rec_num']):
            print(f"\n   🔍 Recursion step {rec+1}/{self.config['rec_num']}")
            os.makedirs(os.path.join(output_dir, 'per-scale', f'scale{rec+1}'), exist_ok=True)
            
            # Prepare input - from inference_coz.py lines 275-290
            prev_sr_output_path = f'{rec_dir}/{rec}.png'
            prev_sr_output_pil = Image.open(prev_sr_output_path).convert('RGB')
            rscale = self.config['upscale']
            w, h = prev_sr_output_pil.size
            new_w, new_h = w // rscale, h // rscale
            
            cropped_region = prev_sr_output_pil.crop(((w-new_w)//2, (h-new_h)//2, (w+new_w)//2, (h+new_h)//2))
            current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
            
            zoomed_image_path = f'{rec_dir}/{rec+1}_input.png'
            current_sr_input_image_pil.save(zoomed_image_path)
            prompt_image_path = [prev_sr_output_path, zoomed_image_path]
            print(f"      📋 Multi-scale mode: Using both previous output and current zoom")

            # Prompt generation - from inference_coz.py lines 292-303
            validation_prompt, lq = self.get_validation_prompt(
                current_sr_input_image_pil, prompt_image_path
            )
            
            if self.config['save_prompts']:
                with open(os.path.join(txt_path, f'{rec}.txt'), 'w', encoding='utf-8') as f:
                    f.write(validation_prompt)
            
            print(f"      💬 Generated prompt: {validation_prompt[:100]}{'...' if len(validation_prompt) > 100 else ''}")

            # Super-resolution inference - from inference_coz.py lines 305-331
            print(f"      🚀 Running super-resolution...")
            with torch.no_grad():
                lq = lq * 2 - 1  # Normalize to [-1, 1]
                
                if self.config['efficient_memory']:
                    print("      🔄 Ensuring SR model components are on CUDA for inference")
                    self.model.text_enc_1.to('cuda')
                    self.model.text_enc_2.to('cuda')
                    self.model.text_enc_3.to('cuda')
                    self.model.transformer.to('cuda', dtype=torch.float32)
                    self.model.vae.to('cuda', dtype=torch.float32)

                output_image = self.model_test(lq, prompt=validation_prompt)
                output_image = torch.clamp(output_image[0].cpu(), -1.0, 1.0)
                output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)

            # Save SR output - from inference_coz.py lines 333-335
            output_pil.save(f'{rec_dir}/{rec+1}.png')
            output_pil.save(os.path.join(output_dir, 'per-scale', f'scale{rec+1}', bname))
            print(f"      ✅ Scale {rec+1} complete")

        # Final concatenation - from inference_coz.py lines 337-351
        print(f"   🔗 Creating final concatenated result...")
        imgs = [Image.open(os.path.join(rec_dir, f'{i}.png')).convert('RGB') for i in range(self.config['rec_num']+1)]
        concat = Image.new('RGB', (sum(im.width for im in imgs), max(im.height for im in imgs)))
        x_off = 0
        for im in imgs:
            concat.paste(im, (x_off, 0))
            x_off += im.width
        
        final_result_path = os.path.join(rec_dir, bname)
        concat.save(final_result_path)
        concat.save(os.path.join(output_dir, 'recursive', bname))
        print(f"   ✅ Image {bname} processing complete!")
        
        return final_result_path


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Example of how to use the service
    
    # Initialize the service once with your model configuration
    coz_service = ChainOfZoomService(
        rec_num=4,              # Number of recursion steps
        upscale=4,              # Upscaling factor per step
        process_size=512,       # Processing size (square)
        mixed_precision='fp16', # Use fp16 for faster inference
        efficient_memory=False, # Set to True if you have limited VRAM
    )
    
    # Process a single image
    # result = coz_service.process_image(
    #     input_path="path/to/your/image.png",
    #     output_dir="path/to/output/directory",
    #     prompt="additional prompt text"  # optional
    # )
    # print(f"Result saved to: {result}")
    
    # Process a directory of images
    # results = coz_service.process_directory(
    #     input_dir="path/to/input/directory",
    #     output_dir="path/to/output/directory", 
    #     prompt="additional prompt text"  # optional
    # )
    # print(f"Processed {len(results)} images") 