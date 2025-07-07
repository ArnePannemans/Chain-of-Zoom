"""
Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment

This script implements the Chain-of-Zoom inference pipeline for extreme super-resolution.
CoZ factorizes SISR into an autoregressive chain of intermediate scale-states with 
multi-scale-aware prompts, achieving extreme resolutions without additional training.
"""

# ==================== IMPORTS ====================
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
# Transform for converting PIL images to tensors
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# ==================== UTILITY FUNCTIONS ====================
def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    """
    Resize and center crop an image to the specified size.
    
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

# ==================== PROMPT GENERATION ====================
def get_validation_prompt(args, image, prompt_image_path, vlm_model=None, device='cuda'):
    """
    Generate validation prompts for super-resolution using VLM multi-scale analysis.
    
    Args:
        args: Command line arguments
        image: Input PIL Image for SR
        prompt_image_path: List of paths [previous_output, current_zoom] for multi-scale analysis
        vlm_model: VLM model for image understanding
        device: Device to run inference on
        
    Returns:
        tuple: (prompt_text, low_res_tensor)
    """
    # Prepare low-res tensor for SR input
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    
    # Use Vision Language Model (VLM) for multi-scale contextual descriptions
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

    # Process VLM input
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Memory management for efficient inference
    original_sr_devices = {}
    if args.efficient_memory and 'model' in globals() and hasattr(model, 'text_enc_1'):
        print("Moving SR model components to CPU for VLM inference.")
        original_sr_devices['text_enc_1'] = model.text_enc_1.device
        original_sr_devices['text_enc_2'] = model.text_enc_2.device
        original_sr_devices['text_enc_3'] = model.text_enc_3.device
        original_sr_devices['transformer'] = model.transformer.device
        original_sr_devices['vae'] = model.vae.device
        
        model.text_enc_1.to('cpu')
        model.text_enc_2.to('cpu')
        model.text_enc_3.to('cpu')
        model.transformer.to('cpu')
        model.vae.to('cpu')

    # Generate text with VLM
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    prompt_text = f"{output_text[0]}, {args.prompt}," if args.prompt else output_text[0]

    # Restore SR model components after VLM inference
    if args.efficient_memory and 'model' in globals() and hasattr(model, 'text_enc_1'):
        print("Restoring SR model components to original devices.")
        model.text_enc_1.to(original_sr_devices['text_enc_1'])
        model.text_enc_2.to(original_sr_devices['text_enc_2'])
        model.text_enc_3.to(original_sr_devices['text_enc_3'])
        model.transformer.to(original_sr_devices['transformer'])
        model.vae.to(original_sr_devices['vae'])
        
    return prompt_text, lq


# ==================== ARGUMENT PARSING ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chain-of-Zoom: Extreme Super-Resolution (Streamlined Version)')
    
    # Input/Output Configuration
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input', 
                       help='Path to input image or directory')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', 
                       help='Directory to save output images')
    
    # Model Configuration
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, 
                       help='Stable Diffusion model path or name')
    parser.add_argument('--lora_path', type=str, default=None, 
                       help='Path to LoRA weights for SR model')
    parser.add_argument('--vae_path', type=str, default=None, 
                       help='Path to VAE weights')
    
    # Inference Configuration (Fixed to recursive_multiscale)
    parser.add_argument('--rec_num', type=int, default=4, 
                       help='Number of recursion steps')
    parser.add_argument('--upscale', type=int, default=4, 
                       help='Upscaling factor per step')
    parser.add_argument('--process_size', type=int, default=512, 
                       help='Processing size (square)')
    
    # Prompt Configuration (Fixed to VLM)
    parser.add_argument('--prompt', type=str, default='', 
                       help='Additional user prompt text')
    parser.add_argument('--save_prompts', type=bool, default=True, 
                       help='Save generated prompts to text files')
    
    # Technical Configuration
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'fp32'], 
                       default='fp16', help='Mixed precision mode')
    parser.add_argument('--efficient_memory', default=False, action='store_true', 
                       help='Enable memory-efficient mode (slower but uses less VRAM)')
    
    # Advanced LoRA Configuration
    parser.add_argument('--merge_and_unload_lora', action='store_true', 
                       help='Merge LoRA weights before inference')
    parser.add_argument('--lora_rank', type=int, default=4, 
                       help='LoRA rank')
    
    # Tiling Configuration for Large Images
    parser.add_argument('--vae_decoder_tiled_size', type=int, default=224, 
                       help='VAE decoder tiling size')
    parser.add_argument('--vae_encoder_tiled_size', type=int, default=1024, 
                       help='VAE encoder tiling size')
    parser.add_argument('--latent_tiled_size', type=int, default=96, 
                       help='Latent space tiling size')
    parser.add_argument('--latent_tiled_overlap', type=int, default=32, 
                       help='Latent space tiling overlap')
    
    args = parser.parse_args()
    
    # Set fixed values for streamlined version
    args.rec_type = 'recursive_multiscale'  # Only supported recursion type
    args.prompt_type = 'vlm'                # Only supported prompt type
    args.align_method = 'nofix'             # No color alignment

    # ==================== SETUP AND INITIALIZATION ====================
    
    # Set precision mode
    global weight_dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # Initialize Super-Resolution model (always needed for recursive_multiscale)
    print("🚀 Initializing Chain-of-Zoom inference...")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {num_gpus} GPU(s)")
    
    if not args.efficient_memory and num_gpus >= 2:
        # Multi-GPU setup (text encoders on GPU 0, transformer/VAE on GPU 1)
        print("📡 Using multi-GPU setup (text encoders on GPU 0, transformer/VAE on GPU 1)")
        from osediff_sd3 import OSEDiff_SD3_TEST, SD3Euler
        model = SD3Euler()
        model.text_enc_1.to('cuda:0')
        model.text_enc_2.to('cuda:0')
        model.text_enc_3.to('cuda:0')
        model.transformer.to('cuda:1', dtype=torch.float32)
        model.vae.to('cuda:1', dtype=torch.float32)
        for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
            p.requires_grad_(False)
        model_test = OSEDiff_SD3_TEST(args, model)
        print("✅ SR model loaded (multi-GPU mode)")
    else:
        # Single-GPU or efficient memory setup
        if args.efficient_memory:
            print("💾 Using memory-efficient mode (components moved CPU/GPU on demand)")
        else:
            print("🔧 Falling back to single-GPU mode (all components on GPU 0)")
        
        from osediff_sd3 import OSEDiff_SD3_TEST_efficient, SD3Euler
        model = SD3Euler()
        
        if args.efficient_memory:
            # For efficient memory, only keep transformer and VAE on GPU initially
            model.transformer.to('cuda', dtype=torch.float32)
            model.vae.to('cuda', dtype=torch.float32)
        else:
            # For single GPU, put everything on cuda:0
            model.text_enc_1.to('cuda:0')
            model.text_enc_2.to('cuda:0') 
            model.text_enc_3.to('cuda:0')
            model.transformer.to('cuda:0', dtype=torch.float32)
            model.vae.to('cuda:0', dtype=torch.float32)
            
        for p in [model.text_enc_1, model.text_enc_2, model.text_enc_3, model.transformer, model.vae]:
            p.requires_grad_(False)
        model_test = OSEDiff_SD3_TEST_efficient(args, model)
        print("✅ SR model loaded (single-GPU/efficient mode)")

    # Gather input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
        print(f"📂 Found {len(image_names)} images in directory: {args.input_image}")
    else:
        image_names = [args.input_image]
        print(f"📷 Single image: {args.input_image}")

    # Load VLM pipeline (only supported prompt type)
    print("🧠 Loading VLM model for contextual understanding...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info as pvi

    # Make these global for use in the prompt generation function
    global vlm_processor, process_vision_info
    process_vision_info = pvi
    
    vlm_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
    print("✅ VLM model loaded")

    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per-sample'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per-scale'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'recursive'), exist_ok=True)
    
    # Print configuration summary
    print(f"\n📊 Configuration Summary (Streamlined Version):")
    print(f"   • Images to process: {len(image_names)}")
    print(f"   • Recursion type: {args.rec_type} (fixed)")
    print(f"   • Recursion steps: {args.rec_num}")
    print(f"   • Upscale factor: {args.upscale}")
    print(f"   • Prompt type: {args.prompt_type} (fixed)")
    print(f"   • Color alignment: {args.align_method} (fixed)")
    print(f"   • Mixed precision: {args.mixed_precision}")
    print(f"   • Efficient memory: {args.efficient_memory}")
    print(f"   • Output directory: {args.output_dir}")
    print(f"\n🔥 Starting Chain-of-Zoom inference...")
    print("=" * 60)

    # ==================== MAIN INFERENCE LOOP ====================
    for idx, image_name in enumerate(image_names):
        bname = os.path.basename(image_name)
        print(f"\n🖼️  Processing image {idx+1}/{len(image_names)}: {bname}")
        
        # Setup per-image directories
        rec_dir = os.path.join(args.output_dir, 'per-sample', bname[:-4])
        os.makedirs(rec_dir, exist_ok=True)
        if args.save_prompts:
            txt_path = os.path.join(rec_dir, 'txt')
            os.makedirs(txt_path, exist_ok=True)

        # ==================== INITIAL IMAGE PROCESSING ====================
        # Load and prepare the initial image (scale 0)
        os.makedirs(os.path.join(args.output_dir, 'per-scale', 'scale0'), exist_ok=True)
        first_image = Image.open(image_name).convert('RGB')
        first_image = resize_and_center_crop(first_image, args.process_size)
        first_image.save(f'{rec_dir}/0.png')
        first_image.save(os.path.join(args.output_dir, 'per-scale', 'scale0', bname))
        print(f"   📐 Initial image prepared: {args.process_size}x{args.process_size}")

        # ==================== RECURSIVE ZOOM LOOP ====================
        for rec in range(args.rec_num):
            print(f"\n   🔍 Recursion step {rec+1}/{args.rec_num}")
            os.makedirs(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}'), exist_ok=True)
            
            # Initialize variables for this recursion step
            current_sr_input_image_pil = None
            prompt_image_path = None
            
            # ==================== PREPARE INPUT (RECURSIVE MULTISCALE ONLY) ====================
            # Use both previous SR output and current zoom level for multi-scale context
            prev_sr_output_path = f'{rec_dir}/{rec}.png'
            prev_sr_output_pil = Image.open(prev_sr_output_path).convert('RGB')
            rscale = args.upscale
            w, h = prev_sr_output_pil.size
            new_w, new_h = w // rscale, h // rscale
            
            # Crop center and resize for next SR step
            cropped_region = prev_sr_output_pil.crop(((w-new_w)//2, (h-new_h)//2, (w+new_w)//2, (h+new_h)//2))
            current_sr_input_image_pil = cropped_region.resize((w, h), Image.BICUBIC)
            
            # Save zoomed image and provide both images for multi-scale prompt
            zoomed_image_path = f'{rec_dir}/{rec+1}_input.png'
            current_sr_input_image_pil.save(zoomed_image_path)
            prompt_image_path = [prev_sr_output_path, zoomed_image_path]
            print(f"      📋 Multi-scale mode: Using both previous output and current zoom")

            # ==================== PROMPT GENERATION ====================
            validation_prompt, lq = get_validation_prompt(
                args, current_sr_input_image_pil, prompt_image_path, vlm_model
            )
            
            # Save prompt if requested
            if args.save_prompts:
                with open(os.path.join(txt_path, f'{rec}.txt'), 'w', encoding='utf-8') as f:
                    f.write(validation_prompt)
            
            print(f"      💬 Generated prompt: {validation_prompt[:100]}{'...' if len(validation_prompt) > 100 else ''}")

            # ==================== SUPER-RESOLUTION INFERENCE ====================
            print(f"      🚀 Running super-resolution...")
            with torch.no_grad():
                # Normalize input to [-1, 1] range
                lq = lq * 2 - 1
                
                # Ensure SR model components are on correct device for inference
                if args.efficient_memory:
                    print("      🔄 Ensuring SR model components are on CUDA for inference")
                    # For efficient memory mode, move text encoders to GPU if needed
                    model.text_enc_1.to('cuda')
                    model.text_enc_2.to('cuda')
                    model.text_enc_3.to('cuda')
                    model.transformer.to('cuda', dtype=torch.float32)
                    model.vae.to('cuda', dtype=torch.float32)

                # Run super-resolution
                output_image = model_test(lq, prompt=validation_prompt)
                output_image = torch.clamp(output_image[0].cpu(), -1.0, 1.0)
                output_pil = transforms.ToPILImage()(output_image * 0.5 + 0.5)
                
                # Note: Color alignment removed in streamlined version

            # Save the SR output
            output_pil.save(f'{rec_dir}/{rec+1}.png')
            output_pil.save(os.path.join(args.output_dir, 'per-scale', f'scale{rec+1}', bname))
            print(f"      ✅ Scale {rec+1} complete")

        # ==================== FINAL CONCATENATION ====================
        print(f"   🔗 Creating final concatenated result...")
        # Load all scale images and create horizontal concatenation
        imgs = [Image.open(os.path.join(rec_dir, f'{i}.png')).convert('RGB') for i in range(args.rec_num+1)]
        concat = Image.new('RGB', (sum(im.width for im in imgs), max(im.height for im in imgs)))
        x_off = 0
        for im in imgs:
            concat.paste(im, (x_off, 0))
            x_off += im.width
        
        # Save concatenated result
        concat.save(os.path.join(rec_dir, bname))
        concat.save(os.path.join(args.output_dir, 'recursive', bname))
        print(f"   ✅ Image {bname} processing complete!")

    print(f"\n🎉 Chain-of-Zoom inference completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("=" * 60)