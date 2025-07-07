"""
Example usage of the Chain-of-Zoom Service

This file demonstrates how to use the ChainOfZoomService for:
1. Processing a single image
2. Processing a directory of images 
3. Different configuration options

"""

from coz_service import ChainOfZoomService

def example_single_image():
    """Example: Process a single image"""
    print("=== Single Image Processing Example ===")
    

    coz_service = ChainOfZoomService(
        rec_num=6,              # Fewer steps for faster processing
        upscale=3,              # 4x upscaling per step
        process_size=768,       # 512x512 processing size
        mixed_precision='fp16', # Use fp16 for faster inference
        efficient_memory=True,  # Use less VRAM (slower but more memory efficient)
    )
    
    # Process a single image
    input_path = "samples/tests/keyboard.png"  # Replace with your image path
    output_dir = "output/single_image_test"
    
    try:
        result_path = coz_service.process_image(
            input_path=input_path,
            output_dir=output_dir,
            prompt="high quality, detailed"  # Optional additional prompt
        )
        print(f"✅ Processing complete! Result saved to: {result_path}")
    except Exception as e:
        print(f"❌ Error processing image: {e}")

def example_batch_processing():
    """Example: Process multiple images in a directory"""
    print("\n=== Batch Processing Example ===")
    
    # Initialize service for batch processing
    coz_service = ChainOfZoomService(
        rec_num=4,              # Full 4 recursion steps
        upscale=4,              # 4x upscaling per step  
        process_size=512,       # 512x512 processing size
        mixed_precision='fp16', # Use fp16 for speed
        efficient_memory=False, # Use more VRAM for faster processing
        save_prompts=True,      # Save generated prompts for analysis
    )
    
    # Process all PNG images in a directory
    input_dir = "preset/datasets/test_dataset/input"    # Replace with your input directory
    output_dir = "output/batch_processing_test"
    
    try:
        result_paths = coz_service.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            prompt="ultra high resolution, sharp details"  # Optional additional prompt
        )
        print(f"✅ Batch processing complete! Processed {len(result_paths)} images")
        for i, path in enumerate(result_paths, 1):
            print(f"   {i}. {path}")
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")

if __name__ == "__main__":
    print("Chain-of-Zoom Service Examples")
    print("=" * 50)
    
    example_single_image()
    