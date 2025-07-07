## Changelog

- **2025-07-07**: Streamlined inference code - removed unused recursion types (nearest, bicubic, onestep, recursive), removed DAPE/null prompt types, removed color alignment methods. Now only supports recursive_multiscale + VLM prompts (best performing configuration).

## Ideas

- Switch to TSD-S for better super resolution
- Implement better prompting techniques
- Explore using a better Vision-Language Model (VLM)
- Create a UI to select the region to zoom in on
- Add a region selector based on word input (e.g., user says "dog" and it zooms in on the bounding box of the dog)