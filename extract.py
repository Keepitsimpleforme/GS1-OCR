import ollama
import os

# --- 1. Configuration ---
# Place your image inside the 'inputs' folder
image_filename = 'back1.jpeg' 
image_path = os.path.join('inputs', image_filename)
model_name = 'maternion/LightOnOCR-2'

# --- 2. Validation ---
if not os.path.exists(image_path):
    print(f"❌ Error: Could not find '{image_filename}' in the /inputs folder.")
    print(f"Current Directory: {os.getcwd()}")
else:
    print(f"🚀 Processing {image_filename}...")

    # --- 3. Execution ---
    try:
        response = ollama.generate(
            model=model_name,
            prompt='Transcribe the text in this image exactly as it appears.',
            images=[image_path]
        )

        # --- 4. Output ---
        print("\n--- EXTRACTED CONTENT ---")
        print(response['response'])
        
    except Exception as e:
        print(f"⚠️ An error occurred: {e}")