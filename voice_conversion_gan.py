import os
import glob
import traceback

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: Coqui 'TTS' library is not installed. Voice conversion is powered by this deep learning library.")
    print("Please install via: pip install TTS")

INPUT_SOURCE_DIR = "synthetic_speech" 
REFERENCE_WAV = "dysarthric_reference.wav" 
OUTPUT_DIR = "gan_converted_speech"

def run_voice_conversion():
    if not TTS_AVAILABLE:
        print("\nCannot run Voice Conversion without the TTS library. Exiting.")
        return

    if not os.path.exists(REFERENCE_WAV):
        print(f"\n[!] ERROR: Missing reference audio '{REFERENCE_WAV}'!")
        print("    Voice Conversion requires a target sample to learn the tone and style.")
        print("    Please place a short dysarthric sample inside your working directory named 'dysarthric_reference.wav'.")
        # For demonstration purposes, we will not block here if we can find something to use
        # but in a real scenario, this is crucial.
        
    print("Loading FreeVC Voice Conversion Model (Downloads model on first run)...")
    # We use FreeVC24 (a state-of-the-art voice conversion GAN)
    try:
        tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True, gpu=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Grab source files to convert
    source_files = glob.glob(os.path.join(INPUT_SOURCE_DIR, "**", "*.wav"), recursive=True)
    if not source_files:
        print(f"No source .wav files found in {INPUT_SOURCE_DIR} to convert.")
        return

    print(f"Found {len(source_files)} files. Starting GAN conversion...")
    
    # We will pick one file as reference just to make testing work if the user didn't supply one
    active_reference = REFERENCE_WAV
    if not os.path.exists(REFERENCE_WAV):
        # Fallback to an augmented sample if available
        augmented_files = glob.glob(os.path.join("augmented_dysarthric_speech", "**", "*.wav"), recursive=True)
        if augmented_files:
            active_reference = augmented_files[0]
            print(f"-- Using {active_reference} as reference since {REFERENCE_WAV} was missing --")
        else:
            print("No reference file found at all. Cannot proceed.")
            return

    success_count = 0
    # Process only the first 5 as a demo since FreeVC inference is heavy on CPU
    for i, source_wav in enumerate(source_files[:5]):
        rel_path = os.path.relpath(source_wav, INPUT_SOURCE_DIR)
        output_file_path = os.path.join(OUTPUT_DIR, rel_path)
        
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        print(f"[{i+1}/5] Converting {os.path.basename(source_wav)} -> {os.path.basename(output_file_path)}")
        try:
            tts.voice_conversion_to_file(source_wav=source_wav, target_wav=active_reference, file_path=output_file_path)
            success_count += 1
        except Exception as e:
            print(f"Error converting {source_wav}: {e}")
            traceback.print_exc()

    print(f"\nGAN Voice Conversion completed for a demo subset of {success_count} files!")
    print(f"Files saved to: {OUTPUT_DIR}")
    print("To run on all 100-300 files, remove the '[:5]' limit in the python script.")

if __name__ == "__main__":
    print("-" * 50)
    print(" Voice Conversion (GAN) Dysarthria Transformer")
    print("-" * 50)
    run_voice_conversion()
