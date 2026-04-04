import os
import random
import glob
import librosa
import soundfile as sf
import pedalboard
from pedalboard import Pedalboard, PitchShift, Distortion, LowpassFilter, Chorus, PeakFilter
import numpy as np

def create_dysarthric_audio(input_file, output_file):
    """
    Apply heavy DSP effects to augment existing samples:
    1. Time stretch (slower speech, typical in dysarthria)
    2. Pitch shift and slight chorus (tremor/monopitch simulation)
    3. Distortion/filtering (imprecise consonants/breathiness)
    """
    try:
        # Load audio
        y, sr = librosa.load(input_file, sr=24000)
        
        # 1. Time Stretch (slower, slurred articulation)
        rate = random.uniform(0.6, 0.75)
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        
        # 2. Apply Pedalboard Effects
        board = Pedalboard([
            PitchShift(semitones=-random.uniform(0.5, 2.0)),
            Chorus(rate_hz=random.uniform(2.0, 5.0), depth=random.uniform(0.1, 0.3), centre_delay_ms=7.0),
            Distortion(drive_db=random.uniform(1.0, 5.0)),
            LowpassFilter(cutoff_frequency_hz=random.uniform(3000, 5000)),
            PeakFilter(cutoff_frequency_hz=500, gain_db=random.uniform(2.0, 5.0), q=1.0)
        ])
        
        y_effected = board(y_stretched, sr)
        
        max_amp = np.max(np.abs(y_effected))
        if max_amp > 0:
            y_effected = y_effected / max_amp * 0.9

        sf.write(output_file, y_effected, sr)
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def augment_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} .wav files to augment.")
    print("Starting augmentation process...")
    
    success_count = 0
    for i, file_path in enumerate(wav_files):
        rel_path = os.path.relpath(file_path, input_dir)
        output_file_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        name, ext = os.path.splitext(output_file_path)
        if not name.endswith("_augmented"):
            output_file_path = f"{name}_augmented{ext}"
            
        if create_dysarthric_audio(file_path, output_file_path):
            success_count += 1
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(wav_files)} files...")
            
    print(f"\nAugmentation complete! Successfully augmented {success_count}/{len(wav_files)} files.")
    print(f"Augmented files are saved in: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment audio dataset to simulate dysarthria.")
    parser.add_argument("--input", "-i", type=str, default="synthetic_speech", help="Input directory containing .wav files")
    parser.add_argument("--output", "-o", type=str, default="augmented_dysarthric_speech", help="Output directory to save augmented files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input directory '{args.input}' does not exist! Please provide a valid input path.")
    else:
        augment_directory(args.input, args.output)
