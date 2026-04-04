import asyncio
import os
import random
import edge_tts
import librosa
import soundfile as sf
import pedalboard
from pedalboard import Pedalboard, PitchShift, Distortion, LowpassFilter, Chorus, PeakFilter
import pandas as pd
import numpy as np

SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the hooked line.",
    "Press the pants and sew a button on the vest.",
    "The swan dive was far short of perfect.",
    "The beauty of the view stunned the young boy.",
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The colt reared and threw the tall rider.",
    "It snowed, rained, and hailed the same morning.",
    "Read verse out loud for pleasure.",
    "Hoist the load to your left shoulder.",
    "Take the winding path to reach the lake.",
    "Note closely the size of the gas tank.",
    "Wipe the grease off his dirty face.",
    "Mend the coat before you go out.",
    "The wrist was badly strained and hung limp.",
    "The stray cat gave birth to kittens.",
    "The young girl gave no clear response.",
    "The meal was cooked before the bell rang.",
    "What joy there is in a clear sky.",
    "A king ruled the state in the early days.",
    "The ship was torn apart on the sharp reef.",
    "Sickness kept him home the third week.",
    "The wide road shimmered in the hot sun.",
    "The lazy cow lay in the cool grass.",
    "Lift the square stone over the fence.",
    "The rope will bind the seven books at once.",
    "Hop over the fence and plunge in.",
    "The friendly gang left the drug store.",
    "Mesh wire keeps chicks inside.",
    "The frosty air passed through the coat.",
    "The crooked maze failed to fool the mouse.",
    "Adding fast leads to wrong sums.",
    "The show was a flop from the very start.",
    "A saw is a tool used for making boards.",
    "The wagon moved on well oiled wheels.",
    "March the soldiers past the next hill.",
    "A cup of sugar makes sweet fudge.",
    "Place a rose on the dark grave.",
    "Play catch with the dog and stick.",
    "The spot on the blotter was made by green ink.",
    "Mud was spattered on the front of his white shirt.",
    "The cigar burned a hole in the desk top.",
    "The empty flask stood on the tin tray.",
    "A speedy man can beat this track mark.",
    "He broke a new shoelace that day.",
    "The coffee stand is too high for the couch.",
    "The urge to write short stories is rare.",
    "The fingers form loops as they bend.",
    "The dot on the i was marked with an x.",
    "Slide the catch back and open the desk.",
    "Help the weak to preserve their strength.",
    "A map had crosses where they found gold.",
    "The simple trick did not fool the boy.",
    "We tried to replace the coin but failed.",
    "She wore warm fleecy woolen overalls.",
    "The cut on his collar bone needed a bandage.",
    "The slant of the sun marks the time of day.",
    "The stamp stuck to the wet envelope.",
    "The young kid jumped the rusty gate.",
    "Guess the results from the first scores.",
    "A salt drop fell from the black cloud.",
    "The child crawled into the dense bramble.",
    "The chill wind made the child shiver.",
    "The brown dog was caught in the mud.",
    "The store walls were lined with colored glass.",
    "A thin stripe runs down the middle.",
    "The pipe began to rust while new.",
    "The frost made a clear mark on the window.",
    "The blind man counted his old coins.",
    "The clock struck to mark the third period.",
    "A strong grip is needed to turn the wheel.",
    "He wrote his name boldly at the top of the sheet.",
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Sphinx of black quartz, judge my vow.",
    "Two driven jocks help fax my big quiz.",
    "The five boxing wizards jump quickly.",
    "Jackdaws love my big sphinx of quartz."
]

OUTPUT_DIR = "synthetic_dataset_female"
TEMP_FILE = "temp_clean.wav"
# VOICE = "en-US-GuyNeural"  
VOICE = "en-US-JennyNeural" 

def create_dysarthric_audio(input_file, output_file):
    """
    Apply DSP effects to simulate dysarthria:
    1. Time stretch (slower speech, typical in dysarthria)
    2. Pitch shift and slight chorus (tremor/monopitch simulation)
    3. Distortion/filtering (imprecise consonants/breathiness)
    """
    y, sr = librosa.load(input_file, sr=24000)
    
    # 1. Time Stretch (slower, slurred articulation)
    rate = random.uniform(0.6, 0.75)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    
    # 2. Add some random momentary pauses or slight stutters (to simulate irregular rhythm)
    
    # 3. Apply Pedalboard Effects
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

async def generate_sample(text, sample_id):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(TEMP_FILE)
    output_filename = f"{sample_id:04d}_dysarthric.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    create_dysarthric_audio(TEMP_FILE, output_path)
    return output_filename

async def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Starting generation of {len(SENTENCES)} synthetic dysarthric samples...")
    
    metadata = []
    
    for i, text in enumerate(SENTENCES):
        sample_id = i + 1
        output_filename = await generate_sample(text, sample_id)
        
        metadata.append({
            "wav_filename": output_filename,
            "transcript": text
        })
        
        if sample_id % 10 == 0:
            print(f"Generated {sample_id}/{len(SENTENCES)} samples...")
            
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
        
    print(f"Dataset generation complete! Files saved to {OUTPUT_DIR}")
    print(f"Metadata saved to {os.path.join(OUTPUT_DIR, 'metadata.csv')}")

if __name__ == "__main__":
    asyncio.run(main())