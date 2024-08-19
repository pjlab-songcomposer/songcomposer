import os
import pretty_midi
import json
import glob

def process_midi_file(midi_path, output_dir):
    try:
        # Load the MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        for i, instrument in enumerate(pm.instruments):
            melody_notes = instrument.notes
            # Check if the instrument is vocal or melody
            if instrument.name.lower() in ["vocal", "melody", "vocals"]:
                # Calculate the duration of each note and the rest duration between notes
                notes_duration = [round(note.end - note.start, 3) for note in melody_notes]
                rest_duration = [round(melody_notes[i].start - melody_notes[i-1].end, 3) for i in range(1, len(melody_notes))]
                rest_duration.append(0.0)  # The last note does not have a rest duration

                # Create a dictionary to store the extracted information
                disc_data = {
                    "best_instrument_id": i,
                    "best_instrument_name": instrument.name,
                    "tempo": pm.estimate_tempo(),
                    "notes": [pretty_midi.note_number_to_name(note.pitch) for note in melody_notes],
                    "notes_duration": notes_duration,
                    "rest_duration": rest_duration
                }

                # Save the data as a JSON file
                output_path = os.path.join(output_dir, f'{os.path.basename(midi_path)[:-4]}_{i}.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(disc_data, f, ensure_ascii=False)
                    print(f'{output_path} save succeeded')

    except Exception as e:
        print(f"Error processing {midi_path}: {e}")

def process_midi_files_in_directory(midi_dir, output_dir):
    # Get a list of all MIDI files in the directory
    midi_files = glob.glob(os.path.join(midi_dir, '*.mid'))
    for midi_file in midi_files:
        # Process each MIDI file
        process_midi_file(midi_file, output_dir)

# Example usage
midi_dir = 'xxxx/midi_pure'
output_dir = 'xxxx/melody_pool'

# Process all MIDI files in the specified directory
process_midi_files_in_directory(midi_dir, output_dir)
