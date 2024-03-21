import pretty_midi
import re
import numpy as np
import json 
import torch

base_tones = {
    'C' : 0, 'C#': 1, 'D' : 2, 'D#': 3,
    'E' : 4, 'F' : 5, 'F#': 6, 'G' : 7,
    
    'G#': 8, 'A' : 9, 'A#':10, 'B' :11,
}
line_index = {
    0: 'first', 1 : 'second', 2: 'third',
    3 : 'fourth', 4 : 'fifth', 
    5: 'sixth', 6 : 'seventh',
    7: 'eighth', 8 : 'ninth', 9: 'tenth',
}


def log_discretize(x, bins=512):
    eps = 1
    x_min = np.log(eps-0.3)
    x_max = np.log(6+eps)
    x = min(6, x)
    x = max(-0.3, x)
    x = np.log(x+eps)
    x = (x-x_min) / (x_max-x_min) * (bins-1)
    return np.round(x).astype(int)

def reverse_log_float(x, bins=512):
    if x == 79:
        return 0
    eps = 1
    x_min = np.log(eps-0.3)
    x_max = np.log(6+eps)
    x = x * (x_max - x_min)/(bins-1) + x_min
    x = np.exp(x) - eps
    return float("{:.3f}".format(x))

def bin_time(list_d):
    bin_list = []
    for item in list_d:
        if not isinstance(item, str):
            item = str(item)
        item_tuple = item.split(' ')
        out = ''
        for item_str in item_tuple:
            item_num = float(item_str)
            # out += f'<{item_num}>'
            bin = log_discretize(item_num)
            out += f'<{bin}>'
        bin_list.append(out)
    return bin_list

def append_song_token(model, tokenizer, config):
    old_token_len = len(tokenizer)
    new_tokens = ['<bol>','<bom>','<bop>','<eol>','<eom>','<eop>']
    for note in base_tones:
        for i in range(-1, 10): # -1 -> 9
            new_tokens.append(f'<{note}{i}>') 
    for t_bin in range(512):
        new_tokens.append(f'<{t_bin}>')
    new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
    new_tokens = list(new_tokens)
    new_tokens.sort()
    tokenizer.add_tokens(new_tokens)
    new_token_len = len(tokenizer)
    model.tokenizer = tokenizer

    weight = nn.Parameter(torch.empty((new_token_len-old_token_len, config.hidden_size)))
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    model.config.vocab_size = new_token_len
    model.output.weight.data = torch.cat([model.output.weight, weight.to(model.device)], dim=0)
    model.output.weight.requires_grad = True

    new_token_embed = torch.randn(new_token_len-old_token_len, config.hidden_size)
    new_weight = torch.cat([model.model.tok_embeddings.weight, new_token_embed.to(model.device)], dim=0)
    model.model.vocab_size = new_token_len
    model.model.tok_embeddings.weight.data = new_weight
    model.model.tok_embeddings.weight.requires_grad = True
    return model, tokenizer


def tuple2dict(line):
    order_string = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
    line = line.replace(" ", "")
    line = line.replace("\n", "")
    line = re.sub(r'\. |\.', '', line)
    # line = re.sub(r'The\d+line:', ' |', line)
    for string in order_string:
        line = line.replace(f'The{string}line:', ' |')
    special_pattern = r'<(.*?)>'
    song = {'lyrics':[], 'notes':[], 'notes_duration':[], 'rest_duration':[], 'pitch':[], 'notes_dict': [], 'rest_dict': []}
     
    for item in line.split('|')[1:]:
        x = item.split(',')
        notes = re.findall(special_pattern,x[1])
        note_ds = re.findall(special_pattern,x[2])
        rest_d = re.findall(special_pattern,x[3])[0]
        assert len(notes)== len(note_ds), f"notes:{'|'.join(notes)}, note_ds:{'|'.join(note_ds)}"
        for i in range(len(notes)):
            if i == 0:
                song['lyrics'].append(x[0])
            else:
                song['lyrics'].append('-')
            song['notes'].append(notes[i])
            song['pitch'].append(int(pretty_midi.note_name_to_number(notes[i])))
            song['notes_duration'].append(reverse_log_float(int(note_ds[i])))
            song['notes_dict'].append(int(note_ds[i]))
            if i == len(notes)-1:
                song['rest_duration'].append(reverse_log_float(int(rest_d)))
                song['rest_dict'].append(int(rest_d))
            else:
                song['rest_duration'].append(0)
                song['rest_dict'].append(0)
    return song

def dict2midi(song):
    # new_midi = pretty_midi.PrettyMIDI(charset="utf-8")#
    new_midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    # print(len(song["notes"]))
    current_time = 0  # Time since the beginning of the song, in seconds
    pitch = []
    for i in range(0, len(song["notes"])):
        #add notes
        notes_duration = song["notes_duration"][i]
        note_obj = pretty_midi.Note(velocity=100, pitch=int(pretty_midi.note_name_to_number(song["notes"][i])), start=current_time,
                                end=current_time + notes_duration)
        instrument.notes.append(note_obj)
        #add lyrics
        # lyric_event = pretty_midi.Lyric(text=str(song["lyrics"][i])+ "\0", time=current_time)
        # new_midi.lyrics.append(lyric_event)
        current_time +=  notes_duration + song["rest_duration"][i]# Update of the time
   
    new_midi.instruments.append(instrument)
    lyrics = ' '.join(song["lyrics"])
    return new_midi, lyrics


def gen_midi(line, file_name):
    song  = tuple2dict(line)
    #song['lyrics'] = ['I','-','you','-','I','-','you','-','I','-','you','-','he','-']
    new_midi, lyrics = dict2midi(song)
    
    # save midi file and lyric text
    new_midi.write(file_name+'.mid')
    
    with open(file_name+'.txt', "w") as file:
        file.write(lyrics)
    print(f'midi saved at ~/{file_name}.mid, lyrics saved at ~/{file_name}.txt')