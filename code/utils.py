from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import pickle
import warnings

#Load the required pretrained models for translation
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
# model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
# processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")

sample_rate = model.config.sampling_rate    #16 kHz

#Load supported languages for different tasks
with open("langs.pkl", 'rb') as pickle_file:
    langs = pickle.load(pickle_file)
source_speech, source_text, target_speech, target_text = langs
print(f"Supported languages for audio input: {list(source_speech.keys())}")
print(f"Supported languages for text input: {list(source_text.keys())}")
print(f"Supported languages for audio output: {list(target_speech.keys())}")
print(f"Supported languages for text output: {list(target_text.keys())}")

def textInput(text : str, src_lang : str, tgt_lang : str):
    '''
    Translate input text into text and/or speech in the specified target language.

    Args:
        text (str): Input text to be translated.
        src_lang (str): Language of the input text.
        tgt_lang (str): Target language for the output text and/or speech.

    Returns:
        audio_array (torch.Tensor or None): Translated audio data tensor (if tgt_lang is supported for speech output).
        translated_text (str or None): Translated text (if tgt_lang is supported for text output).
    '''

    # Check if source language is supported for textual input
    if src_lang not in source_text:
        raise ValueError(f"'{src_lang}' not supported for textual input")
    
    x1, x2 = False, False
    # Check if target language is supported for speech output
    if tgt_lang not in target_speech:
        x1 = True
        warnings.warn(f"'{tgt_lang}' not supported for audio output")

    # Check if target language is supported for text output
    if tgt_lang not in target_text:
        x2 = True
        warnings.warn(f"'{tgt_lang}' not supported for text output")

    # If both audio and text outputs are not supported for tgt_lang, raise an error
    if x1 and x2:
        raise ValueError(f"'{tgt_lang}' not supported for any output type")

    text_inputs = processor(text = text, src_lang = source_text[src_lang], return_tensors="pt")

    # Generate translated text if tgt_lang is supported for text output
    if x2 == False:
        output_tokens = model.generate(**text_inputs, tgt_lang=target_text[tgt_lang], generate_speech=False)
        translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    else:
        translated_text = None
    
    # Generate audio if tgt_lang is supported for speech output
    if x1 == False:
        audio_array = model.generate(**text_inputs, tgt_lang=target_text[tgt_lang])[0].cpu().squeeze()
    else:
        audio_array = None

    return audio_array, translated_text

def audioInput(audio_path : str, tgt_lang : str):
    '''
    Translate an audio file into text and/or speech in the specified target language.

    Args:
        audio_path (str): Path to the audio file in WAV format.
        tgt_lang (str): Target language for the output text and/or speech.

    Returns:
        audio_array (torch.Tensor or None): Translated audio data tensor (if tgt_lang is supported for speech output).
        translated_text (str or None): Translated text (if tgt_lang is supported for text output).
    '''

    x1, x2 = False, False
    if tgt_lang not in target_speech:
        x1 = True
        warnings.warn(f"'{tgt_lang}' not supported for audio output")
    if tgt_lang not in target_text:
        x2 = True
        warnings.warn(f"'{tgt_lang}' not supported for text output")

    if x1 and x2:
        raise ValueError(f"'{tgt_lang}' not supported for any output type")
    
    # Load audio file and resample to 16 kHz
    audio, orig_freq =  torchaudio.load(audio_path)
    audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)
    audio_inputs = processor(audios=audio, return_tensors="pt", sampling_rate = sample_rate)

    # Generate speech if tgt_lang is supported for speech output
    if x1 == False:
        audio_array = model.generate(**audio_inputs, tgt_lang=target_text[tgt_lang])[0].cpu().squeeze()
    else:
        audio_array = None
    
    # Generate text if tgt_lang is supported for text output
    if x2 == False:
        output_tokens = model.generate(**audio_inputs, tgt_lang=target_text[tgt_lang], generate_speech=False)
        translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    else:
        translated_text = None

    return audio_array, translated_text