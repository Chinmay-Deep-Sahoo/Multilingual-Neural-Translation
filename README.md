# Multilingual-Neural-Translation

This repository contains code for a multilingual neural translation project leveraging state-of-the-art models for text and speech translation. The project utilizes the SeamlessM4Tv2 model from Facebook's Transformers library.
## Project Overview ##
The main components of this project include:
1. Pretrained Models Used:
    * I employed the SeamlessM4Tv2 model for translation tasks, which supports multilingual translation for both text and speech.
   
2. Supported Languages:
     * The model supports various languages for different translation tasks, including input from speech or text, and output to speech or text. The supported languages are defined dynamically based on the model's capabilities.
     * Supported languages: https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md
  
## Code Functions ##
The repository provides two main functions:
1. textInput(text, src_lang, tgt_lang):
   * This function translates input text into the specified target language.
   * Inputs:
     * text (str): Input text to be translated.
     * src_lang (str): Language of the input text.
     * tgt_lang (str): Target language for the translated output (text and/or speech).
   * Outputs:
     * audio_array (torch.Tensor or None): Translated audio data tensor (if the target language supports speech output).
     * translated_text (str or None): Translated text (if the target language supports text output).
2. audioInput(audio_path, tgt_lang):
    * This function translates an input audio file into the specified target language.
    * Inputs:
      * audio_path (str): Path to the audio file in WAV format.
      * tgt_lang (str): Target language for the translated output (text and/or speech).
    * Outputs:
      * audio_array (torch.Tensor or None): Translated audio data tensor (if the target language supports speech output).
      * translated_text (str or None): Translated text (if the target language supports text output).

## Requirements ##
To run this project, ensure you have the following installed:
1. transformers library (pip install transformers)
2. torchaudio library (pip install torchaudio)

### Note ###
This project demonstrates the implementation of a multilingual translation system using advanced neural network models. Please refer to the provided functions for more details on usage and capabilities.

For questions or issues, feel free to reach out!
