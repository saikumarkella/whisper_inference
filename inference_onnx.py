"""
    Inference Whisper ONNX Models.
"""

""" Inference on the ONNX Runtime whisper Models
"""
# importing the modules
import numpy as np
import onnxruntime
import torch
from whisper_tokenizer import get_tokenizer
import whisper
from audio import log_mel_spectrogram

class WhisperONNXInference:

    def __init__(self, encoder_onnx_path="encoder.onnx", decoder_onnx_path="decoder.onnx"):
        self.sess_encoder = onnxruntime.InferenceSession(encoder_onnx_path)
        self.sess_decoder = onnxruntime.InferenceSession(decoder_onnx_path)
       
        self.max_tokens = 448
        self.tokenizer = whisper.decoding.get_tokenizer(True, language = "zh", task= "transcribe")
        self.start_tokens = list(self.tokenizer.sot_sequence_including_notimestamps)


    def inference(self, x_mel):
        out_encoder, = self.sess_encoder.run(["out"], {"x": x_mel.numpy()})
        tokens = self.start_tokens
        next_token = self.tokenizer.sot

        while len(tokens) <= self.max_tokens and next_token != self.tokenizer.eot:
            out_decoder = self.sess_decoder.run(
                ["out"],
                {
                    "tokens": np.array([tokens], dtype="int64"),
                    "audio": out_encoder,
                }
            )
            next_token = out_decoder[0, -1].argmax()
            print(next_token)
            tokens.append(next_token)

        transcript = self.tokenizer.decode(tokens[0])
        return transcript
        


if __name__ == "__main__":
    onnx_inference = WhisperONNXInference()
    print(onnx_inference.tokenizer.sot_sequence_including_notimestamps)

    # setting audio path 
    audio_path = "/home/sai/Documents/OpenMic/testing TTS external APIs/audios/final.wav"
    spec = log_mel_spectrogram(audio_path)
    spec = spec.unsqueeze(0)
    print(spec.shape)
    
    text = onnx_inference.inference(spec)
    print(text)



    

        



    
