#: Python inference
import time

import whisper
import torch

# allow to inject real data
if "load_audio_mfc" not in globals():
    load_audio_mfc = lambda: torch.randn(1, 80, 3000)

model = whisper.load_model("/home/sai/Documents/DRDO/ASR_ONNX_Experiment/whisper-tiny-openai.pt")
model.requires_grad_(False)
model.eval()

tokenizer = whisper.decoding.get_tokenizer(
    model.is_multilingual, 
    task="transcribe", 
    language="en",
)

# x_mel shape: [batch, coeff=80, time=3000]
x_mel = load_audio_mfc() 

# encode the audio
# x_audio shape: [batch, time=1500, feature=512]
start = time.time()
x_audio = model.encoder(x_mel)
    
# initialize using the start sequence
# x_tokens shape: [batch, seq<=448]
x_tokens = torch.tensor(
    [tokenizer.sot_sequence_including_notimestamps],
    dtype=torch.long,
)

print("getting the shape of the x_audio ")
print(x_audio.shape)
print("\n\n")
max_tokens = 448
next_token = tokenizer.sot

print("\n\nprinting the tokenizers :: ")
print(tokenizer.decode(tokenizer.sot_sequence_including_notimestamps))

while x_tokens.shape[1] <= max_tokens and next_token != tokenizer.eot:
    y_tokens = model.decoder(x_tokens, x_audio)

    next_token = y_tokens[0, -1].argmax()        
    x_tokens = torch.concat(
        [x_tokens, next_token.reshape(1, 1)], 
        axis=1,
    )


print("printing the tokens information")
print(x_tokens.shape)
print("\n\n")
print("took", time.time() - start, "seconds")
print(tokenizer.decode(x_tokens[0]))

#: ONNX export
torch.onnx.export(
    model.encoder, 
    (x_mel,), 
    "encoder.onnx", 
    input_names=["x"], 
    output_names=["out"],
    dynamic_axes={
        "x": {0: "batch"},
        "out": {0: "batch"},
    },
)

torch.onnx.export(
    model.decoder, 
    (x_tokens, x_audio), 
    "decoder.onnx", 
    input_names=["tokens", "audio"], 
    output_names=["out"], 
    dynamic_axes={
        "tokens": {0: "batch", 1: "seq"},
        "audio": {0: "batch"},
        "out": {0: "batch", 1: "seq"},
    },
)

#: Execute the ONNX model
import numpy as np
import onnxruntime

sess_encoder = onnxruntime.InferenceSession("encoder.onnx")
sess_decoder = onnxruntime.InferenceSession("decoder.onnx")

start = time.time()
out_encoder, = sess_encoder.run(["out"], {"x": x_mel.numpy()})

# initialize the tokens
tokens = list(tokenizer.sot_sequence_including_notimestamps)

next_token = tokenizer.sot
while x_tokens.shape[1] <= max_tokens and next_token != tokenizer.eot:
    out_decoder, = sess_decoder.run(
        ["out"], 
        {
            "tokens": np.asarray([tokens], dtype="int64"), 
            "audio": out_encoder,
        },
    )
    print("shape of decoder ", out_decoder.shape)
    next_token = out_decoder[0, -1].argmax()
    tokens.append(next_token)

print("took", time.time() - start, "seconds")
print(tokens)
print(tokenizer.decode(tokens))

#: PyTorch with kv-caching
start = time.time()
whisper.decode(
    model, 
    x_mel, 
    options=whisper.DecodingOptions(
        fp16=False, 
        without_timestamps=True, 
        suppress_blank=False, 
        suppress_tokens=[],
    ),
)
print(time.time() - start)
