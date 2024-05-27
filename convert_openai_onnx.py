import whisper
import torch
import argparse

PATH_TO_MODEL = "whisper-tiny-openai.pt"



def load_whisper_model(path, lang="en", task="transcribe"):
    """
        Loading the openai whisper model and tokenizer

        Args:
            path (str) : A path to the whisper model
            lang (str) : Target language 
            task (str) : target task ( transcribe )
    """
    # loading the whisper model
    model = whisper.load_model(path)
    model.requires_grad_(False)
    model.eval()

    return model



def export_encoder_to_onnx(model, encoder_input, decoder_input, encoder_onnx_path= "encoder.onnx", decoder_onnx_path = "decoder.onnx"):
    """ Export Encoder to the Onnx Model 

        Args:
            model : whisper model which was loaded using the 
            output_onnx_path (str) : encoder onnx path to the model

    """
    # onnx encoder exporting model
    torch.onnx.export(
        model.encoder,
        (encoder_input, ),
        f"{encoder_onnx_path}",
        input_names=["x"],
        output_names=["out"],
        dynamic_axes={
            "x": {0: "batch"},
            "out": {0: "batch"},
        },
    )

    # onnx decoder exporting model
    torch.onnx.export(
        model.decoder,
        (decoder_input[0], decoder_input[1]),
        f"{decoder_onnx_path}",
        input_names=["tokens", "audio"],
        output_names=["out"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq"},
            "audio" : {0: "batch"},
            "out": {0: "batch", 1: "seq"}
        }


    )
    print("Exporting Done !!!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="a path to the open ai model",
        default="openai_models/whisper.pt"
    )

    parser.add_argument(
        "--encoder_path",
        type=str,
        help="A output path to the encoder onnx file path",
        default="onnx_models/encoder.onnx"
    )

    parser.add_argument(
        "--decoder_path",
        type=str,
        help="A output path to the decoder onnx file path",
        default="onnx_models/decoder.onnx"
    )

    args = parser.parse_args()
    model = load_whisper_model(args.model)
    
    # convert the onnx model
    encoder_input = torch.randn(1, 80, 3000)
    print(encoder_input.shape)
    x_audio = torch.randn(1, 1500, 384)
    print(x_audio.shape)
    decoder_input = (torch.randint(low=0, high=10, size=(1, 7)), x_audio)


    # exporting openai models into the onnx models.
    export_encoder_to_onnx(
        model=model,
        encoder_input=encoder_input,
        decoder_input=decoder_input,
        encoder_onnx_path=args.encoder_path,
        decoder_onnx_path=args.decoder_path
    )

