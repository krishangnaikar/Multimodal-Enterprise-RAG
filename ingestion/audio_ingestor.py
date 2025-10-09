from faster_whisper import WhisperModel

def transcribe_audio(path, model_size = "base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, beam_size=1)
    text = " ".join([seg.text for seg in segments])
    return text.strip()
