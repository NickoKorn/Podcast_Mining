
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os
import chromadb
from chromadb.utils import embedding_functions

class WhisperModell():

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")        
        #self.device = "cpu"
        self.torch_dtype = torch.float16
        #self.torch_dtype = torch.int
        model_id = "primeline/whisper-large-v3-turbo-german"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps="segment",
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        self.is_saved = 0
    
    def save_model(self, local_save_path="../../models/primeline/whisper-large-v3-turbo-german"):

        self.model.save_pretrained(local_save_path)
        self.processor.save_pretrained(local_save_path) # Wichtig: Auch den Processor speichern!

        print(f"Modell und Processor erfolgreich gespeichert unter: {local_save_path}") 

    def load_model(self, local_load_path="primeline/whisper-large-v3-turbo-german")->tuple:

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_load_path,
        torch_dtype = torch.float16, # Sicherstellen, dass es mit dem richtigen dtype geladen wird
        ).to(device)
        AutoProcessor =  AutoProcessor.from_pretrained(local_load_path)
        return model, AutoProcessor

    def transcripteText(self, audioPathes: str)->str:

        if self.is_saved == 0:

            self.save_model()
            self.is_saved = 1

        return self.pipe(audioPathes, return_timestamps=True) 
    
