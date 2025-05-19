from diarizers import SegmentationModel
from pyannote.audio import Pipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

#segmentation_model = SegmentationModel().from_pretrained('diarizers-community/speaker-segmentation-fine-tuned-callhome-deu')

# load the pre-trained pyannote pipeline
pipeline = Pipeline.from_pretrained("diarizers-community/speaker-segmentation-fine-tuned-callhome-deu")
pipeline.to(device)

# replace the segmentation model with your fine-tuned one
model = segmentation_model.to_pyannote_model()
pipeline._segmentation.model = model.to(device)