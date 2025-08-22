from scripts.processor_v2 import OptimizedGPUProcessor
from pathlib import Path


processor = OptimizedGPUProcessor(force_cpu=False)  # Use GPU
processor.process_dataset_batched(
     Path("/Users/pedro/Library/CloudStorage/GoogleDrive-pedro@antfood.com/My Drive/Machine Learning/data/lou_vo"),
     Path("processed/"),
     batch_size=15  # Adjust based on your T4 performance
)
