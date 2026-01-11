README.md
VAE Music Clustering
This project clusters music using neural networks called VAEs. We built three versions: basic VAE, convolutional VAE, and conditional VAE.
Installation
bashpip install -r requirements.txt
Run the code
bashpython src/vae.py
python src/conv_vae.py
python src/cvae.py
python src/hard_task_evaluation.py
Results
CVAE was the best with 51% Silhouette Score (vs 39% for the old method).
Files

src/ - Python code
results/ - CSV metrics and PNG plots
report/ - Research paper

Author
Nahiyan Tabassum, Cse425 Neural Networks Course, January 2026