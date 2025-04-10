#!/bin/bash
nohup jupyter notebook --ip=0.0.0.0 --port=9000 --no-browser --allow-root --notebook-dir=~/ > /var/log/jupyter.log 2>&1 &
echo "Jupyter Notebook started in background on port 9000"

# Start the Coqui TTS API server
cd ~/TTS
nohup python coqui-api.py > /var/log/coqui-api.log 2>&1 &
echo "Coqui TTS API server started in background on port 9002"