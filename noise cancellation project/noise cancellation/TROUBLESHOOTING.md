# NoiseClear - Audio Noise Reduction Application

## ⚡ Quick Start

### Option 1: Batch File (Easiest for Windows)
1. Double-click `RUN_SERVER.bat`
2. Wait for the server to start (you'll see `Uvicorn running on http://0.0.0.0:8000`)
3. Open http://localhost:8000 in your browser
4. Upload an audio file and click "Remove Noise"

### Option 2: PowerShell
```powershell
.\RUN_SERVER.ps1
```

### Option 3: Manual Terminal
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## ❌ Troubleshooting "Failed to Fetch" Error

### 1. **Server Not Running**
   - **Problem**: You see "Failed to fetch" or "Cannot connect to backend"
   - **Solution**: 
     - Make sure you've started the server using one of the methods above
     - Check that you see `Uvicorn running on http://0.0.0.0:8000` in the terminal

### 2. **Backend Takes Too Long**
   - **Problem**: Processing times out after 10 minutes
   - **Solution**: 
     - Try with a smaller audio file first
     - Use "Standard" or "Mild" noise reduction strength (Aggressive takes longer)
     - Make sure you have enough RAM available

### 3. **Port 8000 Already in Use**
   - **Problem**: Error "Address already in use"
   - **Solution**:
     - Close any other instances of the application
     - Or change port in startup script: `--port 8001`

### 4. **Python/Dependencies Issues**
   - **Problem**: Module not found, import errors
   - **Solution**:
     ```bash
     .venv\Scripts\activate
     pip install -r requirements.txt --force-reinstall
     ```

### 5. **Audio File Not Supported**
   - **Problem**: "Unsupported file type" error
   - **Supported formats**: MP3, WAV, OGG, FLAC, M4A
   - **Solution**: Convert your file to one of these formats first

## 🎯 Features

- **Voice Activity Detection (VAD)**: Detects speech patterns to preserve voice quality
- **Spectral Subtraction**: Advanced frequency-domain noise reduction
- **Wiener Filtering**: Adaptive filtering for refinement
- **Multiple Strength Levels**:
  - **Mild** (0.5): Gentle noise removal, preserves more original audio
  - **Standard** (1.0): Balanced noise removal (default)
  - **Aggressive** (1.5): Heavy noise removal, takes longer to process

## 📊 Analysis Graphs

The application generates three comparison graphs:
1. **Time Domain**: Original vs. denoised waveforms
2. **Frequency Domain**: FFT spectra comparison
3. **Spectrogram**: Time-frequency analysis of both versions

## 🔧 Technical Details

- **Backend**: FastAPI with Python audio processing
- **Audio Processing**: librosa, soundfile, noisereduce
- **Frontend**: HTML/CSS/JS with real-time audio player
- **Deployment**: Runs 100% locally (no data sent to cloud)

## 📝 Notes

- Processing time depends on audio length and system specs (usually 5-30 seconds)
- Large files (>100MB) may take longer or require more RAM
- The application uses non-interactive matplotlib backend for graph generation
- All files are processed in temporary directories and cleaned up automatically

## ✅ Verified Working With

- Python 3.8+
- Windows 10/11
- MP3, WAV, OGG, FLAC, M4A audio formats
- Audio files up to several gigabytes (depending on RAM)
