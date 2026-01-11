import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import webrtcvad
import logging
import glob
import multiprocessing
from pythonosc.udp_client import SimpleUDPClient

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def list_microphones():
    """List all available microphones."""
    print("Available microphone devices are: ")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{index}] \"{name}\"")


def get_microphone_by_name_or_index(mic_identifier):
    """
    Get a microphone source by name (partial match) or index.
    If mic_identifier is None, uses the system default microphone.
    Returns (source, device_index, device_name) or (None, None, None) if not found.
    """
    mic_names = sr.Microphone.list_microphone_names()
    
    # Handle None/empty - use system default
    if mic_identifier is None or mic_identifier == "":
        return sr.Microphone(sample_rate=16000), None, "System Default"
    
    # Try to parse as an integer index first
    try:
        device_index = int(mic_identifier)
        if 0 <= device_index < len(mic_names):
            return sr.Microphone(sample_rate=16000, device_index=device_index), device_index, mic_names[device_index]
        else:
            logging.error(f"Microphone index {device_index} out of range (0-{len(mic_names)-1})")
            return None, None, None
    except ValueError:
        pass
    
    # Try to match by name (partial match)
    matches = [(index, name) for index, name in enumerate(mic_names) if mic_identifier in name]
    
    if len(matches) == 0:
        logging.error(f"Microphone '{mic_identifier}' not found")
        return None, None, None
    
    if len(matches) > 1:
        logging.warning(f"Multiple microphones match '{mic_identifier}': {[m[1] for m in matches]}. Using first match.")
    
    index, name = matches[0]
    return sr.Microphone(sample_rate=16000, device_index=index), index, name


def transcribe_microphone(mic_identifier, model_name, energy_threshold, record_timeout, 
                          phrase_timeout, osc_ip, osc_port, transcription_file_name, mic_label):
    """
    Transcription worker function for a single microphone.
    This function is designed to run in a separate process.
    """
    # Configure logging for this process with a unique format
    # Use force=True to reconfigure even if already configured in parent process
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f'%(asctime)s - [{mic_label}] %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    osc_client = SimpleUDPClient(osc_ip, osc_port)
    
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False
    
    # Get the microphone source
    source, device_index, device_name = get_microphone_by_name_or_index(mic_identifier)
    if source is None:
        logging.error(f"[{mic_label}] Failed to initialize microphone: {mic_identifier}")
        return
    
    logging.info(f"[{mic_label}] Using microphone: {device_name} (index {device_index})")
    
    # Load the Whisper model
    logging.info(f"[{mic_label}] Loading Whisper model: {model_name}")
    try:
        audio_model = whisper.load_model(model_name)
        logging.info(f"[{mic_label}] Whisper model loaded successfully")
    except Exception as e:
        logging.error(f"[{mic_label}] Failed to load Whisper model: {e}")
        return
    
    transcription = []
    current_phrase = ""
    
    vad = webrtcvad.Vad(3)
    
    logging.info(f"[{mic_label}] Adjusting for ambient noise. Please wait...")
    with source:
        recorder.adjust_for_ambient_noise(source)
    logging.info(f"[{mic_label}] Ambient noise adjustment complete")
    
    def record_callback(_, audio: sr.AudioData) -> None:
        data_queue.put(audio.get_raw_data())
    
    logging.info(f"[{mic_label}] Starting background listening")
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    
    def is_speech(audio_chunk):
        try:
            return vad.is_speech(audio_chunk, 16000)
        except Exception as e:
            logging.error(f"[{mic_label}] Error in VAD: {e}")
            return False
    
    def print_transcription():
        print(f"\n[{mic_label}] " + "="*40 + "\nTranscription:")
        for line in transcription:
            print(line)
        if current_phrase:
            print(current_phrase, end='', flush=True)
        print("\n" + "="*40 + "\n")
    
    def handle_silence(now):
        nonlocal phrase_time, current_phrase, transcription
        
        if phrase_time and (now - phrase_time > timedelta(seconds=phrase_timeout)):
            if current_phrase.strip():
                transcription.append(current_phrase.strip())
                osc_client.send_message("/trigger", 1)
                
                # Write to file
                try:
                    with open(transcription_file_name, "a", encoding="utf-8") as file:
                        file.write(current_phrase.strip() + "\n")
                        file.flush()
                except UnicodeEncodeError as e:
                    logging.error(f"[{mic_label}] UnicodeEncodeError: {e}")
                
                # Print transcription
                print_transcription()
                
                current_phrase = ""
            phrase_time = None
    
    logging.info(f"[{mic_label}] Model loaded. Ready to transcribe. Output file: {transcription_file_name}")
    
    try:
        with open(transcription_file_name, "w", encoding="utf-8") as file:
            while True:
                now = datetime.utcnow()
                if not data_queue.empty():
                    phrase_complete = False
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                    
                    phrase_time = now
                    
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()
                    
                    frame_duration = 30
                    n = int(16000 * (frame_duration / 1000) * 2)
                    speech_count = sum(1 for i in range(0, len(audio_data), n) if is_speech(audio_data[i:i + n]))
                    
                    if speech_count > 0:
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                        text = result['text'].strip()
                        
                        if phrase_complete:
                            if current_phrase.strip():
                                transcription.append(current_phrase.strip())
                                
                                # Write to file
                                file.write(current_phrase.strip() + "\n")
                                file.flush()
                                
                                # Print transcription
                                print_transcription()
                                
                                osc_client.send_message("/trigger", 1)
                            current_phrase = text
                        else:
                            current_phrase += " " + text
                            osc_client.send_message("/trigger", 0)
                        
                        osc_client.send_message("/transcription", current_phrase.strip())
                    else:
                        handle_silence(now)
                    
                    sleep(0.1)
                else:
                    handle_silence(datetime.utcnow())
                    sleep(0.25)
    except KeyboardInterrupt:
        logging.info(f"[{mic_label}] Transcription stopped by user")
        stop_listening(wait_for_stop=False)
        
        # Add the last phrase to transcription
        if current_phrase.strip():
            transcription.append(current_phrase.strip())
            
            # Write to file
            try:
                with open(transcription_file_name, "a", encoding="utf-8") as file:
                    file.write(current_phrase.strip() + "\n")
                    file.flush()
            except UnicodeEncodeError as e:
                logging.error(f"[{mic_label}] UnicodeEncodeError: {e}")
            
            # Print final transcription
            print(f"\n\n[{mic_label}] Final Transcription:")
            for line in transcription:
                print(line)
            
            osc_client.send_message("/transcription", current_phrase.strip())
            osc_client.send_message("/trigger", 1)
        
        logging.info(f"[{mic_label}] Transcription complete. Output saved to {transcription_file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time transcription using Whisper. Supports multiple microphones.")
    parser.add_argument("--model", default="turbo", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the English model.")
    parser.add_argument("--initial_energy_threshold", default=1000,
                        help="Initial energy level for mic to detect.", type=int)
    parser.add_argument("--initial_record_timeout", default=2,
                        help="Initial timeout for real-time recording in seconds.", type=float)
    parser.add_argument("--initial_phrase_timeout", default=3,
                        help="Initial timeout for considering a new line in transcription.", type=float)
    parser.add_argument("--list_microphones", action='store_true',
                        help="List all available microphones and exit.")
    parser.add_argument("--microphones", nargs='+', default=None,
                        help="List of microphone names or indices to use. Each microphone runs in a separate process. "
                             "Examples: --microphones 0 2 or --microphones 'USB' 'Built-in'")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition (used when --microphones is not specified). "
                                 "Run with --list_microphones to view available microphones.", type=str)
    parser.add_argument("--osc_ip", default="127.0.0.1", help="The IP address to send OSC messages to.")
    parser.add_argument("--osc_port", default=9000, type=int, 
                        help="The base port to send OSC messages to. When using multiple microphones, "
                             "each mic uses an incremented port (e.g., 9000, 9001, 9002).")
    args = parser.parse_args()
    
    # Handle list microphones request
    if args.list_microphones:
        list_microphones()
        return
    
    # Handle legacy single microphone mode for backwards compatibility
    if args.microphones is None:
        # Single microphone mode (original behavior)
        if 'linux' in platform:
            mic_name = args.default_microphone
            if not mic_name or mic_name == 'list':
                list_microphones()
                return
        
        # Determine the next available transcription file name
        existing_files = glob.glob("transcription*.txt")
        idx = len(existing_files)
        transcription_file_name = f"transcription_{idx:02d}.txt"
        
        # Get mic identifier for single mode
        if 'linux' in platform:
            mic_identifier = args.default_microphone
        else:
            # Use system default microphone (None lets speech_recognition pick)
            mic_identifier = None
        
        transcribe_microphone(
            mic_identifier=mic_identifier,
            model_name=args.model,
            energy_threshold=args.initial_energy_threshold,
            record_timeout=args.initial_record_timeout,
            phrase_timeout=args.initial_phrase_timeout,
            osc_ip=args.osc_ip,
            osc_port=args.osc_port,
            transcription_file_name=transcription_file_name,
            mic_label="mic0"
        )
    else:
        # Multi-microphone mode
        processes = []
        
        # Determine the next available transcription file base index
        existing_files = glob.glob("transcription*.txt")
        base_idx = len(existing_files)
        
        logging.info(f"Starting transcription for {len(args.microphones)} microphone(s)")
        
        for i, mic_id in enumerate(args.microphones):
            transcription_file_name = f"transcription_{base_idx + i:02d}.txt"
            osc_port = args.osc_port + i
            mic_label = f"mic{i}"
            
            logging.info(f"Spawning process for microphone '{mic_id}' -> {transcription_file_name}, OSC port {osc_port}")
            
            p = multiprocessing.Process(
                target=transcribe_microphone,
                args=(
                    mic_id,
                    args.model,
                    args.initial_energy_threshold,
                    args.initial_record_timeout,
                    args.initial_phrase_timeout,
                    args.osc_ip,
                    osc_port,
                    transcription_file_name,
                    mic_label
                )
            )
            processes.append(p)
            p.start()
        
        logging.info(f"All {len(processes)} processes started. Press Ctrl+C to stop all.")
        
        try:
            # Wait for all processes
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logging.info("Stopping all transcription processes...")
            for p in processes:
                p.terminate()
                p.join()
            logging.info("All processes stopped.")

if __name__ == "__main__":
    main()