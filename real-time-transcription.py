import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import webrtcvad
import logging
import glob
from pythonosc.udp_client import SimpleUDPClient

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser()
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
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    parser.add_argument("--osc_ip", default="127.0.0.1", help="The IP address to send OSC messages to.")
    parser.add_argument("--osc_port", default=9000, type=int, help="The port to send OSC messages to.")
    args = parser.parse_args()

    osc_client = SimpleUDPClient(args.osc_ip, args.osc_port)

    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.initial_energy_threshold
    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    logging.info(f"Loading Whisper model: {model}")
    try:
        audio_model = whisper.load_model(model)
        logging.info("Whisper model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        return

    record_timeout = args.initial_record_timeout
    phrase_timeout = args.initial_phrase_timeout

    transcription = []
    current_phrase = ""

    vad = webrtcvad.Vad(3)

    logging.info("Adjusting for ambient noise. Please wait...")
    with source:
        recorder.adjust_for_ambient_noise(source)
    logging.info("Ambient noise adjustment complete")

    def record_callback(_, audio: sr.AudioData) -> None:
        data_queue.put(audio.get_raw_data())

    logging.info("Starting background listening")
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Determine the next available transcription file name
    existing_files = glob.glob("transcription*.txt")
    idx = len(existing_files)
    transcription_file_name = f"transcription_{idx:02d}.txt"

    def is_speech(audio_chunk):
        try:
            return vad.is_speech(audio_chunk, 16000)
        except Exception as e:
            logging.error(f"Error in VAD: {e}")
            return False

    def print_transcription():
        print("\n" + "="*40 + "\nTranscription:")
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
                    logging.error(f"UnicodeEncodeError: {e}")
                
                # Print transcription
                print_transcription()
                
                current_phrase = ""
            phrase_time = None

    logging.info("Model loaded. Ready to transcribe.")

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
        logging.info("Transcription stopped by user")
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
                logging.error(f"UnicodeEncodeError: {e}")
            
            # Print final transcription
            print("\n\nFinal Transcription:")
            for line in transcription:
                print(line)
            
            osc_client.send_message("/transcription", current_phrase.strip())
            osc_client.send_message("/trigger", 1)

        logging.info(f"Transcription complete. You can now run offline NLP analysis on the {transcription_file_name} file.")

if __name__ == "__main__":
    main()