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
    parser.add_argument("--language", default=None,
                        help="Spoken language code (e.g. 'en', 'fr'). Skips auto-detection for better accuracy.")
    parser.add_argument("--initial_prompt", default=None,
                        help="Prompt to bias the model toward domain vocabulary, names, or spelling.")
    parser.add_argument("--vad_aggressiveness", default=2, type=int, choices=[0, 1, 2, 3],
                        help="webrtcvad aggressiveness (0=least, 3=most). Lower keeps more soft speech.")
    parser.add_argument("--initial_energy_threshold", default=1000,
                        help="Initial energy level for mic to detect.", type=int)
    parser.add_argument("--initial_record_timeout", default=1,
                        help="How often (seconds) live partial transcription is refreshed and sent over OSC.", type=float)
    parser.add_argument("--initial_phrase_timeout", default=3,
                        help="Longest silence (seconds) tolerated before an utterance is force-finalized.", type=float)
    # Adaptive utterance pacing: the silence needed to finalize an utterance is
    # learned from the speaker's own rhythm rather than fixed, so /trigger fires
    # on natural pauses. Image changes therefore track the cadence of speech.
    parser.add_argument("--min_pause", default=0.7, type=float,
                        help="Shortest silence (seconds) that can ever end an utterance.")
    parser.add_argument("--pause_margin", default=1.6, type=float,
                        help="Multiplier applied to the learned mid-utterance pause to decide an utterance has ended.")
    parser.add_argument("--min_utterance_duration", default=1.2, type=float,
                        help="Minimum spoken seconds before a finalized utterance is allowed to fire /trigger.")
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

    # English-only model variants (.en) are more accurate for English speech.
    # They only exist for tiny/base/small/medium; large and turbo are multilingual only.
    model = args.model
    if model not in ("large", "turbo") and not args.non_english:
        model += ".en"

    # Resolve the language passed to Whisper. English-only models must use "en";
    # otherwise honour an explicit --language and fall back to auto-detection.
    if model.endswith(".en"):
        language = "en"
    else:
        language = args.language

    logging.info(f"Loading Whisper model: {model}")
    try:
        audio_model = whisper.load_model(model)
        logging.info("Whisper model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        return

    record_timeout = args.initial_record_timeout
    max_pause = args.initial_phrase_timeout      # hard ceiling: force-finalize after this much silence
    min_pause = args.min_pause                   # floor: never finalize on a shorter gap
    pause_margin = args.pause_margin
    min_utterance_duration = args.min_utterance_duration

    transcription = []
    current_phrase = ""
    phrase_bytes = bytes()

    # Adaptive pacing state. `intra_pause` is an EWMA of the pauses the speaker
    # takes *within* an utterance; the finalize threshold is derived from it so a
    # fast talker's short gaps don't prematurely cut a thought, while a slow,
    # deliberate speaker gets longer windows. Seeded at the floor.
    intra_pause = min_pause

    def utterance_end_threshold():
        """Silence (seconds) required to consider the current utterance finished."""
        return max(min_pause, min(max_pause, intra_pause * pause_margin))

    def utterance_seconds():
        """Spoken duration currently buffered (16 kHz mono int16)."""
        return len(phrase_bytes) / 2 / 16000

    vad = webrtcvad.Vad(args.vad_aggressiveness)

    # Whisper frequently emits these on near-silence/background noise. Drop them
    # when they are the entire output so they don't pollute the transcript.
    hallucinations = {
        "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
        "you", "bye.", "bye", ".", "you're welcome.", "okay.", "so.",
    }

    def is_hallucination(text):
        return text.strip().lower() in hallucinations

    transcribe_options = dict(
        language=language,
        fp16=torch.cuda.is_available(),
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        initial_prompt=args.initial_prompt,
    )

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

    def finalize_utterance(force=False):
        """End the current utterance: emit it for the image pipeline and reset.

        Returns True if the utterance was emitted. When `force` is False a short
        utterance (below --min_utterance_duration) is held back so trivial
        fragments don't change the image; the held audio keeps accumulating into
        the next utterance. `force` (silence past the hard ceiling, or shutdown)
        always flushes whatever is buffered.
        """
        nonlocal phrase_time, current_phrase, transcription, phrase_bytes

        text = current_phrase.strip()
        if not text:
            phrase_bytes = bytes()
            phrase_time = None
            return False

        if not force and utterance_seconds() < min_utterance_duration:
            # Too little speech to warrant a new image yet — keep listening.
            return False

        transcription.append(text)

        # Write to file
        try:
            with open(transcription_file_name, "a", encoding="utf-8") as f:
                f.write(text + "\n")
                f.flush()
        except UnicodeEncodeError as e:
            logging.error(f"UnicodeEncodeError: {e}")

        print_transcription()

        # Final, paced signal for the LLM -> txt2img step.
        osc_client.send_message("/transcription", text)
        osc_client.send_message("/trigger", 1)

        current_phrase = ""
        phrase_bytes = bytes()
        phrase_time = None
        return True

    def handle_silence(now):
        if not phrase_time:
            return
        silence = (now - phrase_time).total_seconds()
        if silence >= utterance_end_threshold():
            finalize_utterance(force=silence >= max_pause)

    logging.info("Model loaded. Ready to transcribe.")

    try:
        with open(transcription_file_name, "w", encoding="utf-8") as file:
            while True:
                now = datetime.utcnow()
                if not data_queue.empty():
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()

                    frame_duration = 30
                    n = int(16000 * (frame_duration / 1000) * 2)
                    speech_count = sum(1 for i in range(0, len(audio_data), n) if is_speech(audio_data[i:i + n]))

                    if speech_count > 0:
                        # Look at the gap since the last speech to either close the
                        # previous utterance (real pause) or learn the speaker's
                        # within-utterance rhythm (short gap).
                        if phrase_time:
                            gap = (now - phrase_time).total_seconds()
                            if gap >= utterance_end_threshold():
                                finalize_utterance(force=gap >= max_pause)
                            elif gap > 0.25:
                                intra_pause = 0.3 * gap + 0.7 * intra_pause

                        phrase_time = now

                        # Accumulate raw audio for the current utterance and
                        # re-transcribe the whole buffer so Whisper keeps full
                        # context (no words cut at chunk boundaries).
                        phrase_bytes += audio_data
                        audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        result = audio_model.transcribe(audio_np, **transcribe_options)
                        text = result['text'].strip()

                        if text and not is_hallucination(text):
                            current_phrase = text
                            # Live partial feedback: shows speech is being heard
                            # without changing the image (trigger 0 = in progress).
                            osc_client.send_message("/transcription", current_phrase.strip())
                            osc_client.send_message("/trigger", 0)
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