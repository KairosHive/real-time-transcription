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

# faster-whisper (CTranslate2) runs the same Whisper weights noticeably faster
# and with less VRAM than openai-whisper — useful when the GPU is shared with
# other models. It is optional; we fall back to openai-whisper if unavailable.
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


def _decode_kwargs(language, initial_prompt):
    """Decode parameters common to both backends (beam width is added per-call)."""
    return dict(
        language=language,
        initial_prompt=initial_prompt,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )


class OpenAIWhisperBackend:
    """Reference openai-whisper backend."""

    def __init__(self, model_name, language, initial_prompt):
        self.model = whisper.load_model(model_name)
        self.language = language
        self.initial_prompt = initial_prompt
        self.fp16 = torch.cuda.is_available()

    def transcribe(self, audio_np, beam_size):
        # openai-whisper uses greedy decoding when beam_size is None.
        bs = beam_size if beam_size and beam_size > 1 else None
        result = self.model.transcribe(
            audio_np, fp16=self.fp16, logprob_threshold=-1.0,
            beam_size=bs, best_of=bs,
            **_decode_kwargs(self.language, self.initial_prompt),
        )
        return result['text'].strip()


class FasterWhisperBackend:
    """faster-whisper (CTranslate2) backend: same weights, faster + less VRAM."""

    # openai model names -> ordered faster-whisper model ids to try. The bare
    # names need faster-whisper >= 1.1.0; the explicit HF repo id is the
    # version-independent fallback (downloaded from HuggingFace on first use).
    _ALIASES = {
        "turbo": ["large-v3-turbo", "turbo", "deepdml/faster-whisper-large-v3-turbo-ct2"],
        "large": ["large-v3", "large"],
    }

    def __init__(self, model_name, language, initial_prompt, compute_type):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        candidates = self._ALIASES.get(model_name, [model_name])

        self.model = None
        last_err = None
        for model_id in candidates:
            try:
                self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
                logging.info(f"faster-whisper loaded model id: {model_id}")
                break
            except Exception as e:
                last_err = e
                logging.warning(f"faster-whisper could not load '{model_id}': {e}")
        if self.model is None:
            raise RuntimeError(
                f"No faster-whisper model id worked for '{model_name}'. "
                f"Try: pip install -U faster-whisper. Last error: {last_err}"
            )

        self.language = language
        self.initial_prompt = initial_prompt

    def transcribe(self, audio_np, beam_size):
        # faster-whisper uses beam_size=1 for greedy and names the threshold
        # log_prob_threshold; it streams segments that we join into one string.
        bs = max(1, beam_size or 1)
        segments, _ = self.model.transcribe(
            audio_np, log_prob_threshold=-1.0,
            beam_size=bs, best_of=bs,
            **_decode_kwargs(self.language, self.initial_prompt),
        )
        return "".join(seg.text for seg in segments).strip()


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
    parser.add_argument("--final_beam_size", default=5, type=int,
                        help="Beam width for the final, prompt-driving pass. Lower (e.g. 1) cuts GPU load on a busy GPU.")
    parser.add_argument("--backend", default="auto", choices=["auto", "faster", "openai"],
                        help="Inference backend. 'faster' = faster-whisper (CTranslate2): faster and lighter on VRAM. 'auto' uses it when installed.")
    parser.add_argument("--compute_type", default="float16",
                        help="faster-whisper compute type (e.g. float16, int8_float16, int8). int8_float16 cuts VRAM further.")
    parser.add_argument("--initial_energy_threshold", default=1000,
                        help="Initial energy level for mic to detect.", type=int)
    parser.add_argument("--initial_record_timeout", default=0.7,
                        help="How often (seconds) live partial transcription is refreshed and sent over OSC. Lower = snappier feedback, more CPU.", type=float)
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

    backend_choice = args.backend
    if backend_choice == "auto":
        backend_choice = "faster" if FASTER_WHISPER_AVAILABLE else "openai"

    audio_model = None
    if backend_choice == "faster":
        if not FASTER_WHISPER_AVAILABLE:
            logging.warning("faster-whisper not installed (pip install faster-whisper); using openai-whisper")
        else:
            logging.info(f"Loading faster-whisper model: {model} ({args.compute_type})")
            try:
                audio_model = FasterWhisperBackend(model, language, args.initial_prompt, args.compute_type)
                logging.info("faster-whisper model loaded successfully")
            except Exception as e:
                logging.error(f"faster-whisper load failed ({e}); falling back to openai-whisper")

    if audio_model is None:
        logging.info(f"Loading Whisper model: {model}")
        try:
            audio_model = OpenAIWhisperBackend(model, language, args.initial_prompt)
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

    # Live partials run on every refresh, so they decode greedily (beam_size=1)
    # for the lowest latency. The final pass runs once, when the utterance ends
    # and feeds the image prompt, so it can afford a wider beam for accuracy.
    final_beam = max(1, args.final_beam_size)

    def transcribe(audio_bytes, beam_size):
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_model.transcribe(audio_np, beam_size)

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

        # One accurate beam-search pass over the full utterance for the prompt
        # that actually drives the image (the greedy partials were for feedback).
        if phrase_bytes:
            try:
                refined = transcribe(phrase_bytes, final_beam)
                if refined and not is_hallucination(refined):
                    text = refined
            except Exception as e:
                logging.error(f"Final transcription error: {e}")

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
                        # re-transcribe the whole buffer (greedy = fast) so Whisper
                        # keeps full context without words cut at chunk boundaries.
                        phrase_bytes += audio_data
                        text = transcribe(phrase_bytes, beam_size=1)

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