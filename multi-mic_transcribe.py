#!/usr/bin/env python3
"""
Multi-Channel Real-Time Transcription CLI
Supports multiple microphones on same audio interface (e.g., Scarlett 18i20)
"""

import argparse
import sys
import os
import numpy as np
import whisper
import torch
import webrtcvad
import logging
from datetime import datetime, timedelta
from queue import Queue, Empty
from time import sleep
from pathlib import Path
import json
import signal
from typing import Optional, Dict, Any

# Try sounddevice for multi-channel support
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("❌ sounddevice required:  pip install sounddevice")
    sys.exit(1)

# Rich for beautiful CLI (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# OSC (optional)
try:
    from pythonosc.udp_client import SimpleUDPClient
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False


# =============================================================================
# Console UI
# =============================================================================

class UI:
    """Clean console interface"""
    
    def __init__(self, use_rich: bool = RICH_AVAILABLE, quiet: bool = False):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.quiet = quiet
        self. console = Console() if self.use_rich else None
    
    def header(self, title: str, **kwargs):
        """Print header"""
        if self.quiet:
            return
        
        if self.use_rich:
            table = Table(title=f"🎙️  {title}", show_header=False, 
                         border_style="cyan", box=box.ROUNDED)
            for key, value in kwargs.items():
                table.add_row(key. replace('_', ' ').title(), str(value))
            self.console.print(table)
        else:
            print(f"\n{'='*60}")
            print(f"🎙️  {title}")
            print(f"{'='*60}")
            for key, value in kwargs.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            print(f"{'='*60}\n")
    
    def status(self, msg: str, style: str = "info"):
        """Print status message"""
        if self.quiet:
            return
        
        icons = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "error": "❌", "work": "⚙️"}
        icon = icons.get(style, "•")
        
        if self.use_rich:
            colors = {"info": "cyan", "ok":  "green", "warn": "yellow", "error": "red", "work": "magenta"}
            self.console.print(f"{icon} {msg}", style=colors. get(style, "white"))
        else:
            print(f"{icon} {msg}")
    
    def transcription(self, lines: list, current:  str = "", label: str = "Transcription"):
        """Print transcription"""
        if self.quiet:
            return
        
        if self.use_rich:
            content = "\n".join(lines[-10:])  # Last 10 lines
            if current:
                content += f"\n[dim cyan]→ {current}[/dim cyan]"
            
            panel = Panel(content or "[dim]Waiting for speech...[/dim]",
                         title=f"📝 {label}", border_style="green", box=box.ROUNDED)
            self.console.print(panel)
        else:
            print(f"\n{'─'*60}")
            print(f"📝 {label}")
            print(f"{'─'*60}")
            for line in lines[-10:]:
                print(line)
            if current:
                print(f"→ {current}")
            print(f"{'─'*60}\n")
    
    def realtime(self, text: str):
        """Print real-time text (ephemeral)"""
        if self.quiet or not text:
            return
        print(f"\r→ {text[: 80]: <80}", end="", flush=True)


# =============================================================================
# Audio Device Management
# =============================================================================

def list_audio_devices():
    """List all audio devices with channel counts"""
    if not SOUNDDEVICE_AVAILABLE:
        print("sounddevice not available")
        return
    
    print("\n📋 Available Audio Devices:\n")
    print(f"{'ID':<4} {'Name':<50} {'In':<4} {'Out':<4} {'SR':<6}")
    print("─" * 80)
    
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        is_scarlett = "scarlett" in dev['name'].lower()
        prefix = "🎚️ " if is_scarlett else "   "
        
        print(f"{prefix}{idx:<3} {dev['name']:<49} "
              f"{dev['max_input_channels']:<4} "
              f"{dev['max_output_channels']:<4} "
              f"{int(dev['default_samplerate']):<6}")
        
        # Show channel details for multi-channel devices
        if dev['max_input_channels'] > 2:
            print(f"     └─ Channels: 1-{dev['max_input_channels']} available")
    
    print("\n💡 For Scarlett 18i20: Use device ID + channel (1-18)")
    print("   Example: --device 1 --channel 3\n")


def get_device_info(device_id: int) -> dict:
    """Get device information"""
    try:
        return sd.query_devices(device_id)
    except Exception as e: 
        print(f"❌ Invalid device ID: {e}")
        sys.exit(1)


# =============================================================================
# Real-Time Transcriber
# =============================================================================

class ChannelTranscriber:
    """Real-time transcriber for a single channel"""
    
    def __init__(self, config: dict, ui: UI):
        self.config = config
        self.ui = ui
        
        # Audio settings
        self.device_id = config['device_id']
        self. channel = config['channel']
        self.sample_rate = config. get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration', 1.0)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Transcription settings
        self.model_name = config.get('model', 'turbo')
        self.language = config.get('language')
        self.energy_threshold = config.get('energy_threshold', 0.01)
        self.phrase_timeout = config.get('phrase_timeout', 3.0)
        
        # Label for output
        self.label = config. get('label', f"ch{self.channel}")
        
        # State
        self.model = None
        self.vad = None
        self.audio_queue = Queue()
        self.transcription = []
        self.current_phrase = ""
        self.phrase_buffer = []
        self.phrase_duration = 0.0
        self.last_speech_time = None
        self.running = False
        
        # Output
        self.output_dir = Path(config. get('output_dir', './transcriptions'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self._get_output_file()
        
        # OSC (optional)
        self.osc_client = None
        if config.get('osc_enabled') and OSC_AVAILABLE: 
            try:
                self.osc_client = SimpleUDPClient(
                    config. get('osc_ip', '127.0.0.1'),
                    config.get('osc_port', 9000)
                )
                self.ui.status(f"OSC enabled: {config['osc_ip']}:{config['osc_port']}", "ok")
            except Exception as e: 
                self.ui.status(f"OSC failed: {e}", "warn")
    
    def _get_output_file(self) -> Path:
        """Get unique output filename"""
        idx = 0
        while True:
            # Ensure unique filename with 02d suffix to avoid overwriting
            filename = self.output_dir / f"transcript_{self.label}_{idx:02d}.txt"
            if not filename.exists():
                return filename
            idx += 1
    
    def load_model(self):
        """Load Whisper model"""
        self.ui.status(f"Loading Whisper model: {self.model_name}", "work")
        try:
            self.model = whisper.load_model(self. model_name)
            self.ui.status("Model loaded", "ok")
        except Exception as e:
            self.ui. status(f"Model load failed: {e}", "error")
            raise
    
    def setup_vad(self):
        """Setup Voice Activity Detection"""
        try:
            self.vad = webrtcvad.Vad(3)
            self.ui.status("VAD initialized", "ok")
        except Exception: 
            self.ui.status("VAD unavailable (using RMS)", "warn")
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech"""
        # Convert to int16 for VAD
        audio_int16 = (audio * 32768).astype(np.int16)
        
        if self.vad:
            try:
                # Check 30ms frames
                frame_size = int(self.sample_rate * 0.03)
                frames = [audio_int16[i:i+frame_size] 
                         for i in range(0, len(audio_int16), frame_size)
                         if len(audio_int16[i:i+frame_size]) == frame_size]
                
                speech_frames = sum(1 for frame in frames 
                                  if self.vad.is_speech(frame. tobytes(), self.sample_rate))
                return speech_frames > 0
            except Exception:
                pass
        
        # Fallback:  RMS threshold
        rms = np.sqrt(np.mean(audio ** 2))
        return rms >= self.energy_threshold
    
    def transcribe(self, audio: np. ndarray) -> str:
        """Transcribe audio chunk"""
        if self.model is None:
            return ""
        
        try:
            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=torch.cuda.is_available(),
                beam_size=5,
                temperature=0.0,
            )
            return result['text']. strip()
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return ""
    
    def finalize_phrase(self):
        """Finalize current phrase"""
        if not self.current_phrase. strip():
            return
        
        phrase = self.current_phrase.strip()
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {phrase}"
        
        self.transcription.append(line)
        
        # Write to file
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(line + "\n")
                f.flush()
        except Exception as e:
            logging.error(f"Write error: {e}")
        
        # Send OSC
        if self.osc_client:
            try:
                self.osc_client.send_message(f"/{self.label}/phrase", phrase)
                self.osc_client.send_message(f"/{self.label}/trigger", 1)
            except Exception: 
                pass
        
        # Display
        self.ui.transcription(self.transcription, "", self.label)
        
        self.current_phrase = ""
        self.phrase_buffer = []
        self.phrase_duration = 0.0
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status: 
            logging.warning(f"Audio status: {status}")
        
        # Extract single channel
        if indata.shape[1] > 1:
            # Multi-channel: extract our channel
            channel_idx = self.channel - 1  # Convert to 0-indexed
            if channel_idx < indata.shape[1]:
                audio = indata[:, channel_idx]. copy()
            else:
                logging.error(f"Channel {self.channel} not available")
                return
        else:
            audio = indata[: , 0].copy()
        
        self.audio_queue.put(audio)
    
    def process_audio(self):
        """Process audio from queue"""
        while self.running:
            try:
                # Get audio chunk
                audio = self.audio_queue.get(timeout=0.1)
                
                # Check for speech
                has_speech = self.is_speech(audio)
                
                if has_speech:
                    self.last_speech_time = datetime.now()
                    self.phrase_buffer.append(audio)
                    self.phrase_duration += len(audio) / self.sample_rate
                    
                    # Transcribe periodically
                    if self.phrase_duration >= 2.0:
                        phrase_audio = np.concatenate(self. phrase_buffer)
                        text = self.transcribe(phrase_audio)
                        
                        if text:
                            self.current_phrase += " " + text
                            self. ui.realtime(self.current_phrase)
                            
                            # Send OSC
                            if self. osc_client:
                                try:
                                    self. osc_client.send_message(f"/{self.label}/live", 
                                                                self.current_phrase. strip())
                                except Exception: 
                                    pass
                        
                        self.phrase_buffer = []
                        self.phrase_duration = 0.0
                
                # Check for phrase timeout
                if self.last_speech_time: 
                    silence = (datetime.now() - self.last_speech_time).total_seconds()
                    if silence >= self.phrase_timeout:
                        self.finalize_phrase()
                        self.last_speech_time = None
                
            except Empty:
                continue
            except Exception as e: 
                if self.running:
                    logging. error(f"Process error: {e}")
                sleep(0.01)
    
    def start(self):
        """Start transcription"""
        # Load model
        self.load_model()
        self.setup_vad()
        
        # Get device info
        device_info = get_device_info(self. device_id)
        max_channels = device_info['max_input_channels']
        
        if self.channel > max_channels:
            self.ui.status(f"Channel {self.channel} not available (max: {max_channels})", "error")
            sys.exit(1)
        
        # Display config
        self.ui.header(
            f"Channel {self.channel} Transcription",
            device=device_info['name'],
            channel=f"{self.channel} of {max_channels}",
            model=self.model_name,
            output=self.output_file,
            label=self.label
        )
        
        # Start processing thread
        import threading
        self.running = True
        processor = threading.Thread(target=self.process_audio, daemon=True)
        processor.start()
        
        # Start audio stream
        self.ui.status(f"🎙️  Recording from channel {self.channel}...  (Ctrl+C to stop)", "ok")
        
        try:
            with sd. InputStream(
                device=self.device_id,
                channels=max_channels,  # Open all channels
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self.audio_callback
            ):
                while self.running:
                    sleep(0.1)
        
        except KeyboardInterrupt: 
            self.ui.status("\n\nStopping.. .", "warn")
        finally:
            self.stop()
    
    def stop(self):
        """Stop transcription"""
        self.running = False
        
        # Finalize any remaining phrase
        if self.phrase_buffer:
            phrase_audio = np.concatenate(self.phrase_buffer)
            text = self.transcribe(phrase_audio)
            if text: 
                self.current_phrase += " " + text
        
        self.finalize_phrase()
        
        self.ui.status(f"✅ Saved to:  {self.output_file}", "ok")
        self.ui.status(f"📊 Total phrases: {len(self.transcription)}", "info")


# =============================================================================
# CLI
# =============================================================================

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="🎙️  Multi-Channel Real-Time Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List devices
  %(prog)s --list
  
  # Transcribe from Scarlett channel 3
  %(prog)s --device 1 --channel 3 --label presenter
  
  # With OSC output
  %(prog)s --device 1 --channel 5 --osc --osc-port 9001
  
  # Multiple instances (different terminals)
  %(prog)s --device 1 --channel 1 --label mic1 &
  %(prog)s --device 1 --channel 2 --label mic2 &
  
  # Use config file
  %(prog)s --config mic_setup.json
        """
    )
    
    # Device selection
    parser.add_argument('--list', action='store_true',
                       help='List available audio devices and exit')
    parser.add_argument('--device', '-d', type=int,
                       help='Audio device ID (see --list)')
    parser.add_argument('--channel', '-c', type=int, default=1,
                       help='Channel number (1-based, default: 1)')
    
    # Model settings
    parser.add_argument('--model', '-m', default='turbo',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                       help='Whisper model (default: turbo)')
    parser.add_argument('--language', '-l', default=None,
                       help='Language code (default: auto-detect)')
    
    # Labeling
    parser.add_argument('--label', type=str,
                       help='Label for this channel (default: ch{N})')
    
    # Audio settings
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--energy-threshold', type=float, default=0.01,
                       help='Speech detection threshold (default: 0.01)')
    parser.add_argument('--phrase-timeout', type=float, default=3.0,
                       help='Silence timeout for phrase end (default: 3.0s)')
    
    # Output
    parser.add_argument('--output-dir', '-o', type=str, default='./transcriptions',
                       help='Output directory (default:  ./transcriptions)')
    
    # OSC
    parser.add_argument('--osc', action='store_true',
                       help='Enable OSC output')
    parser.add_argument('--osc-ip', default='127.0.0.1',
                       help='OSC IP address (default: 127.0.0.1)')
    parser.add_argument('--osc-port', type=int, default=9000,
                       help='OSC port (default:  9000)')
    
    # Config file
    parser.add_argument('--config', type=str,
                       help='Load settings from JSON config file')
    
    # UI
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (only errors)')
    parser.add_argument('--no-rich', action='store_true',
                       help='Disable rich formatting')
    
    return parser


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Config load error: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Force UTF-8 encoding for stdout/stderr (prevents print errors on Windows)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    # Setup logging
    log_level = logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check dependencies
    if not SOUNDDEVICE_AVAILABLE:
        print("❌ sounddevice required:  pip install sounddevice")
        sys.exit(1)
    
    # List devices
    if args.list:
        list_audio_devices()
        sys.exit(0)
    
    # Build config
    if args.config:
        config = load_config(args.config)
    else:
        if args.device is None:
            print("❌ --device required (use --list to see devices)")
            parser.print_help()
            sys.exit(1)
        
        config = {
            'device_id': args.device,
            'channel': args.channel,
            'model': args.model,
            'language': args.language,
            'label': args. label or f"ch{args.channel}",
            'sample_rate':  args.sample_rate,
            'energy_threshold': args.energy_threshold,
            'phrase_timeout': args.phrase_timeout,
            'output_dir': args.output_dir,
            'osc_enabled': args.osc,
            'osc_ip': args.osc_ip,
            'osc_port': args.osc_port,
        }
    
    # Create UI
    ui = UI(use_rich=not args.no_rich, quiet=args.quiet)
    
    # Create and start transcriber
    transcriber = ChannelTranscriber(config, ui)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        transcriber.running = False
    
    signal.signal(signal. SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start
    transcriber.start()


if __name__ == "__main__":
    main()