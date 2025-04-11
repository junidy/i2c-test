import asyncio
import json
import logging
import platform # For OS detection
import subprocess
import time
import os # Added for path operations and file deletion
import glob # Added for finding wav files
from copy import deepcopy
import struct

import websockets
import jsonpatch # For applying JSON patches RFC 6902

# --- Inferencing Configuration ---
# --- NEW: Define base directory and relative paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model") # Assuming model script is in 'model' subdir
INPUT_TRACKS_DIR = os.path.join(MODEL_DIR, "input_tracks") # Where teammate puts raw audio
INFERENCE_SCRIPT_PATH = os.path.join(MODEL_DIR, "inference.py") # Path to the script
INFERENCE_OUTPUT_DIR = MODEL_DIR # Save output.json in the model dir
INFERENCE_OUTPUT_FILENAME = "output.json" # Standardized output filename
DEFAULT_CKPT_FILENAME = "dmc_2.ckpt" # Model checkpoint filename
MODEL_CKPT_PATH = os.path.join(MODEL_DIR, DEFAULT_CKPT_FILENAME) # Full path to checkpoint

RECORDING_DELAY_SECONDS = 6 # Renamed from INFERENCE_DELAY_SECONDS for clarity
POST_RECORDING_BUFFER_SECONDS = 1 # Buffer time before processing

# --- I2C Configuration ---
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8765       # WebSocket port
STM32_I2C_ADDR = 0x42 # Example I2C address for your STM32
LOG_LEVEL = logging.INFO

I2C_COMMAND_DELAY_S = 0.002 # Delay in seconds between sending I2C commands (e.g., 2ms)

# --- Parameter Name Mapping (From Inference Output to Internal State) ---
# Used when parsing output.json
PARAM_MAP = {
    "Gain dB": "digital_gain", # Map external name to internal name
    "low_shelf_gain_db": "equalizer/lowShelf/gain_db",
    "low_shelf_cutoff_freq": "equalizer/lowShelf/cutoff_freq",
    "low_shelf_q_factor": "equalizer/lowShelf/q_factor",
    "band0_gain_db": "equalizer/band0/gain_db",
    "band0_cutoff_freq": "equalizer/band0/cutoff_freq",
    "band0_q_factor": "equalizer/band0/q_factor",
    "band1_gain_db": "equalizer/band1/gain_db",
    "band1_cutoff_freq": "equalizer/band1/cutoff_freq",
    "band1_q_factor": "equalizer/band1/q_factor",
    "band2_gain_db": "equalizer/band2/gain_db",
    "band2_cutoff_freq": "equalizer/band2/cutoff_freq",
    "band2_q_factor": "equalizer/band2/q_factor",
    "band3_gain_db": "equalizer/band3/gain_db",
    "band3_cutoff_freq": "equalizer/band3/cutoff_freq",
    "band3_q_factor": "equalizer/band3/q_factor",
    "high_shelf_gain_db": "equalizer/highShelf/gain_db",
    "high_shelf_cutoff_freq": "equalizer/highShelf/cutoff_freq",
    "high_shelf_q_factor": "equalizer/highShelf/q_factor",
    "threshold_db": "compressor/threshold_db",
    "ratio": "compressor/ratio",
    "attack_ms": "compressor/attack_ms",
    "release_ms": "compressor/release_ms",
    "knee_db": "compressor/knee_db",
    "makeup_gain_db": "compressor/makeup_gain_db",
    "Pan": "panning", # Map external name to internal name
}
# --- End Model JSON Parameter Mapping ---



# --- STM Command Protocol ID Mappings (Python Side) ---

# Channel IDs:
# 0-8: Direct Channel Index (0=Master, 1-8=Inputs)
# 9: soloing_active
# 10: inferencing_active
# 11: hw_init_ready
CH_ID_SOLOING_ACTIVE = 9
CH_ID_INFERENCING_ACTIVE = 10
CH_ID_HW_INIT_READY = 11

# Effect IDs (within a channel, index 0-8):
# 0: Direct Channel Parameters (mute, pan, gain, etc.)
# 1: Equalizer
# 2: Compressor
# 3: Distortion
# 4: Phaser
# 5: Reverb
FX_ID_DIRECT = 0
FX_ID_EQ = 1
FX_ID_COMP = 2
FX_ID_DIST = 3
FX_ID_PHASER = 4
FX_ID_REVERB = 5

# Parameter IDs for Direct Channel (FX_ID_DIRECT = 0):
# Based on order in ChannelParameters definition (mixer_schema.json / mixer_state.h)
PARAM_ID_DIRECT_MUTED = 0
PARAM_ID_DIRECT_SOLOED = 1
PARAM_ID_DIRECT_PANNING = 2
PARAM_ID_DIRECT_DIGITAL_GAIN = 3
PARAM_ID_DIRECT_ANALOG_GAIN = 4 # Special handling on STM side
PARAM_ID_DIRECT_STEREO = 5

# Parameter IDs for Equalizer (FX_ID_EQ = 1):
PARAM_ID_EQ_ENABLED = 0
PARAM_ID_EQ_LS_GAIN = 1
PARAM_ID_EQ_LS_FREQ = 2
PARAM_ID_EQ_LS_Q = 3
PARAM_ID_EQ_HS_GAIN = 4
PARAM_ID_EQ_HS_FREQ = 5
PARAM_ID_EQ_HS_Q = 6
PARAM_ID_EQ_B0_GAIN = 7
PARAM_ID_EQ_B0_FREQ = 8
PARAM_ID_EQ_B0_Q = 9
PARAM_ID_EQ_B1_GAIN = 10
PARAM_ID_EQ_B1_FREQ = 11
PARAM_ID_EQ_B1_Q = 12
PARAM_ID_EQ_B2_GAIN = 13
PARAM_ID_EQ_B2_FREQ = 14
PARAM_ID_EQ_B2_Q = 15
PARAM_ID_EQ_B3_GAIN = 16
PARAM_ID_EQ_B3_FREQ = 17
PARAM_ID_EQ_B3_Q = 18

# Parameter IDs for Compressor (FX_ID_COMP = 2):
PARAM_ID_COMP_ENABLED = 0
PARAM_ID_COMP_THRESH = 1
PARAM_ID_COMP_RATIO = 2
PARAM_ID_COMP_ATTACK = 3
PARAM_ID_COMP_RELEASE = 4
PARAM_ID_COMP_KNEE = 5
PARAM_ID_COMP_MAKEUP = 6

# Parameter IDs for Distortion (FX_ID_DIST = 3):
PARAM_ID_DIST_ENABLED = 0
PARAM_ID_DIST_DRIVE = 1
PARAM_ID_DIST_OUTPUT = 2

# Parameter IDs for Phaser (FX_ID_PHASER = 4):
PARAM_ID_PHASER_ENABLED = 0
PARAM_ID_PHASER_RATE = 1
PARAM_ID_PHASER_DEPTH = 2

# Parameter IDs for Reverb (FX_ID_REVERB = 5):
PARAM_ID_REVERB_ENABLED = 0
PARAM_ID_REVERB_DECAY = 1
PARAM_ID_REVERB_WET = 2

# Parameter ID for Top-Level Bool Flags (Channel ID 9-11)
PARAM_ID_TOPLEVEL_BOOL = 0 # The channel ID itself identifies the flag

# --- End Protocol ID Mappings ---



# --- I2C Specific Configuration ---
# For RPi (smbus2)
PI_I2C_BUS_NUM = 1
# For Mac (pyftdi) - Find your FT232H URL (e.g., use 'python -m pyftdi.tools.list_devices')
# Common URLs: 'ftdi://ftdi:232h/1', 'ftdi://::/1'
FTDI_DEVICE_URL = 'ftdi://ftdi:232h/1'
# FTDI I2C frequency (optional, defaults usually ok)
FTDI_I2C_FREQ = 10000 # 100kHz example

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Global Async Queue for I2C Commands ---
i2c_command_queue = asyncio.Queue()

# --- I2C Adapter Class ---
# Provides a consistent interface for different I2C libraries
class I2CAdapter:
    def __init__(self, bus_or_port, adapter_type):
        self._interface = bus_or_port
        self._type = adapter_type # 'smbus' or 'pyftdi'
        logging.info(f"I2C Adapter initialized using {self._type}.")

    def write_block(self, address, data, offset=0):
        """
        Writes a block of data to the I2C device.

        Args:
            address (int): The I2C device address.
            data (list or bytes or bytearray): The data bytes to send.
            offset (int): The register offset (primarily for smbus2).
        """
        if self._interface is None:
            logging.error("I2C write attempt failed: Interface not initialized.")
            return False

        try:
            data_bytes = bytes(data) # Ensure data is bytes
            if self._type == 'smbus':
                # smbus2 write_i2c_block_data expects list of ints for data
                data_list = list(data_bytes)
                self._interface.write_i2c_block_data(address, offset, data_list)
                # logging.debug(f"SMBus write_i2c_block_data(addr=0x{address:02X}, offset={offset}, data={data_list})")
            elif self._type == 'pyftdi':
                # pyftdi write_to takes address and bytes
                self._interface.write_to(address, data_bytes)
                # logging.debug(f"PyFTDI write_to(addr=0x{address:02X}, data={list(data_bytes)})")
            return True
        except (OSError, IOError) as e:
            logging.error(f"I2C write error ({self._type}): {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected I2C write error ({self._type}): {e}")
            return False

    def close(self):
        """Closes the underlying I2C connection if necessary."""
        if self._interface:
            if self._type == 'pyftdi':
                try:
                    # pyftdi controller has terminate()
                    if hasattr(self._interface, 'controller'):
                         self._interface.controller.terminate()
                         logging.info("PyFTDI I2C interface terminated.")
                    else:
                         logging.warning("Could not find controller to terminate on PyFTDI interface.")
                except Exception as e:
                    logging.error(f"Error closing PyFTDI interface: {e}")
            # smbus2 SMBus object doesn't have an explicit close method in typical usage
            elif self._type == 'smbus':
                logging.debug("SMBus interface does not require explicit closing.")
            self._interface = None


# --- Conditional I2C Initialization ---
I2C_ENABLED = False
i2c_adapter = None # Use the adapter instance
system_os = platform.system()

logging.info(f"Detected Operating System: {system_os}")

if system_os == "Linux":
    try:
        import smbus2
        bus = smbus2.SMBus(PI_I2C_BUS_NUM)
        # Perform a quick check (optional, e.g., try reading from device if possible)
        # bus.read_byte(STM32_I2C_ADDR) # Example check - might fail if device isn't ready
        i2c_adapter = I2CAdapter(bus, 'smbus')
        I2C_ENABLED = True
        logging.info(f"SMBus I2C Bus {PI_I2C_BUS_NUM} initialized successfully.")
    except ImportError:
        logging.warning("smbus2 library not found. I2C communication disabled on Linux.")
    except FileNotFoundError:
        logging.warning(f"I2C Bus {PI_I2C_BUS_NUM} not found. I2C communication disabled.")
    except OSError as e:
        logging.error(f"Failed to initialize/access SMBus {PI_I2C_BUS_NUM}: {e}. I2C disabled.")
    except Exception as e:
        logging.error(f"Unexpected error initializing SMBus {PI_I2C_BUS_NUM}: {e}. I2C disabled.")

elif system_os == "Darwin": # macOS
    try:
        from pyftdi.i2c import I2cController, I2cNackError

        # Create and configure the FTDI I2C controller
        i2c_controller = I2cController()
        i2c_controller.configure(FTDI_DEVICE_URL, frequency=FTDI_I2C_FREQ)
        # Get a port to the specific device address (pyftdi handles addressing this way)
        # We don't need a port to a specific *device* address here,
        # as we pass the address in write_to. Get the primary port.
        ftdi_port = i2c_controller.get_port(STM32_I2C_ADDR) # Use target address to check connection maybe?
        # Or more generally, just get the controller itself ready if port isn't needed yet
        # ftdi_port = i2c_controller # Might need adjustment based on how you use it

        # A simple test - try writing an empty byte array or a known safe command?
        # ftdi_port.write_to(STM32_I2C_ADDR, b'') # Example check - might cause error or do nothing

        i2c_adapter = I2CAdapter(ftdi_port, 'pyftdi') # Pass the port object
        I2C_ENABLED = True
        logging.info(f"PyFTDI I2C interface initialized successfully via {FTDI_DEVICE_URL}.")

    except ImportError:
        logging.warning("pyftdi library not found. I2C communication disabled on macOS.")
    except Exception as e: # Catch FTDI connection errors etc.
        logging.error(f"Failed to initialize PyFTDI I2C ({FTDI_DEVICE_URL}): {e}. I2C disabled.")

else:
    logging.warning(f"Unsupported OS for I2C: {system_os}. I2C communication disabled.")


# --- Global State (same as before) ---
def create_default_channel(index): # Pass index
    is_main_bus = index == 0
    default_analog_gain = 0.0 # Master default
    if 1 <= index <= 4: default_analog_gain = -3.0 # Ch 1-4
    elif 5 <= index <= 8: default_analog_gain = -9.0 # Ch 5-8

    return {
        "muted": False, "soloed": False, "panning": 0.5, "digital_gain": 0.0,
        "analog_gain": default_analog_gain, # <-- ADDED with default logic
        "stereo": is_main_bus,
        "equalizer": { # ... (same defaults) ...
            "enabled": False, "lowShelf": {"gain_db": 0.0, "cutoff_freq": 80, "q_factor": 0.707},
            "highShelf": {"gain_db": 0.0, "cutoff_freq": 12000, "q_factor": 0.707},
            "band0": {"gain_db": 0.0, "cutoff_freq": 250, "q_factor": 1.0},
            "band1": {"gain_db": 0.0, "cutoff_freq": 1000, "q_factor": 1.0},
            "band2": {"gain_db": 0.0, "cutoff_freq": 4000, "q_factor": 1.0},
            "band3": {"gain_db": 0.0, "cutoff_freq": 10000, "q_factor": 1.0},
        },
        "compressor": { # ... (same defaults) ...
            "enabled": False, "threshold_db": -20.0, "ratio": 2.0, "attack_ms": 10.0,
            "release_ms": 50.0, "knee_db": 0.0, "makeup_gain_db": 0.0,
        },
        "distortion": {"enabled": False, "drive": 0.0, "output_gain_db": 0.0},
        "phaser": {"enabled": False, "rate": 1.0, "depth": 50.0},
        "reverb": {"enabled": is_main_bus, "decay_time": 1.5, "wet_level": 25.0},
    }

MASTER_STATE = {
    "channels": [create_default_channel(i == 0) for i in range(9)],
    "soloing_active": False,
    "inferencing_active": False,
    "inferencing_state": "idle", # NEW: idle, countdown, recording, inferencing
    "hw_init_ready": False,
}
CONNECTED_CLIENTS = set()
STATE_LOCK = asyncio.Lock()
inference_task = None
recording_timer_task = None # NEW: Handle for the 6s recording timer


# --- Helper Functions ---
async def broadcast_patch(patch_list, *, exclude_client=None):
    """Sends a patch list to connected clients, optionally excluding one."""
    if not patch_list: return # Don't broadcast empty patches

    message_str = json.dumps(patch_list)
    targets = CONNECTED_CLIENTS - ({exclude_client} if exclude_client else set())

    if targets:
        logging.debug(f"Broadcasting patch to {len(targets)} clients (excluding sender: {exclude_client is not None}). Patch: {patch_list}")
        tasks = [asyncio.create_task(client.send(message_str)) for client in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client = list(targets)[i]
                logging.error(f"Error broadcasting patch to {client.remote_address}: {result}")

# --- Helper for I2C Message Value Encoding ---
def encode_value(value):
    """Encodes a Python value (float, bool) into a uint32 integer."""
    if isinstance(value, bool):
        return 1 if value else 0
    elif isinstance(value, (int, float)):
        # Pack float as 4 bytes, then unpack as uint32
        try:
            packed_float = struct.pack('<f', float(value)) # '<f' = little-endian float
            return struct.unpack('<I', packed_float)[0]    # '<I' = little-endian uint32
        except (struct.error, TypeError, OverflowError) as e:
            logging.error(f"Failed to encode value '{value}': {e}")
            return 0 # Return 0 on error? Or raise?
    else:
        logging.warning(f"Unsupported value type for encoding: {type(value)}")
        return 0


## --- Updated I2C Communication Function ---
async def send_to_stm32(patch):
    """
    Encodes and sends relevant state changes over I2C to the STM32
    using the initialized I2C adapter and the bespoke 16-byte protocol.
    """
    if not I2C_ENABLED or i2c_adapter is None:
        return

    commands_generated = 0

    for operation in patch:
        if operation.get('op') != 'replace':
            logging.warning(f"Unsupported patch operation for I2C: {operation.get('op')}")
            continue

        path = operation.get('path', '')
        value = operation.get('value')
        parts = path.strip('/').split('/') # e.g., ['channels', '1', 'compressor', 'ratio'] or ['soloing_active']

        channel_id = 0
        effect_id = 0
        param_id = 0
        encoded_value = 0
        valid_command = False

        try:
            # --- Parse Path and Determine IDs ---
            if len(parts) == 1: # Top-level parameter
                param_name = parts[0]
                param_id = PARAM_ID_TOPLEVEL_BOOL # Parameter ID is trivial here
                if param_name == 'soloing_active':
                    channel_id = CH_ID_SOLOING_ACTIVE
                    valid_command = True
                elif param_name == 'inferencing_active':
                    channel_id = CH_ID_INFERENCING_ACTIVE
                    valid_command = True
                elif param_name == 'hw_init_ready':
                    channel_id = CH_ID_HW_INIT_READY
                    valid_command = True
                elif param_name == 'inferencing_state': # NEW: Ignore sending this specific state string via I2C
                     valid_command = False # Do not send state string itself
                # --- Note: inferencing_state is handled implicitly by inferencing_active for STM ---

            elif len(parts) >= 3 and parts[0] == 'channels':
                channel_id = int(parts[1]) # 0-8
                if not (0 <= channel_id <= 8): raise ValueError("Invalid channel index")

                if len(parts) == 3: # Direct channel parameter
                    effect_id = FX_ID_DIRECT
                    param_name = parts[2]
                    if param_name == 'muted': param_id = PARAM_ID_DIRECT_MUTED; valid_command = True
                    elif param_name == 'soloed': param_id = PARAM_ID_DIRECT_SOLOED; valid_command = True
                    elif param_name == 'panning': param_id = PARAM_ID_DIRECT_PANNING; valid_command = True
                    elif param_name == 'digital_gain': param_id = PARAM_ID_DIRECT_DIGITAL_GAIN; valid_command = True
                    elif param_name == 'analog_gain': param_id = PARAM_ID_DIRECT_ANALOG_GAIN; valid_command = True # Will be handled specially by C
                    elif param_name == 'stereo': param_id = PARAM_ID_DIRECT_STEREO; valid_command = True

                elif len(parts) == 4: # Effect enable/disable
                    effect_name = parts[2]
                    param_name = parts[3]
                    if param_name == 'enabled':
                         if effect_name == 'equalizer': effect_id = FX_ID_EQ; param_id = PARAM_ID_EQ_ENABLED; valid_command = True
                         elif effect_name == 'compressor': effect_id = FX_ID_COMP; param_id = PARAM_ID_COMP_ENABLED; valid_command = True
                         elif effect_name == 'distortion': effect_id = FX_ID_DIST; param_id = PARAM_ID_DIST_ENABLED; valid_command = True
                         elif effect_name == 'phaser': effect_id = FX_ID_PHASER; param_id = PARAM_ID_PHASER_ENABLED; valid_command = True
                         elif effect_name == 'reverb': effect_id = FX_ID_REVERB; param_id = PARAM_ID_REVERB_ENABLED; valid_command = True

                elif len(parts) == 5: # Effect parameter (non-EQ band)
                    effect_name = parts[2]
                    param_name = parts[4] # e.g., ratio, drive, rate, decay_time
                    # Determine Effect ID
                    if effect_name == 'compressor': effect_id = FX_ID_COMP
                    elif effect_name == 'distortion': effect_id = FX_ID_DIST
                    elif effect_name == 'phaser': effect_id = FX_ID_PHASER
                    elif effect_name == 'reverb': effect_id = FX_ID_REVERB
                    else: raise ValueError(f"Unknown effect name: {effect_name}")

                    # Determine Parameter ID within Effect
                    if effect_id == FX_ID_COMP:
                        if param_name == 'threshold_db': param_id = PARAM_ID_COMP_THRESH; valid_command = True
                        elif param_name == 'ratio': param_id = PARAM_ID_COMP_RATIO; valid_command = True
                        elif param_name == 'attack_ms': param_id = PARAM_ID_COMP_ATTACK; valid_command = True
                        elif param_name == 'release_ms': param_id = PARAM_ID_COMP_RELEASE; valid_command = True
                        elif param_name == 'knee_db': param_id = PARAM_ID_COMP_KNEE; valid_command = True
                        elif param_name == 'makeup_gain_db': param_id = PARAM_ID_COMP_MAKEUP; valid_command = True
                    elif effect_id == FX_ID_DIST:
                        if param_name == 'drive': param_id = PARAM_ID_DIST_DRIVE; valid_command = True
                        elif param_name == 'output_gain_db': param_id = PARAM_ID_DIST_OUTPUT; valid_command = True
                    elif effect_id == FX_ID_PHASER:
                         if param_name == 'rate': param_id = PARAM_ID_PHASER_RATE; valid_command = True
                         elif param_name == 'depth': param_id = PARAM_ID_PHASER_DEPTH; valid_command = True
                    elif effect_id == FX_ID_REVERB:
                         if param_name == 'decay_time': param_id = PARAM_ID_REVERB_DECAY; valid_command = True
                         elif param_name == 'wet_level': param_id = PARAM_ID_REVERB_WET; valid_command = True

                elif len(parts) == 6 and parts[2] == 'equalizer': # EQ Band parameter
                    effect_id = FX_ID_EQ
                    band_name = parts[3] # e.g., lowShelf, band0
                    param_name = parts[5] # e.g., gain_db, cutoff_freq, q_factor

                    # Map band/param name to single EQ param_id
                    if band_name == 'lowShelf':
                        if param_name == 'gain_db': param_id = PARAM_ID_EQ_LS_GAIN; valid_command = True
                        elif param_name == 'cutoff_freq': param_id = PARAM_ID_EQ_LS_FREQ; valid_command = True
                        elif param_name == 'q_factor': param_id = PARAM_ID_EQ_LS_Q; valid_command = True
                    elif band_name == 'highShelf':
                        if param_name == 'gain_db': param_id = PARAM_ID_EQ_HS_GAIN; valid_command = True
                        elif param_name == 'cutoff_freq': param_id = PARAM_ID_EQ_HS_FREQ; valid_command = True
                        elif param_name == 'q_factor': param_id = PARAM_ID_EQ_HS_Q; valid_command = True
                    elif band_name == 'band0':
                        if param_name == 'gain_db': param_id = PARAM_ID_EQ_B0_GAIN; valid_command = True
                        elif param_name == 'cutoff_freq': param_id = PARAM_ID_EQ_B0_FREQ; valid_command = True
                        elif param_name == 'q_factor': param_id = PARAM_ID_EQ_B0_Q; valid_command = True
                    elif band_name == 'band1':
                        if param_name == 'gain_db': param_id = PARAM_ID_EQ_B1_GAIN; valid_command = True
                        elif param_name == 'cutoff_freq': param_id = PARAM_ID_EQ_B1_FREQ; valid_command = True
                        elif param_name == 'q_factor': param_id = PARAM_ID_EQ_B1_Q; valid_command = True
                    elif band_name == 'band2':
                        if param_name == 'gain_db': param_id = PARAM_ID_EQ_B2_GAIN; valid_command = True
                        elif param_name == 'cutoff_freq': param_id = PARAM_ID_EQ_B2_FREQ; valid_command = True
                        elif param_name == 'q_factor': param_id = PARAM_ID_EQ_B2_Q; valid_command = True
                    elif band_name == 'band3':
                        if param_name == 'gain_db': param_id = PARAM_ID_EQ_B3_GAIN; valid_command = True
                        elif param_name == 'cutoff_freq': param_id = PARAM_ID_EQ_B3_FREQ; valid_command = True
                        elif param_name == 'q_factor': param_id = PARAM_ID_EQ_B3_Q; valid_command = True

            # --- Encode Value and Queue Command ---
            if valid_command:
                encoded_value = encode_value(value)
                i2c_message_bytes = struct.pack('<IIII', channel_id, effect_id, param_id, encoded_value)
                try:
                    # Put the command onto the queue asynchronously
                    await i2c_command_queue.put(i2c_message_bytes)
                    commands_generated += 1
                    logging.debug(f"Queued I2C Cmd: Path='{path}' -> Chan={channel_id}, Fx={effect_id}, Param={param_id}, EncVal={encoded_value} -> Bytes={list(i2c_message_bytes)}")
                except asyncio.QueueFull:
                     # Should not happen with default unbounded queue, but good practice
                     logging.error("I2C command queue is unexpectedly full!")
                except Exception as qe:
                     logging.error(f"Error putting command onto I2C queue: {qe}")

            else:
                 # Only log if it wasn't just an unhandled path (like intermediate objects)
                 # This might be noisy, adjust logging level if needed
                 # logging.warning(f"Could not map path to I2C command: {path}")
                 pass

        except (ValueError, IndexError, TypeError) as e:
            logging.error(f"Error parsing path or value for I2C command: {path} - {e}")
        except Exception as e:
            logging.error(f"Unexpected error encoding I2C command for path {path}: {e}")

    if commands_generated > 0:
        logging.debug(f"Queued {commands_generated} I2C command(s) resulting from patch.")



# --- I2C Command Sender Task (Consumer) ---
async def i2c_sender_task():
    """
    Continuously consumes commands from the queue and sends them over I2C
    with a delay between each command.
    """
    logging.info("Starting I2C sender task...")
    while True:
        try:
            # Wait for a command from the queue
            command_bytes = await i2c_command_queue.get()

            if command_bytes is None: # Sentinel value to stop the task (optional)
                logging.info("Received stop signal for I2C sender task.")
                break

            if not I2C_ENABLED or i2c_adapter is None:
                logging.warning(f"I2C not enabled/ready. Discarding queued command: {list(command_bytes)}")
                i2c_command_queue.task_done() # Mark task as done even if discarded
                continue # Skip sending

            # Send the command using the adapter
            logging.debug(f"I2C Sender Task: Sending command: {list(command_bytes)}")
            success = i2c_adapter.write_block(STM32_I2C_ADDR, command_bytes)

            if not success:
                logging.warning(f"I2C Sender Task: Failed to send command: {list(command_bytes)}")
                # Optional: Implement retry logic here? Or just log and continue.

            # Mark this command as processed in the queue
            i2c_command_queue.task_done()

            # --- Introduce the delay ---
            await asyncio.sleep(I2C_COMMAND_DELAY_S)

        except asyncio.CancelledError:
            logging.info("I2C sender task cancelled.")
            break # Exit loop if task is cancelled
        except Exception as e:
            logging.error(f"Unexpected error in I2C sender task: {e}", exc_info=True)
            # Avoid continuous error loops - maybe add a small delay on generic error?
            await asyncio.sleep(1) # Delay before trying to get next item after error


# --- UPDATED AI Inference Workflow ---
async def run_inference_workflow():
    """Handles the AI auto-mixing process after recording."""
    global MASTER_STATE, inference_task # Keep inference_task to prevent overlaps
    logging.info("Starting post-recording inference workflow...")

    # Ensure directories exist
    if not os.path.exists(INPUT_TRACKS_DIR):
        logging.error(f"Input tracks directory does not exist: {INPUT_TRACKS_DIR}")
        # Maybe try to create it? Or just fail.
        try: os.makedirs(INPUT_TRACKS_DIR)
        except OSError as e: logging.error(f"Could not create input dir: {e}"); return
    if not os.path.exists(INFERENCE_OUTPUT_DIR):
        try: os.makedirs(INFERENCE_OUTPUT_DIR)
        except OSError as e: logging.error(f"Could not create output dir: {INFERENCE_OUTPUT_DIR}: {e}"); return

    inference_output_json_path = os.path.join(INFERENCE_OUTPUT_DIR, INFERENCE_OUTPUT_FILENAME)

    try:
        # 1. Buffer Time (Wait for file finalization)
        logging.info(f"Waiting {POST_RECORDING_BUFFER_SECONDS}s for audio file finalization...")
        await asyncio.sleep(POST_RECORDING_BUFFER_SECONDS)

        # 2. Delete Muted Tracks
        logging.info("Deleting muted tracks before inference...")
        deleted_count = 0
        async with STATE_LOCK: # Need current mute state
            muted_channels = {
                i for i, ch in enumerate(MASTER_STATE['channels'])
                if i >= 1 and i <= 8 and ch['muted'] # Only check input channels 1-8
            }
        logging.debug(f"Channels marked as muted: {muted_channels}")

        for i in range(1, 9): # Iterate through physical input channels 1-8
            if i in muted_channels:
                track_filename = f"{i}.wav"
                track_filepath = os.path.join(INPUT_TRACKS_DIR, track_filename)
                if os.path.exists(track_filepath):
                    try:
                        os.remove(track_filepath)
                        logging.info(f"  - Deleted muted track file: {track_filepath}")
                        deleted_count += 1
                    except OSError as e:
                        logging.error(f"  - Failed to delete muted track file {track_filepath}: {e}")
                # else: # Optional: Log if muted track file wasn't found anyway
                #    logging.debug(f"  - Muted track file not found (already gone?): {track_filepath}")

        logging.info(f"Finished deleting {deleted_count} muted track(s).")


        # 3. Construct and Run the inference.py Script
        cmd = [
            'python', # Or specific python executable if needed e.g., sys.executable
            INFERENCE_SCRIPT_PATH,
            '--track_dir', INPUT_TRACKS_DIR,
            '--output_dir', INFERENCE_OUTPUT_DIR, # Tells script where to write output.json
            '--ckpt_path', MODEL_CKPT_PATH
            # No --song needed if using fixed output filename
        ]
        logging.info(f"Running inference script: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd, # Unpack command list
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=MODEL_DIR # <-- ADD THIS LINE
        )
        stdout, stderr = await process.communicate()

        stdout_decoded = stdout.decode().strip() if stdout else ""
        stderr_decoded = stderr.decode().strip() if stderr else ""

        if stdout_decoded: logging.info(f"Inference script stdout:\n---\n{stdout_decoded}\n---")
        if stderr_decoded: logging.warning(f"Inference script stderr:\n---\n{stderr_decoded}\n---")

        # 4. Check Return Code & Parse Output
        if process.returncode != 0:
            logging.error(f"Inference script failed with return code {process.returncode}.")
            # Don't apply results, just transition state back to idle below
            raise ChildProcessError("Inference script failed.")

        logging.info(f"Inference script completed. Parsing {inference_output_json_path}...")
        try:
            with open(inference_output_json_path, 'r') as f:
                inferred_results = json.load(f)

            if "tracks" not in inferred_results or not isinstance(inferred_results["tracks"], dict):
                 logging.error("Invalid format in output.json: Missing or invalid 'tracks' key.")
                 raise ValueError("Invalid inference output format")

            # 5. Generate JSON Patch from Parsed Results
            inference_patch_list = []
            processed_channels = set() # Keep track of channels we got results for

            for channel_idx_str, params in inferred_results["tracks"].items():
                try:
                    channel_index = int(channel_idx_str) # Convert string key "1", "3" etc. to int
                    if not (1 <= channel_index <= 8): # Validate range
                        logging.warning(f"Skipping inference results for invalid channel index: {channel_idx_str}")
                        continue
                    processed_channels.add(channel_index)

                    # --- Map parameters to JSON Patch operations ---
                    for json_key, internal_path_suffix in PARAM_MAP.items():
                        if json_key in params:
                            patch_path = f"/channels/{channel_index}/{internal_path_suffix}"
                            inference_patch_list.append({
                                "op": "replace",
                                "path": patch_path,
                                "value": params[json_key] # Use the value directly from JSON
                            })
                        # else: # Optional: Log if expected param is missing in output
                        #    logging.debug(f"Param '{json_key}' not found in inference output for channel {channel_index}")

                    # --- Enable EQ and Compressor for processed channels ---
                    # Assuming model always provides settings for these if it processes a track
                    inference_patch_list.append({"op": "replace", "path": f"/channels/{channel_index}/equalizer/enabled", "value": True})
                    inference_patch_list.append({"op": "replace", "path": f"/channels/{channel_index}/compressor/enabled", "value": True})

                except ValueError:
                     logging.warning(f"Skipping inference results for non-integer channel key: {channel_idx_str}")
                except Exception as e:
                     logging.error(f"Error processing inference results for channel {channel_idx_str}: {e}")

            # --- Reset other effects (Dist, Phaser, Reverb) for PROCESSED channels ---
            for channel_index in processed_channels:
                 logging.debug(f"Resetting Distortion/Phaser/Reverb for processed channel {channel_index}")
                 inference_patch_list.append({"op": "replace", "path": f"/channels/{channel_index}/distortion/enabled", "value": False})
                 inference_patch_list.append({"op": "replace", "path": f"/channels/{channel_index}/phaser/enabled", "value": False})
                 # Reverb only applies to master, which isn't processed by inference, so no need to reset here
                 # if channel_index == 0: # If master was somehow processed
                 #    inference_patch_list.append({"op": "replace", "path": f"/channels/0/reverb/enabled", "value": False}) # Example

            logging.info(f"Generated {len(inference_patch_list)} patch operations from inference.")

            # 6. Apply, Broadcast, Send to STM32
            if inference_patch_list:
                final_solo_patch = None
                async with STATE_LOCK:
                    try:
                        jsonpatch.apply_patch(MASTER_STATE, inference_patch_list, in_place=True)
                        # Check solo status again after applying patch
                        current_soloing = MASTER_STATE['soloing_active']
                        new_soloing = any(ch['soloed'] for ch in MASTER_STATE['channels'])
                        if new_soloing != current_soloing:
                            MASTER_STATE['soloing_active'] = new_soloing
                            final_solo_patch = [{"op": "replace", "path": "/soloing_active", "value": new_soloing}]
                    except Exception as e:
                        logging.error(f"Error applying inference patch list: {e}")
                        raise # Re-raise to trigger outer error handling

                # Broadcast inference results (TO ALL)
                await broadcast_patch(inference_patch_list)
                await send_to_stm32(inference_patch_list) # Send inference commands

                # Broadcast solo update if needed (TO ALL)
                if final_solo_patch:
                    await broadcast_patch(final_solo_patch)
                    await send_to_stm32(final_solo_patch)

        except FileNotFoundError:
            logging.error(f"Inference output file not found: {inference_output_json_path}")
            raise # Propagate error
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from inference output file: {inference_output_json_path}")
            raise # Propagate error
        except Exception as e:
            logging.error(f"Unexpected error processing inference results: {e}")
            raise # Propagate error

    except (asyncio.CancelledError, ChildProcessError, ValueError, FileNotFoundError, json.JSONDecodeError, Exception) as e:
        # Catch specific errors from above and general errors
        if isinstance(e, asyncio.CancelledError):
             logging.info("AI inference workflow cancelled.")
             # State cleanup might be handled by the caller that cancelled it
        else:
            logging.error(f"Error during inference workflow: {e}")
        # Fall through to finally block for state reset

    finally:
        logging.info("AI inference workflow concluding.")
        # 7. Reset inferencing_state back to "idle" REGARDLESS of success/failure/cancel
        # Ensure state/active flags are consistent
        final_state_patch = []
        async with STATE_LOCK:
            if MASTER_STATE['inferencing_state'] != "idle":
                 final_state_patch.append({"op": "replace", "path": "/inferencing_state", "value": "idle"})
                 MASTER_STATE['inferencing_state'] = "idle"
            if MASTER_STATE['inferencing_active']: # Ensure active is also false
                 final_state_patch.append({"op": "replace", "path": "/inferencing_active", "value": False})
                 MASTER_STATE['inferencing_active'] = False

        if final_state_patch:
            await broadcast_patch(final_state_patch) # Notify UIs state is idle
            await send_to_stm32(final_state_patch) # Ensure STM gets active=false if it wasn't already

        # Clean up the global task handle
        inference_task = None # Allow another run

## --- WebSocket Handler ---
async def handler(websocket, path):
    global MASTER_STATE, inference_task, recording_timer_task
    client_addr = websocket.remote_address
    logging.info(f"Client connected: {client_addr} on path '{path}'")
    CONNECTED_CLIENTS.add(websocket)

    try:
        # 1. Send current full state
        async with STATE_LOCK: current_state_copy = deepcopy(MASTER_STATE)
        await websocket.send(json.dumps(current_state_copy))
        logging.info(f"Sent initial state to {client_addr}")

        # 2. Listen for messages
        async for message in websocket:
            logging.debug(f"Received message from {client_addr}: {message}")
            try:
                patch = json.loads(message)
                if not isinstance(patch, list): continue

                applied_patch_ops = [] # Store ops that actually changed state
                solo_status_potentially_changed = False
                inference_state_change_op = None # Track specific inference state change

                # --- Apply patch and check for specific changes ---
                async with STATE_LOCK:
                    state_before = deepcopy(MASTER_STATE)
                    temp_state = deepcopy(state_before)
                    try:
                        # Apply the patch to the temporary state
                        jsonpatch.apply_patch(temp_state, patch, in_place=True)

                        # Check if state actually changed
                        if temp_state != state_before:
                            MASTER_STATE = temp_state # Commit the change
                            applied_patch_ops = patch # Store the patch that caused the change

                            # Check specific flags/paths modified by this patch
                            for op in applied_patch_ops:
                                op_path = op.get("path", "")
                                op_value = op.get("value")

                                # Check if any solo status changed
                                if op_path.startswith("/channels/") and op_path.endswith("/soloed"):
                                    solo_status_potentially_changed = True

                                # --- Track INFERENCING_STATE change ---
                                if op_path == "/inferencing_state":
                                    inference_state_change_op = op # Store the whole operation dict
                                # --- End Tracking ---

                        else:
                             logging.debug(f"Patch from {client_addr} resulted in no state change.")

                    except jsonpatch.JsonPatchException as e:
                        logging.error(f"Invalid patch from {client_addr}: {e} - Patch: {patch}")
                    except Exception as e:
                         logging.error(f"Error applying patch from {client_addr}: {e} - Patch: {patch}")


                # --- Post-Patch Processing (Outside main lock, but effects need care) ---

                # Only proceed if the patch was valid and changed something
                if applied_patch_ops:
                    # 3a. Handle Soloing Active Update
                    solo_active_update_patch = None
                    if solo_status_potentially_changed:
                        async with STATE_LOCK: # Re-acquire lock briefly to check/update solo flag
                            current_soloing = MASTER_STATE['soloing_active']
                            new_soloing = any(ch['soloed'] for ch in MASTER_STATE['channels'])
                            if new_soloing != current_soloing:
                                MASTER_STATE['soloing_active'] = new_soloing
                                solo_active_update_patch = [{"op": "replace", "path": "/soloing_active", "value": new_soloing}]
                                logging.info(f"Soloing active status changed to: {new_soloing}")

                    # 3b. Broadcast original patch EXCLUDING sender
                    await broadcast_patch(applied_patch_ops, exclude_client=websocket)

                    # 3c. Broadcast soloing_active update TO ALL (if changed)
                    if solo_active_update_patch:
                         await broadcast_patch(solo_active_update_patch) # exclude_client=None

                    # 4. Send changes to STM32
                    await send_to_stm32(applied_patch_ops)
                    if solo_active_update_patch:
                         await send_to_stm32(solo_active_update_patch)

# 5. Handle Inference State Transitions triggered by CLIENT
                    if inference_state_change_op:
                        new_inference_state = inference_state_change_op.get("value")
                        logging.info(f"Client {client_addr} set inferencing_state to: {new_inference_state}")

                        if new_inference_state == "countdown":
                           # If entering countdown, cancel any potentially running tasks first
                           if recording_timer_task and not recording_timer_task.done():
                                logging.warning("Client started countdown, cancelling existing recording timer.")
                                recording_timer_task.cancel()
                                recording_timer_task = None
                           if inference_task and not inference_task.done():
                                logging.warning("Client started countdown, cancelling existing inference task.")
                                inference_task.cancel()
                                inference_task = None
                           # Ensure active is false if starting countdown
                           if MASTER_STATE.get('inferencing_active', False):
                               cancel_active_patch = [{"op": "replace", "path": "/inferencing_active", "value": False}]
                               async with STATE_LOCK: MASTER_STATE['inferencing_active'] = False
                               await broadcast_patch(cancel_active_patch)
                               await send_to_stm32(cancel_active_patch)

                        elif new_inference_state == "recording":
                            # Cancel any existing timer (redundant safety)
                            if recording_timer_task and not recording_timer_task.done():
                                 recording_timer_task.cancel()
                                 recording_timer_task = None # Clear handle

                            # Check if main inference task is somehow running - should ideally not happen
                            if inference_task and not inference_task.done():
                                 logging.warning("Client triggered 'recording' but main inference task is running. Cancelling main task.")
                                 inference_task.cancel()
                                 inference_task = None

                            # Set active = True and start timer
                            active_patch = [{"op": "replace", "path": "/inferencing_active", "value": True}]
                            async with STATE_LOCK: MASTER_STATE['inferencing_active'] = True
                            await broadcast_patch(active_patch)
                            await send_to_stm32(active_patch)
                            # *** Assign to GLOBAL variable ***
                            recording_timer_task = asyncio.create_task(recording_timer_complete())

                        elif new_inference_state == "idle": # Client cancel
                             if recording_timer_task and not recording_timer_task.done():
                                  logging.info("Client set state to idle, cancelling recording timer.")
                                  recording_timer_task.cancel()
                                  recording_timer_task = None # Clear handle
                             if inference_task and not inference_task.done():
                                  logging.info("Client set state to idle, cancelling inference task.")
                                  inference_task.cancel()
                                  inference_task = None # Clear handle
                             if MASTER_STATE.get('inferencing_active', False):
                                 cancel_active_patch = [{"op": "replace", "path": "/inferencing_active", "value": False}]
                                 async with STATE_LOCK: MASTER_STATE['inferencing_active'] = False
                                 await broadcast_patch(cancel_active_patch)
                                 await send_to_stm32(cancel_active_patch)

            except json.JSONDecodeError: # ... (rest of error handling unchanged) ...
                logging.warning(f"Could not decode JSON from {client_addr}: {message}")
            except Exception as e: # ... (rest of error handling unchanged) ...
                logging.error(f"Unexpected error handling message from {client_addr}: {e}")

    except websockets.exceptions.ConnectionClosedOK: # ... (rest of connection closing unchanged) ...
        logging.info(f"Client disconnected gracefully: {client_addr}")
    except websockets.exceptions.ConnectionClosedError as e: # ... (rest of connection closing unchanged) ...
        logging.warning(f"Client connection closed with error: {client_addr} - {e}")
    except Exception as e: # ... (rest of connection closing unchanged) ...
        logging.error(f"Unexpected error in handler for {client_addr}: {e}", exc_info=True)
    finally: # ... (rest of connection closing unchanged) ...
        logging.info(f"Removing client: {client_addr}")
        CONNECTED_CLIENTS.remove(websocket)
        # If the last client disconnects, maybe cancel ongoing inference/recording?
        # if not CONNECTED_CLIENTS:
        #    if recording_timer_task and not recording_timer_task.done(): recording_timer_task.cancel()
        #    if inference_task and not inference_task.done(): inference_task.cancel()

# --- NEW: Separate Task for Recording Timer ---
async def recording_timer_complete():
    """Waits 6 seconds, then transitions state from 'recording' to 'inferencing'."""
    global MASTER_STATE, inference_task, recording_timer_task # Declare globals
    try:
        logging.info(f"Recording timer started ({RECORDING_DELAY_SECONDS}s)...")
        await asyncio.sleep(RECORDING_DELAY_SECONDS)

        logging.info("Recording timer finished.")
        # --- Prepare state transition patches ---
        state_transition_patch = []
        active_flag_patch = []
        solo_update_patch = None

        async with STATE_LOCK:
            # Only proceed if we are still in the 'recording' state
            if MASTER_STATE['inferencing_state'] == "recording":
                logging.info("Transitioning state from 'recording' to 'inferencing'")
                MASTER_STATE['inferencing_state'] = "inferencing"
                state_transition_patch.append({"op": "replace", "path": "/inferencing_state", "value": "inferencing"})

                # Set active flag to false
                if MASTER_STATE['inferencing_active']: # Should be true, but check
                    MASTER_STATE['inferencing_active'] = False
                    active_flag_patch.append({"op": "replace", "path": "/inferencing_active", "value": False})

                # Double check solo status (though unlikely to change here)
                current_soloing = MASTER_STATE['soloing_active']
                new_soloing = any(ch['soloed'] for ch in MASTER_STATE['channels'])
                if new_soloing != current_soloing:
                    MASTER_STATE['soloing_active'] = new_soloing
                    solo_update_patch = [{"op": "replace", "path": "/soloing_active", "value": new_soloing}]
            else:
                 logging.warning(f"Recording timer finished, but state was already '{MASTER_STATE['inferencing_state']}'. No action taken.")
                 recording_timer_task = None # Clear task handle
                 return # Exit if state changed (e.g., cancelled)

        # --- Broadcast and Send I2C (outside lock) ---
        if state_transition_patch: await broadcast_patch(state_transition_patch) # Notify UIs of state change
        if active_flag_patch:
             await broadcast_patch(active_flag_patch) # Notify ALL of active=false
             await send_to_stm32(active_flag_patch) # Send active=false to STM
        if solo_update_patch:
             await broadcast_patch(solo_update_patch)
             await send_to_stm32(solo_update_patch)

        # --- Trigger the main inference workflow ---
        # Check if another inference isn't somehow already running
        if inference_task and not inference_task.done():
             logging.warning("Recording timer finished, but inference task seems to be already running. Not starting another.")
        else:
             logging.info("Recording timer finished, starting main inference workflow task.")
             inference_task = asyncio.create_task(run_inference_workflow())

    except asyncio.CancelledError:
        logging.info("Recording timer task cancelled.")
        # No need to reset flags here, the cancelling code should handle it
        # Or the main workflow's finally block will catch it if state is weird
    except Exception as e:
        logging.error(f"Error in recording timer task: {e}")
        # Consider resetting state to idle here as a fallback?
    finally:
         recording_timer_task = None # Clear task handle


## --- Main Server Execution ---
async def main():
    # Start the dedicated I2C sender task
    sender_task = asyncio.create_task(i2c_sender_task())

    # Start the WebSocket server
    try:
        async with websockets.serve(handler, HOST, PORT):
            logging.info(f"WebSocket server started on ws://{HOST}:{PORT}")
            logging.info(f"I2C Communication {'ENABLED' if I2C_ENABLED else 'DISABLED'} using adapter type: {i2c_adapter._type if i2c_adapter else 'None'}")
            await asyncio.Future()  # Run forever
    finally:
        # Ensure sender task is cancelled when server stops
        logging.info("Shutting down I2C sender task...")
        if sender_task and not sender_task.done():
             sender_task.cancel()
             try:
                 await sender_task # Wait for cancellation to complete
             except asyncio.CancelledError:
                 logging.info("I2C sender task cancellation confirmed.")
             except Exception as e:
                 logging.error(f"Error awaiting sender task cancellation: {e}")


if __name__ == "__main__":
    i2c_cleanup_needed = False # Flag to track if adapter needs closing
    try:
        # Check if adapter was initialized before starting asyncio loop
        if i2c_adapter:
            i2c_cleanup_needed = True
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    except Exception as e:
        logging.error(f"Unhandled exception in main execution: {e}", exc_info=True)
    finally:
        # Cleanup I2C adapter ONLY if it was successfully initialized
        if i2c_cleanup_needed and i2c_adapter:
            logging.info("Closing I2C adapter...")
            i2c_adapter.close()
        logging.info("State manager shutdown complete.")