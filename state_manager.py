import asyncio
import json
import logging
import platform # For OS detection
import subprocess
import time
from copy import deepcopy
import struct

import websockets
import jsonpatch # For applying JSON patches RFC 6902

# --- Configuration ---
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8765       # WebSocket port
STM32_I2C_ADDR = 0x42 # Example I2C address for your STM32
INFERENCE_SCRIPT_PATH = "your_model_script.py" # Path to your PyTorch model script
INFERENCE_OUTPUT_PATH = "inference_output.json" # Where the model script saves results
INFERENCE_DELAY_SECONDS = 6
LOG_LEVEL = logging.INFO

I2C_COMMAND_DELAY_S = 0.002 # Delay in seconds between sending I2C commands (e.g., 2ms)


# --- Protocol ID Mappings (Python Side) ---

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
    "hw_init_ready": False,
}
CONNECTED_CLIENTS = set()
STATE_LOCK = asyncio.Lock()
inference_task = None



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


## --- AI Inference Workflow ---
async def run_inference_workflow():
    """Handles the AI auto-mixing process."""
    global MASTER_STATE, inference_task
    logging.info("Starting AI inference workflow...")

    try:
        # 1. Delay for audio capture
        await asyncio.sleep(INFERENCE_DELAY_SECONDS)

        # 2. Reset inferencing_active flag (apply, broadcast TO ALL, send I2C)
        logging.info(f"Inference capture time ({INFERENCE_DELAY_SECONDS}s) elapsed. Resetting flag.")
        reset_patch = [{"op": "replace", "path": "/inferencing_active", "value": False}]
        solo_active_update_patch = None # Track if soloing needs update too
        async with STATE_LOCK:
            try:
                if MASTER_STATE['inferencing_active']: # Only apply/broadcast if still true
                    jsonpatch.apply_patch(MASTER_STATE, reset_patch, in_place=True)
                    # Check if soloing_active needs recalculation after any state change
                    current_soloing = MASTER_STATE['soloing_active']
                    new_soloing = any(ch['soloed'] for ch in MASTER_STATE['channels'])
                    if new_soloing != current_soloing:
                        MASTER_STATE['soloing_active'] = new_soloing
                        solo_active_update_patch = [{"op": "replace", "path": "/soloing_active", "value": new_soloing}]

            except asyncio.CancelledError:
                 logging.info("Inference workflow cancelled before flag reset.")
                 raise # Re-raise cancellation
            except Exception as e:
                 logging.error(f"Error applying inference reset patch: {e}")

        # Broadcast reset flag TO ALL clients
        await broadcast_patch(reset_patch) # Uses helper, exclude_client=None
        # Broadcast soloing update if needed
        if solo_active_update_patch:
             await broadcast_patch(solo_active_update_patch)
        # Send reset flag to STM32
        await send_to_stm32(reset_patch)
        if solo_active_update_patch:
             await send_to_stm32(solo_active_update_patch)


        # 3. Run the model script
        # ... (subprocess logic unchanged) ...
        logging.info(f"Running inference script: {INFERENCE_SCRIPT_PATH}...")
        process = await asyncio.create_subprocess_exec(
            'python', INFERENCE_SCRIPT_PATH,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        # 4. Parse output, generate/apply/broadcast/send patch
        if process.returncode == 0:
            # ... (parsing logic unchanged) ...
            try:
                with open(INFERENCE_OUTPUT_PATH, 'r') as f: inferred_params = json.load(f)
                inference_patch_list = None
                solo_active_update_patch_inf = None
                async with STATE_LOCK:
                    # ... (make_patch logic unchanged) ...
                    inference_patch = jsonpatch.make_patch(MASTER_STATE, inferred_params)
                    inference_patch_list = inference_patch.patch
                    if inference_patch_list:
                        jsonpatch.apply_patch(MASTER_STATE, inference_patch_list, in_place=True)
                        # Check soloing_active again after inference patch
                        current_soloing = MASTER_STATE['soloing_active']
                        new_soloing = any(ch['soloed'] for ch in MASTER_STATE['channels'])
                        if new_soloing != current_soloing:
                            MASTER_STATE['soloing_active'] = new_soloing
                            solo_active_update_patch_inf = [{"op": "replace", "path": "/soloing_active", "value": new_soloing}]

                if inference_patch_list:
                    await broadcast_patch(inference_patch_list) # Broadcast TO ALL
                    await send_to_stm32(inference_patch_list)
                if solo_active_update_patch_inf:
                    await broadcast_patch(solo_active_update_patch_inf) # Broadcast TO ALL
                    await send_to_stm32(solo_active_update_patch_inf)

            except Exception as e: # ... (error handling unchanged) ...
                 logging.error(f"Error processing inference results: {e}")
        else: # ... (script failure handling unchanged) ...
            logging.error(f"Inference script failed...")

    except asyncio.CancelledError:
        logging.info("AI inference workflow cancelled.")
        # Ensure flag is false if cancelled mid-way
        if MASTER_STATE.get('inferencing_active', False):
             logging.warning("Workflow cancelled but flag might still be true. Forcing false.")
             cancel_patch = [{"op": "replace", "path": "/inferencing_active", "value": False}]
             async with STATE_LOCK: # Ensure state consistency
                 try: jsonpatch.apply_patch(MASTER_STATE, cancel_patch, in_place=True)
                 except Exception as e: logging.error(f"Error applying cancel patch: {e}")
             await broadcast_patch(cancel_patch)
             await send_to_stm32(cancel_patch)
             # Consider also checking/broadcasting soloing_active here if needed
        raise # Re-raise cancellation if needed higher up
    except Exception as e:
        logging.error(f"Unexpected error in inference workflow: {e}")
    finally:
        logging.info("AI inference workflow finished.")
        inference_task = None

## --- WebSocket Handler ---
async def handler(websocket, path):
    global MASTER_STATE, inference_task
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
                inference_triggered = False
                inference_cancelled_by_client = False

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

                                # Check for inference start/stop trigger from client
                                if op_path == "/inferencing_active":
                                    if op_value is True:
                                        inference_triggered = True
                                    else: # op_value is False
                                        inference_cancelled_by_client = True

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

                    # 5. Handle Inference Trigger / Cancellation
                    if inference_triggered:
                        if inference_task and not inference_task.done():
                            logging.warning("Inference already running, ignoring trigger.")
                        else:
                            logging.info("Inference triggered by client patch.")
                            inference_task = asyncio.create_task(run_inference_workflow())
                    elif inference_cancelled_by_client:
                        if inference_task and not inference_task.done():
                            logging.info("Inference manually cancelled by client.")
                            inference_task.cancel()
                            # Workflow's finally/except CancelledError handles state/broadcast
                        else:
                            logging.info("Client requested inference cancel, but no task was running.")
                            # Ensure flag is false if somehow client sent false without task running
                            if MASTER_STATE.get('inferencing_active', False):
                                 cancel_patch = [{"op": "replace", "path": "/inferencing_active", "value": False}]
                                 async with STATE_LOCK: jsonpatch.apply_patch(MASTER_STATE, cancel_patch, in_place=True)
                                 await broadcast_patch(cancel_patch, exclude_client=websocket) # Inform others
                                 send_to_stm32(cancel_patch)

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