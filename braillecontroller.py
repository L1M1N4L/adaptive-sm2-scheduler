import serial
import serial.tools.list_ports
import threading
import time
from queue import Queue

class BrailleHardwareController:
    """Manages bidirectional communication with Arduino Braille display"""
    
    def __init__(self):
        self.serial_port = None
        self.connected = False
        self.baud_rate = 9600
        self.rating_queue = Queue()
        self.show_hide_queue = Queue()
        self.read_thread = None
        self.running = False
        
    def connect(self, port_name=None):
        """Connect to Arduino with auto-detection"""
        try:
            if port_name is None:
                # Auto-detect Arduino (more robust detection)
                ports = serial.tools.list_ports.comports()
                for port in ports:
                    # Common Arduino identifiers
                    arduino_identifiers = ['arduino', 'ch340', 'ch341', 'ftdi', 'cp210']
                    port_info = f"{port.description} {port.manufacturer}".lower()
                    
                    if any(id in port_info for id in arduino_identifiers):
                        port_name = port.device
                        print(f"Found Arduino on {port_name}")
                        break
                
                # If still not found, try any available port
                if port_name is None and ports:
                    port_name = ports[0].device
                    print(f"Trying {port_name}")
            
            if port_name:
                self.serial_port = serial.Serial(
                    port=port_name,
                    baudrate=self.baud_rate,
                    timeout=1,
                    write_timeout=1
                )
                time.sleep(2)  # Wait for Arduino reset
                self.connected = True
                
                # Start thread to read button presses
                self.running = True
                self.read_thread = threading.Thread(target=self._read_serial, daemon=True)
                self.read_thread.start()
                
                # Send test character
                self.send_character('A')
                return True
        except Exception as e:
            print(f"Connection error: {e}")
        return False
    
    def _read_serial(self):
        """Read serial data in background thread"""
        while self.running and self.connected and self.serial_port:
            try:
                if self.serial_port.in_waiting > 0:
                    # Read a line
                    line = self.serial_port.readline().decode(errors='ignore').strip()
                    
                    if line.startswith("BTN:"):
                        try:
                            value = int(line[4:])  # Get number after "BTN:"
                            
                            if value == 99:  # Show/hide toggle button
                                print("Show/Hide button pressed on Arduino")
                                self.show_hide_queue.put(True)
                            elif value in [0, 3, 4, 5]:  # Rating buttons
                                self.rating_queue.put(value)
                                print(f"Received button rating: {value}")
                                
                        except ValueError:
                            print(f"Could not parse button value: {line}")
                    
                    # Also handle other serial messages (for debugging)
                    elif line:
                        print(f"Arduino: {line}")
                        
            except Exception as e:
                print(f"Read error: {e}")
                self.connected = False
                break
            time.sleep(0.01)

    
    def send_character(self, character):
        """Send character to Braille display"""
        if self.connected and self.serial_port:
            try:
                self.serial_port.write(character.encode())
                print(f"Sent character: {character}")
                return True
            except Exception as e:
                print(f"Send error: {e}")
                self.connected = False
        return False
    
    def clear_display(self):
        """Clear all solenoids"""
        if self.connected and self.serial_port:
            try:
                self.serial_port.write(b'0')  # Send '0' to clear
                print("Cleared display")
                return True
            except Exception as e:
                print(f"Clear error: {e}")
        return False
    
    def get_rating(self):
        """Get button rating from Arduino (non-blocking)"""
        try:
            return self.rating_queue.get_nowait()
        except:
            return None
        
    def get_show_hide(self):
        """Get show/hide toggle signal from Arduino (non-blocking)"""
        try:
            return self.show_hide_queue.get_nowait()
        except:
            return None
    
    def disconnect(self):
        """Close serial connection"""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=1)
        
        if self.serial_port:
            self.serial_port.close()
            self.connected = False