import unittest
import os
import tempfile
import csv
from pydub import AudioSegment

# Assuming audioprocessing package is installed or in PYTHONPATH
try:
    from audioprocessing.io_utils import get_audio_duration, export_dict_to_csv
except ImportError:
    # This allows tests to be run even if the package is not formally installed,
    # for example, when testing directly from the project directory structure.
    # It requires that the project root is in sys.path.
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from audioprocessing.io_utils import get_audio_duration, export_dict_to_csv


class TestIoUtils(unittest.TestCase):
    """Test suite for audioprocessing.io_utils module."""

    def test_get_audio_duration(self):
        """Test the get_audio_duration function."""
        # Create a dummy WAV file (1.5 seconds of silence at 44.1kHz)
        duration_ms = 1500
        sample_rate = 44100
        silence = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)

        # Use a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
            tmp_wav_path = tmp_wav_file.name
        
        try:
            silence.export(tmp_wav_path, format="wav")
            
            # Call get_audio_duration
            calculated_duration = get_audio_duration(tmp_wav_path)
            
            # Assert that the duration is correct
            self.assertEqual(calculated_duration, duration_ms, 
                             f"Expected duration {duration_ms}ms, but got {calculated_duration}ms.")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

    def test_export_dict_to_csv(self):
        """Test the export_dict_to_csv function."""
        sample_data = {
            "file1.wav": [1000, 4.5],
            "file2.wav": [2000, 3.9],
            "file3.wav": [1500, "Error"] # Test with string value as well
        }
        headers = ["AudioPath", "Length", "Score"]
        
        # Use NamedTemporaryFile to get a temporary CSV file path
        # delete=False is important on some OS (like Windows) to allow reading the file after writing
        # before it's closed. We'll manually close and it will be deleted.
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False, newline='') as tmp_csv_file_obj:
            tmp_csv_path = tmp_csv_file_obj.name

        try:
            export_dict_to_csv(sample_data, tmp_csv_path, headers=headers)

            # Read the content of the temporary CSV file to verify
            with open(tmp_csv_path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter='\t') # Assuming tab delimiter as per io_utils
                read_headers = next(reader)
                self.assertEqual(read_headers, headers, "CSV headers do not match.")
                
                read_data = list(reader)
                expected_data = [
                    ["file1.wav", "1000", "4.5"],
                    ["file2.wav", "2000", "3.9"],
                    ["file3.wav", "1500", "Error"]
                ]
                self.assertEqual(read_data, expected_data, "CSV data rows do not match.")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_csv_path):
                os.remove(tmp_csv_path)

if __name__ == '__main__':
    unittest.main()
