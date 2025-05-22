import unittest
import os
import tempfile
from pydub import AudioSegment

# Assuming audioprocessing package is installed or in PYTHONPATH
try:
    from audioprocessing.resampling import resample_audio_file
except ImportError:
    # This allows tests to be run even if the package is not formally installed,
    # for example, when testing directly from the project directory structure.
    # It requires that the project root is in sys.path.
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from audioprocessing.resampling import resample_audio_file

class TestResampling(unittest.TestCase):
    """Test suite for audioprocessing.resampling module."""

    def test_resample_audio_file_samplerate(self):
        """Test the resample_audio_file function for correct sample rate change."""
        original_sr = 44100
        target_sr = 22050
        duration_ms = 1000  # 1 second

        # Create a dummy WAV file
        silence = AudioSegment.silent(duration=duration_ms, frame_rate=original_sr)

        # Use temporary files that are deleted manually for more control
        # Input file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in_file:
            tmp_in_path = tmp_in_file.name
        
        # Output file
        # We need a .wav suffix for pydub to correctly infer format on export within resample_audio_file
        # and on load for verification.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out_file:
            tmp_out_path = tmp_out_file.name

        try:
            silence.export(tmp_in_path, format="wav")

            # Call the function to be tested
            success = resample_audio_file(
                input_path=tmp_in_path,
                output_path=tmp_out_path,
                target_samplerate=target_sr,
                audio_format="wav" # Ensure WAV format is specified
            )
            self.assertTrue(success, "resample_audio_file returned False, indicating failure.")

            # Load the resampled audio file to check its properties
            self.assertTrue(os.path.exists(tmp_out_path), "Output file was not created.")
            self.assertGreater(os.path.getsize(tmp_out_path), 0, "Output file is empty.")
            
            resampled_audio = AudioSegment.from_file(tmp_out_path)
            
            # Assert that the sample rate is correct
            self.assertEqual(resampled_audio.frame_rate, target_sr,
                             f"Expected sample rate {target_sr}Hz, but got {resampled_audio.frame_rate}Hz.")
            
            # Optional: Check if duration is approximately maintained (within a small tolerance)
            # Resampling can sometimes slightly alter duration due to frame rounding.
            self.assertAlmostEqual(len(resampled_audio), duration_ms, delta=50, # Allow 50ms delta
                                   msg=f"Duration changed significantly after resampling. Original: {duration_ms}ms, Resampled: {len(resampled_audio)}ms")

        finally:
            # Clean up temporary files
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)

if __name__ == '__main__':
    unittest.main()
