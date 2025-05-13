import unittest
import os
import torch
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

# Import the components to test from the app
from new_app import (
    Wav2VecIntent,
    process_audio,
    record_audio,
    load_model,
    LABELS
)

class TestWav2VecIntent(unittest.TestCase):
    """Test the Wav2VecIntent model architecture"""
    
    def setUp(self):
        # Mock the Wav2Vec2Model to avoid loading actual pretrained models
        self.wav2vec_mock_patcher = patch('new_app.Wav2Vec2Model')
        self.wav2vec_mock = self.wav2vec_mock_patcher.start()
        
        # Configure the mock to return an object with necessary attributes
        model_mock = MagicMock()
        model_mock.config.hidden_size = 768
        self.wav2vec_mock.from_pretrained.return_value = model_mock
        
        # Create model instance for testing
        self.model = Wav2VecIntent(num_classes=31, pretrained_model="test/model")
    
    def tearDown(self):
        self.wav2vec_mock_patcher.stop()
    
    def test_model_init(self):
        """Test model initialization and architecture"""
        # Check that model components exist
        self.assertIsNotNone(self.model.wav2vec)
        self.assertIsNotNone(self.model.layer_norm)
        self.assertIsNotNone(self.model.attention)
        self.assertIsNotNone(self.model.dropout)
        self.assertIsNotNone(self.model.fc)
        
        # Check output layer dimensions
        self.assertEqual(self.model.fc.out_features, 31)
    
    def test_forward_pass(self):
        """Test the model's forward pass"""
        # Mock the wav2vec output
        mock_hidden_states = torch.rand(2, 10, 768)  # [batch, sequence, hidden]
        self.model.wav2vec.return_value = MagicMock(last_hidden_state=mock_hidden_states, return_dict=True)
        
        # Create dummy input
        input_values = torch.rand(2, 16000)  # [batch, time]
        attention_mask = torch.ones(2, 16000)
        
        # Forward pass
        with patch.object(self.model.wav2vec, '__call__', 
                         return_value=MagicMock(last_hidden_state=mock_hidden_states)):
            output = self.model(input_values, attention_mask)
        
        # Check output dimensions
        self.assertEqual(output.shape, (2, 31))


class TestAudioProcessing(unittest.TestCase):
    """Test audio processing functions"""
    
    def setUp(self):
        # Create a temporary directory for test audio files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple test audio file
        self.test_audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        sample_rate = 16000
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        sf.write(self.test_audio_path, audio, sample_rate)
        
        # Mock the feature extractor
        self.feature_extractor_patcher = patch('new_app.Wav2Vec2FeatureExtractor')
        self.feature_extractor_mock = self.feature_extractor_patcher.start()
        
        # Configure mock to return proper input values
        extractor_instance = MagicMock()
        extractor_instance.return_value = MagicMock(input_values=torch.rand(1, 16000))
        self.feature_extractor_mock.from_pretrained.return_value = extractor_instance
        
        # Mock the model
        self.model_mock = MagicMock()
        self.model_mock.return_value = torch.rand(1, 31)  # Random logits for 31 classes
        
    def tearDown(self):
        self.feature_extractor_patcher.stop()
        self.temp_dir.cleanup()
    
    @patch('new_app.torchaudio.load')
    def test_process_audio(self, mock_load):
        """Test the audio processing function"""
        # Mock torchaudio.load to return our test waveform
        mock_waveform = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])  # Simple test waveform
        mock_load.return_value = (mock_waveform, 16000)
        
        # Mock the feature extractor
        mock_feature_extractor = MagicMock()
        mock_feature_inputs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_feature_extractor.return_value = MagicMock(input_values=mock_feature_inputs)
        
        # Mock the model
        mock_model = MagicMock()
        mock_outputs = torch.tensor([[0.1, 0.9, 0.1, 0.1]])  # Class 1 has highest probability
        mock_model.return_value = mock_outputs
        
        # Mock device
        mock_device = torch.device("cpu")
        
        # Patch the dependencies
        with patch('new_app.feature_extractor', mock_feature_extractor), \
             patch('new_app.model', mock_model), \
             patch('new_app.device', mock_device), \
             patch('new_app.torch.argmax', return_value=torch.tensor(1)), \
             patch('new_app.torch.nn.functional.softmax', return_value=torch.tensor([[0.1, 0.9, 0.1, 0.1]])): 
            
            # We're simplifying to just test if LABELS and softmax are processed properly
            # The test would be more complex for real audio files
            intent, confidence = process_audio(self.test_audio_path)
        
            # Check if a prediction was made
            self.assertIsNotNone(intent)
            self.assertIsNotNone(confidence)
            
            # Since actual predictions depend on the model's weights, we'll just check the types
            self.assertIsInstance(intent, str)
            self.assertIsInstance(confidence, float)

    @patch('new_app.sd.rec')
    @patch('new_app.sd.wait')
    def test_record_audio(self, mock_wait, mock_rec):
        """Test the audio recording function"""
        # Mock sounddevice recording to return dummy audio
        dummy_audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_rec.return_value = np.expand_dims(dummy_audio, 1)  # Add channel dimension
        
        # Mock streamlit components
        with patch('new_app.st.markdown'), \
             patch('new_app.st.progress') as mock_progress, \
             patch('new_app.st.error'):
            
            # Mock the progress bar
            mock_progress_bar = MagicMock()
            mock_progress.return_value = mock_progress_bar
            
            # Call function
            audio, sample_rate = record_audio(duration=0.5, sample_rate=16000)
            
            # Check if recording was attempted
            mock_rec.assert_called()
            mock_wait.assert_called()
            
            # Check if audio is returned with correct sample rate
            self.assertIsNotNone(audio)
            self.assertEqual(sample_rate, 16000)


class TestModelLoading(unittest.TestCase):
    """Test the model loading function"""
    
    def setUp(self):
        # Patch hugging face hub download
        self.hf_download_patcher = patch('new_app.hf_hub_download')
        self.mock_download = self.hf_download_patcher.start()
        self.mock_download.return_value = "mock_model_path.pt"
        
        # Patch torch.load
        self.torch_load_patcher = patch('new_app.torch.load')
        self.mock_torch_load = self.torch_load_patcher.start()
        self.mock_torch_load.return_value = {}  # Empty state dict
        
        # Patch Wav2Vec2Model
        self.wav2vec_model_patcher = patch('new_app.Wav2Vec2Model')
        self.mock_wav2vec_model = self.wav2vec_model_patcher.start()
        
        # Configure the mock to return an object with necessary attributes
        model_mock = MagicMock()
        model_mock.config.hidden_size = 768
        self.mock_wav2vec_model.from_pretrained.return_value = model_mock
        
        # Patch feature extractor
        self.feature_extractor_patcher = patch('new_app.Wav2Vec2FeatureExtractor')
        self.mock_feature_extractor = self.feature_extractor_patcher.start()
        
        # Patch streamlit
        self.st_patcher = patch('new_app.st')
        self.mock_st = self.st_patcher.start()
        
    def tearDown(self):
        self.hf_download_patcher.stop()
        self.torch_load_patcher.stop()
        self.wav2vec_model_patcher.stop()
        self.feature_extractor_patcher.stop()
        self.st_patcher.stop()
    
    def test_load_model_success(self):
        """Test successful model loading"""
        # Mock successful model loading
        mock_model_instance = MagicMock()
        with patch('new_app.Wav2VecIntent') as mock_model_class:
            mock_model_class.return_value = mock_model_instance
            
            # Call the function
            model, device, feature_extractor = load_model()
            
            # Check the function returns all three components
            self.assertIsNotNone(model)
            self.assertIsNotNone(device)
            self.assertIsNotNone(feature_extractor)
    
    def test_load_model_download_failure(self):
        """Test handling of download failure"""
        # Make the download fail
        self.mock_download.side_effect = Exception("Download failed")
        
        mock_model_instance = MagicMock()
        with patch('new_app.Wav2VecIntent') as mock_model_class:
            mock_model_class.return_value = mock_model_instance
            
            # Call the function
            model, device, feature_extractor = load_model()
            
            # Should still return a model but maybe no feature extractor
            self.assertIsNotNone(model)
            self.assertIsNotNone(device)
            self.assertIsNone(feature_extractor)
            
            # Should show error message
            self.mock_st.error.assert_called()
    
    def test_load_model_state_dict_failure(self):
        """Test handling of state dict loading failure"""
        # Make the state dict loading fail
        self.mock_torch_load.side_effect = Exception("Failed to load state dict")
        
        mock_model_instance = MagicMock()
        with patch('new_app.Wav2VecIntent') as mock_model_class:
            mock_model_class.return_value = mock_model_instance
            
            # Call the function
            model, device, feature_extractor = load_model()
            
            # Should still return components
            self.assertIsNotNone(model)
            self.assertIsNotNone(device)
            self.assertIsNotNone(feature_extractor)
            
            # Should show warning
            self.mock_st.warning.assert_called()


if __name__ == '__main__':
    unittest.main()