"""Tests for the refactored audio module."""

import tempfile
from pathlib import Path

import pytest
from pydub import AudioSegment

from src.audio import (
    VoiceInfo,
    VoiceProvider,
    clean_tts_text,
    export_audio,
    get_voice_model,
    get_voice_models,
    join_audio_segments,
    load_voices_from_json,
)


class TestVoiceLoading:
    """Tests for voice configuration loading."""

    def test_load_voices_from_json_default(self):
        """Test loading voices from default path."""
        voices = load_voices_from_json()
        assert isinstance(voices, dict)
        assert "fr-FR" in voices
        assert "en-GB" in voices

    def test_load_voices_from_json_custom_path(self):
        """Test loading voices from custom path."""
        voices_file = Path(__file__).parent.parent / "src" / "preferred_voices.json"
        voices = load_voices_from_json(voices_file)
        assert isinstance(voices, dict)

    def test_load_voices_from_json_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_voices_from_json(Path("/nonexistent/path.json"))

    def test_get_voice_model_valid(self):
        """Test getting a valid voice model."""
        voice = get_voice_model("fr-FR", "FEMALE", "flashcard")
        assert isinstance(voice, VoiceInfo)
        assert voice.language_code == "fr-FR"
        assert voice.provider in [
            VoiceProvider.GOOGLE,
            VoiceProvider.AZURE,
            VoiceProvider.ELEVENLABS,
        ]
        assert isinstance(voice.voice_id, str)

    def test_get_voice_model_invalid_language(self):
        """Test error for invalid language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            get_voice_model("xx-XX", "FEMALE", "flashcard")

    def test_get_voice_model_invalid_gender(self):
        """Test error for invalid gender."""
        with pytest.raises(ValueError, match="Gender"):
            get_voice_model("fr-FR", "NONEXISTENT", "flashcard")

    def test_get_voice_model_invalid_audio_type(self):
        """Test error for invalid audio type."""
        with pytest.raises(ValueError, match="Audio type"):
            get_voice_model("fr-FR", "FEMALE", "invalid_type")

    def test_get_voice_models_pair(self):
        """Test getting both male and female voices."""
        female, male = get_voice_models("fr-FR", "flashcard")
        assert isinstance(female, VoiceInfo)
        assert isinstance(male, VoiceInfo)
        assert female.language_code == "fr-FR"
        assert male.language_code == "fr-FR"

    def test_get_voice_model_with_preloaded_config(self):
        """Test getting voice with preloaded configuration."""
        voices = load_voices_from_json()
        voice = get_voice_model("fr-FR", "FEMALE", "flashcard", voices_config=voices)
        assert isinstance(voice, VoiceInfo)


class TestTextProcessing:
    """Tests for text processing utilities."""

    def test_clean_tts_text(self):
        """Test HTML entity cleaning."""
        assert clean_tts_text("Hello") == "Hello"
        assert clean_tts_text("It&#39;s") == "It's"
        assert clean_tts_text("&quot;test&quot;") == '"test"'


class TestAudioProcessing:
    """Tests for audio processing functions."""

    def test_join_audio_segments_single(self):
        """Test joining a single audio segment."""
        audio = AudioSegment.silent(duration=1000)
        result = join_audio_segments([audio])
        assert len(result) == 1000

    def test_join_audio_segments_multiple(self):
        """Test joining multiple audio segments."""
        audio1 = AudioSegment.silent(duration=1000)
        audio2 = AudioSegment.silent(duration=1000)
        result = join_audio_segments([audio1, audio2], gap_ms=100)
        # 1000 + 100 + 1000 = 2100
        assert len(result) == 2100

    def test_join_audio_segments_empty(self):
        """Test error for empty list."""
        with pytest.raises(ValueError):
            join_audio_segments([])

    def test_join_audio_segments_custom_gap(self):
        """Test custom gap duration."""
        audio = AudioSegment.silent(duration=1000)
        result = join_audio_segments([audio, audio], gap_ms=500)
        # 1000 + 500 + 1000 = 2500
        assert len(result) == 2500

    def test_export_audio_default_filename(self):
        """Test audio export with auto-generated filename."""
        audio = AudioSegment.silent(duration=1000)
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                filename = export_audio(audio)
                assert filename.endswith(".mp3")
                assert Path(filename).exists()
            finally:
                os.chdir(original_dir)

    def test_export_audio_custom_filename(self):
        """Test audio export with custom filename."""
        audio = AudioSegment.silent(duration=1000)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"
            result = export_audio(audio, filename=str(output_path))
            assert output_path.exists()
            assert result == str(output_path)


class TestVoiceInfo:
    """Tests for VoiceInfo dataclass."""

    def test_voice_info_creation(self):
        """Test creating a VoiceInfo instance."""
        voice = VoiceInfo(
            provider=VoiceProvider.GOOGLE,
            voice_id="en-US-Neural2-A",
            language_code="en-US",
        )
        assert voice.provider == VoiceProvider.GOOGLE
        assert voice.voice_id == "en-US-Neural2-A"
        assert voice.language_code == "en-US"

    def test_voice_provider_enum(self):
        """Test VoiceProvider enum values."""
        assert VoiceProvider.GOOGLE.value == "google"
        assert VoiceProvider.AZURE.value == "azure"
        assert VoiceProvider.ELEVENLABS.value == "elevenlabs"


class TestIntegration:
    """Integration tests for audio module."""

    def test_full_workflow_voice_loading(self):
        """Test complete workflow of loading voices and getting voice models."""
        voices = load_voices_from_json()
        assert len(voices) > 0

        for language_code in ["fr-FR", "en-GB", "de-DE"]:
            voice = get_voice_model(
                language_code, "FEMALE", "flashcard", voices_config=voices
            )
            assert voice.language_code == language_code
            assert isinstance(voice.provider, VoiceProvider)

    def test_full_workflow_audio_processing(self):
        """Test complete workflow of audio processing."""
        # Create some dummy audio
        audio1 = AudioSegment.silent(duration=500)
        audio2 = AudioSegment.silent(duration=500)

        # Join them
        joined = join_audio_segments([audio1, audio2], gap_ms=100)
        assert len(joined) == 1100

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp3"
            export_audio(joined, filename=str(output_path))
            assert output_path.exists()
