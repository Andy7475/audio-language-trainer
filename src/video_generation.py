import os
import tempfile
from typing import List, Tuple
from pydub import AudioSegment
from pydub.generators import Sine
from moviepy.editor import (
    TextClip,
    CompositeVideoClip,
    AudioFileClip,
    concatenate_videoclips,
)
import ffmpeg
from dotenv import load_dotenv
import json
import subprocess

load_dotenv()


def create_lesson_video(
    audio_text_pairs: List[Tuple[AudioSegment, str]], output_filename: str
):
    clips = []
    for audio, text in audio_text_pairs:
        # Export audio segment to a temporary file
        temp_audio_file = f"temp_audio_{id(audio)}.wav"
        audio.export(temp_audio_file, format="wav")

        # Create audio clip
        audio_clip = AudioFileClip(temp_audio_file)

        # Create text clip
        text_clip = (
            TextClip(
                text, fontsize=24, color="white", bg_color="black", size=(720, 480)
            )
            .set_position("center")
            .set_duration(audio_clip.duration)
        )

        # Combine text and audio
        video_clip = CompositeVideoClip([text_clip]).set_audio(audio_clip)
        clips.append(video_clip)

    # Concatenate all clips
    final_clip = concatenate_videoclips(clips)

    # Write the result to a file
    final_clip.write_videofile(output_filename, fps=24)

    # Clean up
    for clip in clips:
        clip.close()
    final_clip.close()
    for file in os.listdir():
        if file.startswith("temp_audio_") and file.endswith(".wav"):
            os.remove(file)


def join_video_files(
    video_files: List[str], chapter_titles: List[str], output_filename: str
):
    assert len(video_files) == len(
        chapter_titles
    ), "Number of video files and chapter titles must match"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a list of input files for ffmpeg
        list_file_path = os.path.join(temp_dir, "input_list.txt")
        with open(list_file_path, "w") as f:
            for video_file in video_files:
                f.write(f"file '{os.path.abspath(video_file)}'\n")

        # Create metadata file with chapters
        metadata_file_path = os.path.join(temp_dir, "metadata.txt")
        with open(metadata_file_path, "w") as m:
            m.write(";FFMETADATA1\n")
            start_time = 0
            for i, (video_file, title) in enumerate(zip(video_files, chapter_titles)):
                duration = float(
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1",
                            video_file,
                        ]
                    )
                    .decode()
                    .strip()
                )

                end_time = start_time + duration
                m.write(
                    f"\n[CHAPTER]\nTIMEBASE=1/1000000000\nSTART={int(start_time*1000000000)}\nEND={int(end_time*1000000000)}\ntitle={title}\n"
                )
                start_time = end_time

        # Run ffmpeg command to concatenate videos and add chapters
        command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file_path,
            "-i",
            metadata_file_path,
            "-map_metadata",
            "1",
            "-c",
            "copy",  # This copies both video and audio without re-encoding
            output_filename,
        ]

        subprocess.run(command, check=True)

    print(f"Video successfully created: {output_filename}")


# Example usage
if __name__ == "__main__":
    # Create some example audio segments and text for two lessons

    def generate_tone(frequency, duration_ms):

        # Generate the tone using Sine generator
        generator = Sine(frequency)
        sine_wave = generator.to_audio_segment(duration=duration_ms, volume=0)

        return sine_wave

    lesson1_audio_text_pairs = [
        (generate_tone(150, duration_ms=3000), "Lesson 1: First phrase"),
        (generate_tone(300, duration_ms=2000), "Lesson 1: Second phrase"),
    ]
    lesson2_audio_text_pairs = [
        (generate_tone(250, duration_ms=3000), "Lesson 2: First phrase"),
        (generate_tone(350, duration_ms=2000), "Lesson 2: Second phrase"),
    ]

    # Create individual lesson videos
    create_lesson_video(lesson1_audio_text_pairs, "lesson1.mp4")
    create_lesson_video(lesson2_audio_text_pairs, "lesson2.mp4")

    # Join lesson videos with chapters
    video_files = ["lesson1.mp4", "lesson2.mp4"]
    chapter_titles = ["Introduction to Greetings", "Basic Conversation"]
    join_video_files(video_files, chapter_titles, "full_course.mp4")
