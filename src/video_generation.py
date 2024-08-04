import os
from typing import List, Tuple
from pydub import AudioSegment
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips
import ffmpeg

def create_lesson_video(audio_text_pairs: List[Tuple[AudioSegment, str]], output_filename: str):
    clips = []
    for audio, text in audio_text_pairs:
        # Export audio segment to a temporary file
        temp_audio_file = f"temp_audio_{id(audio)}.wav"
        audio.export(temp_audio_file, format="wav")
        
        # Create audio clip
        audio_clip = AudioFileClip(temp_audio_file)
        
        # Create text clip
        text_clip = (TextClip(text, fontsize=24, color='white', bg_color='black', size=(720, 480))
                     .set_position('center')
                     .set_duration(audio_clip.duration))
        
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

def join_video_files(video_files: List[str], chapter_titles: List[str], output_filename: str):
    # Ensure we have the same number of titles as videos
    assert len(video_files) == len(chapter_titles), "Number of video files and chapter titles must match"

    # Create a list to hold the input arguments for ffmpeg
    inputs = []
    for video_file in video_files:
        inputs.extend(['-i', video_file])

    # Create the filter complex string for concatenation
    filter_complex = f'concat=n={len(video_files)}:v=1:a=1 [outv] [outa]'

    # Prepare the chapter metadata
    metadata = ";FFMETADATA1\n"
    current_time = 0
    for i, (video_file, title) in enumerate(zip(video_files, chapter_titles)):
        duration = float(ffmpeg.probe(video_file)['streams'][0]['duration'])
        metadata += f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(current_time * 1000)}\n"
        current_time += duration
        metadata += f"END={int(current_time * 1000)}\ntitle={title}\n"

    with open("chapters.txt", "w") as f:
        f.write(metadata)

    # Run ffmpeg command
    (
        ffmpeg
        .input(*inputs)
        .filter_complex(filter_complex)
        .output(output_filename, map=['[outv]', '[outa]'], map_metadata="chapters.txt")
        .global_args('-i', 'chapters.txt')
        .overwrite_output()
        .run()
    )

    # Clean up
    os.remove("chapters.txt")

# Example usage
if __name__ == "__main__":
    # Create some example audio segments and text for two lessons
    lesson1_audio_text_pairs = [
        (AudioSegment.silent(duration=3000), "Lesson 1: First phrase"),
        (AudioSegment.silent(duration=2000), "Lesson 1: Second phrase"),
    ]
    lesson2_audio_text_pairs = [
        (AudioSegment.silent(duration=2500), "Lesson 2: First phrase"),
        (AudioSegment.silent(duration=3500), "Lesson 2: Second phrase"),
    ]

    # Create individual lesson videos
    create_lesson_video(lesson1_audio_text_pairs, "lesson1.mp4")
    create_lesson_video(lesson2_audio_text_pairs, "lesson2.mp4")

    # Join lesson videos with chapters
    video_files = ["lesson1.mp4", "lesson2.mp4"]
    chapter_titles = ["Introduction to Greetings", "Basic Conversation"]
    join_video_files(video_files, chapter_titles, "full_course.mp4")

    # Clean up individual lesson videos
    for file in video_files:
        os.remove(file)