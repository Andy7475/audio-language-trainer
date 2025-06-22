#!/usr/bin/env python3
"""
Flashcard Tutorial Generator

Creates a single-file HTML tutorial for explaining how to use FirePhrase flashcards.
This tutorial uses a card-based navigation system and includes live examples of the flashcards.
"""

import os
from typing import Dict, Any, List, Optional

# Import the necessary functions from your modules
try:
    from src.template_testing import (
        build_phrase_dict_from_gcs,
        image_to_base64_html,
        audio_to_base64_html,
    )
    from src.config_loader import config
    from src.utils import load_template
    from src.gcs_storage import upload_to_gcs, get_tutorial_path
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure you're running from the project root directory")


def generate_flashcard_tutorial(
    phrase_key: str = "the_cake_tastes_delicious",
    collection: str = "WarmUp150",
    bucket_name: str = None,
    language: Optional[str] = None,
) -> str:
    """
    Generate a complete flashcard tutorial as a single HTML file.

    Args:
        phrase_key: The example phrase to use for demonstrations
        collection: Collection to pull the example from
        bucket_name: GCS bucket name (uses config default if None)
        output_file: Local output file name
        upload_to_gcs_bucket: Whether to upload to GCS public bucket (default: True)
        language: Target language (defaults to config.TARGET_LANGUAGE_NAME)

    Returns:
        str: Path to the generated HTML file (local path) or GCS URI if uploaded
    """

    # Tutorial card configuration - easily editable
    tutorial_cards = [
        {
            "type": "welcome",
            "title": "How FirePhrase Transforms Language Learning",
            "content": """
                <div class="welcome-content">
                    <h3>FirePhrase uses a phrase-based approach developed by researcher Michael Lewis. Instead of memorizing individual words, you learn complete meaningful phrases that your brain naturally mixes and matches to create new sentences.</h3>
                    
                    <div class="feature-list">
                        <div class="feature-item">
                            <span class="feature-icon">🧠</span>
                            <div>
                                <strong>Learn Phrases, Not Words</strong>
                                <p>When you learn phrases, your brain automatically picks up patterns and starts creating new combinations on its own</p>
                            </div>
                        </div>
                        
                        <div class="feature-item">
                            <span class="feature-icon">🎯</span>
                            <div>
                                <strong>Real Language from Native Speakers</strong>
                                <p>Every phrase comes from how native speakers actually talk, not from textbooks</p>
                            </div>
                        </div>
                        
                        <div class="feature-item">
                            <span class="feature-icon">🔄</span>
                            <div>
                                <strong>How Natives Actually Learn</strong>
                                <p>This copies how native speakers learned their language - through meaningful chunks, not grammar rules</p>
                            </div>
                        </div>
                    </div>
                    
                    <p><strong>The result: After learning hundreds of phrases, your brain automatically starts creating completely new sentences by mixing and matching the patterns you've absorbed.</strong></p>
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Why Images + Audio Work So Well",
            "content": """
                <div class="info-content">
                    <h3>FirePhrase combines pictures, sound, and context because research shows that using multiple senses helps you remember much better than just reading text.</h3>
                    
                    <div class="card-type-grid">
                        <div class="card-type-item">
                            <h3>👁️ Pictures Help Memory</h3>
                            <p>Your brain remembers words much better when they're paired with images instead of just text alone.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>👂 Normal and Slow Speaker Audio</h3>
                            <p>Hearing high quality audio helps you learn proper pronunciation and recognize speech patterns.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>🔗 Situational Context</h3>
                            <p>Seeing when and where phrases are used helps you remember them and use them correctly.</p>
                        </div>
                    </div>
                    
                    <p><strong>This approach works especially well because most natural speech is made up of common phrases and chunks that people use over and over.</strong></p>
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Don't Worry - You're Not Expected to Know Everything",
            "content": """
                <div class="info-content">
                    <h3>Flashcard systems are designed for learning, not testing. Not knowing an answer at first is completely normal and actually helps you learn better.</h3>
                    
                    <div class="card-type-grid">
                        <div class="card-type-item">
                            <h3>🔬 Testing Helps Learning</h3>
                            <p>Studies show that trying to recall something (even when you get it wrong) actually strengthens your memory.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>🧠 Struggling Helps</h3>
                            <p>Having to think hard about an answer creates stronger memories than when things are too easy.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>📈 The System Adapts</h3>
                            <p>The app learns from your answers and shows difficult cards more often, easy cards less often.</p>
                        </div>
                    </div>
                    
                    <p><strong>Focus on getting lots of exposure and practice rather than perfect recall right away.</strong></p>
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Speaking Practice Comes First",
            "content": """
                <div class="info-content">
                    <h3>Research shows that actually trying to speak creates stronger memories than just reading or listening.</h3>
                    
                    <div class="card-type-grid">
                        <div class="card-type-item">
                            <h3>🗣️ Speaking Builds Memory</h3>
                            <p>When you speak, your brain creates stronger connections that help you remember better and speak more fluently.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>⚡ Active Recall Works</h3>
                            <p>Trying to remember and say something builds stronger memory pathways than just reading it.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>🎯 Practice for Real Conversations</h3>
                            <p>Speaking practice builds the automatic responses you need for real-time conversation.</p>
                        </div>
                    </div>
                    
                    <p><strong>Start with Speaking Cards to build your speaking skills from day one.</strong></p>
                </div>
            """,
        },
        {
            "type": "example_speaking_front",
            "title": "Speaking Card Practice",
            "content": """
                <div class="example-explanation">
                    <p><strong>Practice with this Speaking Card example:</strong></p>
                    <ul>
                        <li>Read the English text below</li>
                        <li>Try to say it in your target language before checking</li>
                        <li>Focus on meaning, not word-for-word translation</li>
                        <li>Natural expressions often differ significantly from literal translations</li>
                    </ul>
                </div>
                <div class="example-card">
                    EXAMPLE_SPEAKING_FRONT_PLACEHOLDER
                </div>
            """,
        },
        {
            "type": "example_back",
            "title": "Universal Answer Side - All Cards Look Like This! 🔄",
            "content": """
                <div class="example-explanation">
                    <p><strong>Every card type has the same answer side with these powerful features:</strong></p>
                    <ul>
                        <li><strong>Audio buttons:</strong> Normal speed and slow speed for pronunciation practice</li>
                        <li><strong>Wiktionary links:</strong> Click any word to see detailed definitions and grammar</li>
                        <li><strong>Learning Insights:</strong> AI-powered explanations copied to your clipboard</li>
                        <li><strong>Copy text:</strong> Click text to copy phrases for further study</li>
                        <li><strong>Anki buttons:</strong> Rate your performance to optimize the learning algorithm</li>
                    </ul>
                </div>
                <div class="example-card">
                    EXAMPLE_BACK_PLACEHOLDER
                    <div class="anki-buttons">
                        <button class="anki-button again" data-tooltip="Didn't know it or got it wrong - show again soon">Again</button>
                        <button class="anki-button hard" data-tooltip="Got it right but took more than 10 seconds">Hard</button>
                        <button class="anki-button good" data-tooltip="Got it right within a few seconds - most common choice">Good</button>
                        <button class="anki-button easy" data-tooltip="Knew it immediately without hesitation">Easy</button>
                    </div>
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Master the Anki Buttons",
            "content": """
                <div class="info-content">
                    <h3>Use these buttons to tell the app how well you knew each phrase:</h3>
                    
                    <div class="card-type-grid">
                        <div class="card-type-item">
                            <h3>❌ "Again"</h3>
                            <p>Didn't know it or got it wrong. You'll see this card again soon.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>⏱️ "Hard"</h3>
                            <p>Got it right but took more than 10 seconds to remember.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>✅ "Good"</h3>
                            <p>Got it right within a few seconds - this is your most common choice.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>⚡ "Easy"</h3>
                            <p>Knew it immediately without any hesitation.</p>
                        </div>
                    </div>
                    
                    <p><strong>Being honest about how well you knew each card helps the app show you the right cards at the right time.</strong></p>
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Learning Insights: Your AI Tutor 🤖",
            "content": """
                <div class="info-content">
                    <h3>Click "Learning Insights" for detailed explanations powered by AI</h3>
                    
                    <div class="card-type-grid">
                        <div class="card-type-item">
                            <h3>📋 Auto-Copied Prompt</h3>
                            <p>A detailed prompt about the phrase is automatically copied to your clipboard when you click the button.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>Learning Insights Opens Claude.ai</h3>
                            <p>The button redirects you to Claude.ai where you can paste the prompt and get detailed explanations.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>🔄 Use Any AI</h3>
                            <p>Prefer ChatGPT or another AI? Just paste the clipboard prompt into your preferred AI tool instead!</p>
                        </div>
                    </div>
                    
                    <p><strong>What you'll learn:</strong> Grammar explanations, cultural context, usage examples, and why the phrase is constructed that way.</p>
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Other Card Types: Reading & Listening 📚🎧",
            "content": """
                <div class="info-content">
                    <h3>Now that you've mastered Speaking cards, let's explore the other two types:</h3>
                    
                    <div class="card-type-grid">
                        <div class="card-type-item">
                            <h3>📖 Reading Cards</h3>
                            <p>See the target language text first. Try to understand the meaning, then reveal the English translation.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>🎧 Listening Cards</h3>
                            <p>Hear the audio first without seeing text. Try to understand what's being said, then reveal everything.</p>
                        </div>
                        <div class="card-type-item">
                            <h3>🔄 Same Back Side</h3>
                            <p>All three card types flip to the same back side you saw earlier, with all the same powerful features.</p>
                        </div>
                    </div>
                    
                    <p><strong>Pro tip:</strong> Use all three types together for complete mastery - comprehension AND production!</p>
                </div>
            """,
        },
        {
            "type": "example_reading_front",
            "title": "Try This: Reading Card 📖",
            "content": """
                <div class="example-explanation">
                    <p><strong>Practice with this Reading Card:</strong></p>
                    <ul>
                        <li>You see the target language text below</li>
                        <li>Try to understand what it means</li>
                        <li>Can you work out the missing word?</li>
                        <li>Click the covered up word to reveal it</li>
                        <li>Click <strong>"Reveal English"</strong> to help</li>
                        <li>Use the image for context clues</li>
                    </ul>
                </div>
                <div class="example-card">
                    EXAMPLE_READING_FRONT_PLACEHOLDER
                </div>
            """,
        },
        {
            "type": "example_listening_front",
            "title": "Try This: Listening Card 🎧",
            "content": """
                <div class="example-explanation">
                    <p><strong>Practice with this Listening Card:</strong></p>
                    <ul>
                        <li>Click the audio button to hear the phrase</li>
                        <li>Try to understand what's being said</li>
                        <li>Click <strong>"Reveal Image"</strong> for visual context as a hint</li>
                        <li>Then check your understanding by flipping the card</li>
                    </ul>
                </div>
                <div class="example-card">
                    EXAMPLE_LISTENING_FRONT_PLACEHOLDER
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Ready for Real Stories? 📖✨",
            "content": """
                <div class="info-content">
                    <h3>Here's where FirePhrase gets really exciting - your flashcard vocabulary comes alive in real stories</h3>
                    
                    <div class="card-type-grid">
                        <div class="card-type-item">
                            <h3>🎯 80% Recognition</h3>
                            <p>Every story uses vocabulary from your deck - giving you about 80% word recognition</p>
                        </div>
                        <div class="card-type-item">
                            <h3>🔄 New Contexts</h3>
                            <p>See your familiar words in fresh situations - this is how you develop true fluency and flexibility</p>
                        </div>
                        <div class="card-type-item">
                            <h3>👂 Listening Skills</h3>
                            <p>Practice picking out words you know from continuous speech - a crucial real-world conversation skill</p>
                        </div>
                    </div>
                    
                    <div class="story-features">
                        <div class="story-feature">
                            <span class="story-icon">🎧</span>
                            <div>
                                <strong>Normal Speed Audio</strong>
                                <p>Listen at natural pace while following along with the text to build comprehension</p>
                            </div>
                        </div>
                        
                        <div class="story-feature">
                            <span class="story-icon">⚡</span>
                            <div>
                                <strong>Fast Audio Challenge</strong>
                                <p>Once familiar, try the fast version. This trains your brain's word segmentation - how natives separate words in rapid speech</p>
                            </div>
                        </div>
                    </div>
                    
                    <p><strong>The magic moment:</strong> When you realize you're understanding a whole story using words you learned as individual phrases!</p>
                    
                    <div class="story-cta">
                        <p><strong>Ready to see a story?</strong></p>
                        <a href="https://storage.googleapis.com/audio-language-trainer-stories/index.html" target="_blank" rel="noopener" class="collection-link">Explore FirePhrase Stories →</a>
                    </div>
                </div>
            """,
        },
        {
            "type": "text",
            "title": "Getting More Decks 🛒",
            "content": """
                <div class="shop-content">
                    <h3>Ready for more language learning content?</h3>
                    
                    <div class="shop-collections">
                        <div class="collection-item">
                            <span class="collection-icon">🔥</span>
                            <div>
                                <strong>Warm Up Collection</strong>
                                <p>Perfect for beginners! Shorter phrases to build your confidence and get you speaking quickly.</p>
                                <a href="https://firephrase.co.uk/collections/warm-up" target="_blank" rel="noopener" class="collection-link">Explore Warm Up →</a>
                            </div>
                        </div>
                        
                        <div class="collection-item">
                            <span class="collection-icon">🚀</span>
                            <div>
                                <strong>First1000 Collection</strong>
                                <p>Our flagship course covering 1000+ common words so you can understand native content!</p>
                                <a href="https://firephrase.co.uk/collections/complete" target="_blank" rel="noopener" class="collection-link">Explore First1000 →</a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="free-sample">
                        <p><strong>Want to try before you buy?</strong></p>
                        <p>Get the first pack of each collection completely free!</p>
                        <a href="https://firephrase.co.uk/collections/free" target="_blank" rel="noopener" class="free-link">Get Free Samples →</a>
                    </div>
                </div>
            """,
        },
        {
            "type": "conclusion",
            "title": "You're Ready to Go! 🚀",
            "content": """
                <div class="conclusion-content">
                    <h3>Congratulations! You now know how to use FirePhrase flashcards effectively.</h3>
                    
                    <div class="conclusion-summary">
                        <p><strong>Remember:</strong></p>
                        <ul>
                            <li>🎧 <strong>Listening Cards</strong> - Hear first, then reveal</li>
                            <li>📖 <strong>Reading Cards</strong> - Read and comprehend, then check</li>
                            <li>🗣️ <strong>Speaking Cards</strong> - Produce the language, then verify</li>
                        </ul>
                    </div>
                    
                    <div class="final-cta">
                        <p>Start your language learning journey with consistent daily practice!</p>
                        <p><strong>Happy learning! 🔥</strong></p>
                    </div>
                </div>
            """,
        },
    ]

    # Set default language if not provided
    if language is None:
        language = config.TARGET_LANGUAGE_NAME

    # Try to get phrase data for examples
    example_cards = {}
    try:
        phrase_dict = build_phrase_dict_from_gcs(
            collection=collection,
            bucket_name=bucket_name,
            phrase_keys=[phrase_key],
            language=language,
        )

        if phrase_dict and phrase_key in phrase_dict:
            phrase_data = phrase_dict[phrase_key]
            example_cards = generate_example_card_content(phrase_data, language)
            print(f"✅ Loaded example phrase data for: {phrase_key}")
        else:
            print(
                f"⚠️  Could not find phrase with key '{phrase_key}' - tutorial will have placeholder content"
            )

    except Exception as e:
        print(f"⚠️  Could not load example phrase data: {str(e)}")
        print("Tutorial will be generated with placeholder content")

    # Generate the complete HTML
    html_content = generate_tutorial_html(tutorial_cards, example_cards)

    # Upload to GCS or save locally
    try:
        # Upload to GCS (which also creates local copy)
        gcs_bucket = config.GCS_PUBLIC_BUCKET
        tutorial_path = get_tutorial_path(language)
        gcs_uri = upload_to_gcs(
            obj=html_content,
            bucket_name=gcs_bucket,
            file_name=tutorial_path,
            content_type="text/html",
            save_local=True,  # This creates local copy automatically
            local_base_dir="outputs/gcs",
        )

        # Generate public URL
        public_url = gcs_uri.replace("gs://", "https://storage.googleapis.com/")

        print(f"✅ Flashcard tutorial generated and uploaded to GCS")
        print(f"🌐 GCS URI: {gcs_uri}")
        print(f"🔗 Public URL: {public_url}")

        return gcs_uri

    except Exception as e:
        print(f"❌ Error generating tutorial: {e}")
        return None


def generate_flashcard_tutorials_batch(
    languages: List[str],
    phrase_key: str = "the_cake_tastes_delicious",
    collection: str = "WarmUp150",
    bucket_name: str = None,
) -> Dict[str, str]:
    """
    Generate flashcard tutorials for multiple languages.

    Args:
        languages: List of language names to generate tutorials for
        phrase_key: The example phrase to use for demonstrations
        collection: Collection to pull the example from
        bucket_name: GCS bucket name (uses config default if None)

    Returns:
        Dict[str, str]: Dictionary mapping language names to their generated tutorial paths/URIs
    """
    results = {}

    print(f"🚀 Generating tutorials for {len(languages)} languages...")

    for i, language in enumerate(languages, 1):
        print(f"\n📝 [{i}/{len(languages)}] Generating tutorial for {language}...")

        try:
            result = generate_flashcard_tutorial(
                phrase_key=phrase_key,
                collection=collection,
                bucket_name=bucket_name,
                language=language,
            )

            if result:
                results[language] = result
                print(f"✅ {language} tutorial completed: {result}")
            else:
                results[language] = None
                print(f"❌ {language} tutorial failed")

        except Exception as e:
            results[language] = None
            print(f"❌ {language} tutorial failed with error: {str(e)}")

    # Summary
    successful = sum(1 for result in results.values() if result)
    failed = len(results) - successful

    print(f"\n🎯 Batch Generation Complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    if successful > 0:
        print(f"\n🌐 Generated tutorials:")
        for lang, result in results.items():
            if result:
                public_url = result.replace("gs://", "https://storage.googleapis.com/")
                print(f"  {lang}: {public_url}")

    return results


def generate_example_card_content(
    phrase_data: Dict[str, Any], language: str
) -> Dict[str, str]:
    """Generate the example card HTML content."""

    try:
        # Convert media to base64
        picture_html = (
            image_to_base64_html(phrase_data["image"]) if phrase_data["image"] else ""
        )
        target_audio_html = (
            audio_to_base64_html(phrase_data["audio_normal"], "normal")
            if phrase_data["audio_normal"]
            else ""
        )
        target_audio_slow_html = (
            audio_to_base64_html(phrase_data["audio_slow"], "slow")
            if phrase_data["audio_slow"]
            else ""
        )

        english_text = phrase_data["english_text"]
        target_text = phrase_data["target_text"]
        wiktionary_links = phrase_data.get("wiktionary_links", "")
        # Add target="_blank" to wiktionary links for tutorial
        if wiktionary_links:
            wiktionary_links = wiktionary_links.replace(
                '">', '" target="_blank" rel="noopener">'
            )

        # Load templates
        back_template = load_template("card_back_template.html", "src/templates")
        listening_front_template = load_template(
            "listening_card_front_template.html", "src/templates"
        )
        reading_front_template = load_template(
            "reading_card_front_template.html", "src/templates"
        )
        speaking_front_template = load_template(
            "speaking_card_front_template.html", "src/templates"
        )

        example_cards = {}

        # Generate listening front with unique IDs
        listening_front = listening_front_template
        # Make IDs unique for tutorial
        listening_front = listening_front.replace(
            'id="imageFlip"', 'id="tutorial-listening-imageFlip"'
        )

        for placeholder, value in [
            ("{{Picture}}", picture_html),
            ("{{TargetText}}", target_text),
            ("{{EnglishText}}", english_text),
            ("{{TargetAudio}}", target_audio_html),
            ("{{TargetAudioSlow}}", target_audio_slow_html),
        ]:
            listening_front = listening_front.replace(placeholder, value)
        example_cards["listening_front"] = listening_front

        # Generate reading front with unique IDs
        reading_front = reading_front_template
        # Make IDs unique for tutorial
        reading_front = reading_front.replace(
            'id="englishFlip"', 'id="tutorial-reading-englishFlip"'
        )
        reading_front = reading_front.replace(
            'id="target-text-container"', 'id="tutorial-reading-target-text-container"'
        )

        for placeholder, value in [
            ("{{Picture}}", picture_html),
            ("{{TargetText}}", target_text),
            ("{{EnglishText}}", english_text),
            ("{{TargetAudio}}", target_audio_html),
            ("{{TargetAudioSlow}}", target_audio_slow_html),
        ]:
            reading_front = reading_front.replace(placeholder, value)
        example_cards["reading_front"] = reading_front

        # Generate speaking front
        speaking_front = speaking_front_template
        for placeholder, value in [
            ("{{Picture}}", picture_html),
            ("{{TargetText}}", target_text),
            ("{{EnglishText}}", english_text),
            ("{{TargetAudio}}", target_audio_html),
            ("{{TargetAudioSlow}}", target_audio_slow_html),
        ]:
            speaking_front = speaking_front.replace(placeholder, value)
        example_cards["speaking_front"] = speaking_front

        # Generate back template (used for all card types)
        back_content = back_template
        for placeholder, value in [
            ("{{Picture}}", picture_html),
            ("{{TargetText}}", target_text),
            ("{{EnglishText}}", english_text),
            ("{{TargetAudio}}", target_audio_html),
            ("{{TargetAudioSlow}}", target_audio_slow_html),
            ("{{WiktionaryLinks}}", wiktionary_links),
            ("{{TargetLanguageName}}", language.lower()),
            ("{{Tags}}", "tutorial_example"),
        ]:
            back_content = back_content.replace(placeholder, value)
        example_cards["back"] = back_content

        return example_cards

    except Exception as e:
        print(f"Error generating example cards: {e}")
        return {}


def generate_tutorial_html(
    tutorial_cards: List[Dict[str, str]], example_cards: Dict[str, str]
) -> str:
    """Generate the complete tutorial HTML."""

    # Load CSS - use card_styles.css for the flashcard examples
    try:
        css = load_template("card_styles.css", "src/templates")
    except Exception as e:
        print(f"Warning: Could not load card_styles.css: {e}")
        css = "/* Card styles could not be loaded */"

    # Generate cards HTML
    cards_html = []

    for i, card in enumerate(tutorial_cards):
        card_html = f"""
        <div class="tutorial-card{' active' if i == 0 else ''}" id="card-{i}">
            <h2 class="card-title">{card['title']}</h2>
            <div class="card-content">
        """

        # Handle different card types
        content = card["content"]

        # Replace example placeholders with actual content
        if "EXAMPLE_LISTENING_FRONT_PLACEHOLDER" in content:
            content = content.replace(
                "EXAMPLE_LISTENING_FRONT_PLACEHOLDER",
                example_cards.get(
                    "listening_front", "<p>Example content not available</p>"
                ),
            )
        elif "EXAMPLE_READING_FRONT_PLACEHOLDER" in content:
            content = content.replace(
                "EXAMPLE_READING_FRONT_PLACEHOLDER",
                example_cards.get(
                    "reading_front", "<p>Example content not available</p>"
                ),
            )
        elif "EXAMPLE_SPEAKING_FRONT_PLACEHOLDER" in content:
            content = content.replace(
                "EXAMPLE_SPEAKING_FRONT_PLACEHOLDER",
                example_cards.get(
                    "speaking_front", "<p>Example content not available</p>"
                ),
            )
        elif "EXAMPLE_BACK_PLACEHOLDER" in content:
            content = content.replace(
                "EXAMPLE_BACK_PLACEHOLDER",
                example_cards.get("back", "<p>Example content not available</p>"),
            )

        card_html += content

        card_html += """
            </div>
        </div>
        """

        cards_html.append(card_html)

    total_cards = len(tutorial_cards)
    cards_html_joined = "\n".join(cards_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FirePhrase Flashcard Tutorial</title>
    <style>
        {css}
        
        /* Tutorial-specific styles */
        body {{
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }}
        
        .tutorial-container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .tutorial-header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #e9a649;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        .tutorial-header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: bold;
            color: #000000;
        }}
        
        .tutorial-header p {{
            margin: 10px 0 0 0;
            color: #000000;
            font-size: 1.1rem;
            font-weight: 500;
        }}
        
        .card-container {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        
        .tutorial-card {{
            background: #2d2d2d;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            width: 100%;
            max-width: 800px;
            min-height: 400px;
            display: none;
            position: relative;
        }}
        
        .tutorial-card.active {{
            display: block;
            animation: slideIn 0.3s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(20px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        
        .card-title {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 20px;
            color: #64ffda;
            text-align: center;
        }}
        
        .card-content {{
            font-size: 1.1rem;
            line-height: 1.6;
        }}
        
        /* Navigation */
        .navigation {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 30px;
            padding: 20px;
            background: #2d2d2d;
            border-radius: 12px;
        }}
        
        .nav-button {{
            background: #e9a649;
            color: #000000;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            min-width: 100px;
        }}
        
        .nav-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 87, 51, 0.4);
            background: #FF4520;
        }}
        
        .nav-button:disabled {{
            background: #555;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        
        .progress-info {{
            text-align: center;
            font-size: 1rem;
            color: #aaa;
        }}
        
        .progress-bar {{
            width: 200px;
            height: 6px;
            background: #555;
            border-radius: 3px;
            margin: 10px auto;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e9a649, #FF4520);
            border-radius: 3px;
            transition: width 0.3s ease;
        }}
        
        /* Back to FirePhrase button */
        .firephrase-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #e9a649;
            color: #000000;
            text-decoration: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            z-index: 1000;
        }}
        
        .firephrase-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(255, 87, 51, 0.4);
            color: #000000;
            text-decoration: none;
            background: #FF4520;
        }}
        
        /* Content-specific styles */
        .welcome-content, .info-content, .tips-content, .shop-content, .conclusion-content {{
            text-align: center;
        }}
        
        .card-type-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .card-type-item {{
            background: #3d3d3d;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .card-type-item h3 {{
            color: #64ffda;
            margin-bottom: 10px;
        }}
        
        .feature-list {{
            margin: 30px 0;
        }}
        
        .feature-item {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin: 15px 0;
            padding: 15px;
            background: #3d3d3d;
            border-radius: 8px;
        }}
        
        .feature-icon {{
            font-size: 1.5rem;
            min-width: 30px;
        }}
        
        .tip-item {{
            display: flex;
            align-items: flex-start;
            gap: 20px;
            margin: 20px 0;
            padding: 20px;
            background: #3d3d3d;
            border-radius: 8px;
            text-align: left;
        }}
        
        .tip-number {{
            background: #64ffda;
            color: #1a1a1a;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.1rem;
            flex-shrink: 0;
        }}
        
        .shop-features {{
            margin: 30px 0;
        }}
        
        .shop-feature {{
            display: flex;
            align-items: flex-start;
            gap: 20px;
            margin: 20px 0;
            padding: 20px;
            background: #3d3d3d;
            border-radius: 8px;
            text-align: left;
        }}
        
        .shop-icon {{
            font-size: 2rem;
            min-width: 40px;
        }}
        
        .story-features {{
            margin: 30px 0;
        }}
        
        .story-feature {{
            display: flex;
            align-items: flex-start;
            gap: 20px;
            margin: 20px 0;
            padding: 20px;
            background: #3d3d3d;
            border-radius: 8px;
            text-align: left;
        }}
        
        .story-icon {{
            font-size: 2rem;
            min-width: 40px;
        }}
        
        .shop-cta {{
            margin-top: 30px;
            padding: 20px;
            background: #e9a649;
            color: #000000;
            border-radius: 8px;
            font-weight: 600;
        }}
        
        /* Link styles */
        .collection-link, .free-link, .story-link {{
            color: #ffffff !important;
            text-decoration: none;
            font-weight: 600;
            border-bottom: 2px solid #e9a649;
            padding-bottom: 2px;
            transition: all 0.3s ease;
        }}
        
        .collection-link:hover, .free-link:hover, .story-link:hover {{
            color: #e9a649 !important;
            border-bottom-color: #ffffff;
            text-decoration: none;
        }}
        
        .story-cta {{
            margin-top: 30px;
            padding: 20px;
            background: #3d3d3d;
            border-radius: 8px;
            text-align: center;
        }}
        
        .conclusion-summary {{
            margin: 30px 0;
            padding: 20px;
            background: #3d3d3d;
            border-radius: 8px;
        }}
        
        .final-cta {{
            margin-top: 30px;
            font-size: 1.2rem;
        }}
        
        /* Example card styling */
        .example-card {{
            margin-top: 20px;
            padding: 20px;
            background: #1a1a1a;
            border-radius: 8px;
            border: 2px solid #4a4a4a;
        }}
        
        .example-explanation {{
            text-align: left;
            margin-bottom: 20px;
            padding: 15px;
            background: #3d3d3d;
            border-radius: 8px;
        }}
        
        .example-explanation ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        
        .example-explanation li {{
            margin: 8px 0;
        }}
        
        /* Anki buttons simulation */
        .anki-buttons {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            justify-content: center;
        }}
        
        .anki-button {{
            flex: 1;
            padding: 10px 15px;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
            color: white;
        }}
        
        .anki-button.again {{
            background: #dc3545;
        }}
        
        .anki-button.hard {{
            background: #6c757d;
        }}
        
        .anki-button.good {{
            background: #28a745;
        }}
        
        .anki-button.easy {{
            background: #007bff;
        }}
        
        .anki-button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .anki-button.again:hover {{
            background: #c82333;
        }}
        
        .anki-button.hard:hover {{
            background: #5a6268;
        }}
        
        .anki-button.good:hover {{
            background: #218838;
        }}
        
        .anki-button.easy:hover {{
            background: #0056b3;
        }}
        
        /* Tooltip styles */
        .anki-button::after {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 120%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.8rem;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease;
            z-index: 1000;
        }}
        
        .anki-button::before {{
            content: '';
            position: absolute;
            bottom: 110%;
            left: 50%;
            transform: translateX(-50%);
            border: 5px solid transparent;
            border-top-color: #333;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease;
        }}
        
        .anki-button:hover::after,
        .anki-button:hover::before {{
            opacity: 1;
            visibility: visible;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .tutorial-container {{
                padding: 10px;
            }}
            
            .tutorial-card {{
                padding: 20px;
                min-height: 300px;
            }}
            
            .card-title {{
                font-size: 1.5rem;
            }}
            
            .tutorial-header h1 {{
                font-size: 2rem;
            }}
            
            .firephrase-button {{
                position: static;
                margin: 20px auto;
                display: block;
                text-align: center;
                width: fit-content;
            }}
            
            .anki-buttons {{
                flex-direction: column;
                gap: 8px;
            }}
            
            .anki-button {{
                font-size: 0.8rem;
                padding: 8px 12px;
            }}
            
            /* Hide tooltips on mobile (they don't work well on touch) */
            .anki-button::after,
            .anki-button::before {{
                display: none !important;
            }}
            
            /* Prevent hover states on mobile that can interfere with touch */
            .anki-button:hover::after,
            .anki-button:hover::before {{
                display: none !important;
                opacity: 0 !important;
                visibility: hidden !important;
            }}
            
            .card-type-grid {{
                grid-template-columns: 1fr;
            }}
            
            .tip-item, .shop-feature, .story-feature {{
                flex-direction: column;
                text-align: center;
                gap: 10px;
            }}
            
            /* Stack navigation buttons vertically on mobile */
            .navigation {{
                flex-direction: column;
                gap: 15px;
                padding: 15px;
            }}
            
            .nav-button {{
                width: 100%;
                min-width: unset;
                padding: 14px 20px;
                font-size: 1.1rem;
            }}
            
            .progress-info {{
                order: -1; /* Show progress info at the top */
            }}
            
            .progress-bar {{
                width: 100%;
                margin: 10px 0;
            }}
        }}
        

        
        /* Mobile tooltip styles */
        .mobile-tooltip {{
            position: fixed;
            background: #333 !important;
            color: white !important;
            padding: 8px 12px !important;
            border-radius: 4px !important;
            font-size: 0.8rem !important;
            z-index: 10000 !important;
            white-space: nowrap !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
            pointer-events: none !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            max-width: 250px;
            white-space: normal !important;
            text-align: center;
            line-height: 1.3;
        }}
        
        /* Additional styles for interactive elements */
        .hide-button {{
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 2px 6px;
            border-radius: 4px;
            cursor: pointer;
            font-size: inherit;
            font-family: inherit;
            transition: all 0.3s ease;
        }}
        
        .hide-button:hover {{
            background: #ff5252;
            transform: scale(1.05);
        }}
        
        .hide-button.revealed {{
            background: #4CAF50;
        }}
        
        /* Ensure flip containers work properly in tutorial */
        .example-card .flip-container {{
            cursor: pointer;
        }}
        
        .example-card .flip-container.flipped {{
            transform: rotateY(180deg);
        }}
        
        .example-card .flip-front,
        .example-card .flip-back {{
            backface-visibility: hidden;
        }}
        
        .example-card .flip-back {{
            transform: rotateY(180deg);
        }}
    </style>
</head>
<body>
    <a href="https://firephrase.co.uk" class="firephrase-button" target="_blank" rel="noopener">
        🔥 Back to FirePhrase
    </a>
    
    <div class="tutorial-container">
        <div class="tutorial-header">
            <h1>🔥 FirePhrase Flashcard Tutorial</h1>
            <p>Learn how to make the most of your language learning flashcards</p>
        </div>
        
        <div class="card-container">
            {cards_html_joined}
        </div>
        
        <div class="navigation">
            <button class="nav-button" id="prevBtn" onclick="changeCard(-1)">Previous</button>
            <div class="progress-info">
                <div id="cardCounter">1 of {total_cards}</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: {100/total_cards:.1f}%"></div>
                </div>
            </div>
            <button class="nav-button" id="nextBtn" onclick="changeCard(1)">Next</button>
        </div>
    </div>
    

    
    <script>
        let currentCard = 0;
        const totalCards = {total_cards};
        
        function updateNavigation() {{
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const counter = document.getElementById('cardCounter');
            const progressFill = document.getElementById('progressFill');
            
            prevBtn.disabled = currentCard === 0;
            nextBtn.disabled = currentCard === totalCards - 1;
            
            counter.textContent = `${{currentCard + 1}} of ${{totalCards}}`;
            progressFill.style.width = `${{((currentCard + 1) / totalCards) * 100}}%`;
        }}
        
        function showCard(index) {{
            document.querySelectorAll('.tutorial-card').forEach((card, i) => {{
                card.classList.toggle('active', i === index);
            }});
        }}
        
        function changeCard(direction) {{
            const newCard = currentCard + direction;
            if (newCard >= 0 && newCard < totalCards) {{
                currentCard = newCard;
                showCard(currentCard);
                updateNavigation();
                // Scroll to top when changing cards for better reading experience
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft') {{
                changeCard(-1);
            }} else if (e.key === 'ArrowRight') {{
                changeCard(1);
            }}
        }});
        
        // Touch/swipe navigation for mobile
        let touchStartX = null;
        let touchStartY = null;
        const minSwipeDistance = 50; // Minimum distance for a swipe
        const maxVerticalMovement = 100; // Max vertical movement to still count as horizontal swipe
        
        document.addEventListener('touchstart', function(e) {{
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        }}, {{ passive: true }});
        
        document.addEventListener('touchend', function(e) {{
            if (touchStartX === null || touchStartY === null) {{
                return; // No valid touch start
            }}
            
            const touchEndX = e.changedTouches[0].clientX;
            const touchEndY = e.changedTouches[0].clientY;
            
            const diffX = touchEndX - touchStartX;
            const diffY = Math.abs(touchEndY - touchStartY);
            
            // Reset touch coordinates
            touchStartX = null;
            touchStartY = null;
            
            // Check if this was a horizontal swipe (not vertical scroll)
            if (Math.abs(diffX) > minSwipeDistance && diffY < maxVerticalMovement) {{
                if (diffX > 0) {{
                    // Swipe right - go to previous card
                    changeCard(-1);
                }} else {{
                    // Swipe left - go to next card
                    changeCard(1);
                }}
            }}
        }}, {{ passive: true }});
        
        // Initialize
        showCard(0);
        updateNavigation();
        
        // Copy functionality for target text
        function copyToClipboard(element) {{
            const text = element.textContent;
            navigator.clipboard.writeText(text).then(() => {{
                element.classList.add('copied');
                setTimeout(() => {{
                    element.classList.remove('copied');
                }}, 1000);
            }});
        }}
        
        // Modified function to handle the copy before navigation
        function copyPromptBeforeNavigate(event) {{
            const fullPrompt = document.querySelector('.prompt-template').textContent;
            const button = document.querySelector('.insights-button');
            
            navigator.clipboard.writeText(fullPrompt).then(() => {{
                button.classList.add('copied');
                setTimeout(() => {{
                    button.classList.remove('copied');
                }}, 500);
            }}).catch(err => {{
                console.error('Failed to copy text: ', err);
            }});
            
            return true;
        }}
        
        // Function to copy template text without navigation
        function copyInsightsTemplate() {{
            const templateDiv = document.querySelector('.prompt-template');
            const textToCopy = templateDiv.textContent.trim();
            const button = document.querySelector('.copy-button');
            
            navigator.clipboard.writeText(textToCopy).then(() => {{
                // Visual feedback
                button.textContent = '✓';
                setTimeout(() => {{
                    button.textContent = '📋';
                }}, 1000);
            }}).catch(err => {{
                console.error('Failed to copy text: ', err);
            }});
        }}
        
        // Function to show mobile tooltip
        function showMobileTooltip(button, message) {{
            // Remove any existing mobile tooltips
            const existingTooltip = document.querySelector('.mobile-tooltip');
            if (existingTooltip) {{
                existingTooltip.remove();
            }}
            
            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.className = 'mobile-tooltip';
            tooltip.textContent = message;
            
            // Position it relative to the button
            const rect = button.getBoundingClientRect();
            tooltip.style.position = 'fixed';
            tooltip.style.top = (rect.top - 60) + 'px';
            tooltip.style.left = (rect.left + rect.width / 2) + 'px';
            tooltip.style.transform = 'translateX(-50%)';
            tooltip.style.background = '#333';
            tooltip.style.color = 'white';
            tooltip.style.padding = '8px 12px';
            tooltip.style.borderRadius = '4px';
            tooltip.style.fontSize = '0.8rem';
            tooltip.style.zIndex = '10000';
            tooltip.style.whiteSpace = 'nowrap';
            tooltip.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
            tooltip.style.pointerEvents = 'none';
            
            // Add animation
            tooltip.style.opacity = '0';
            tooltip.style.transition = 'opacity 0.3s ease';
            
            // Add to body
            document.body.appendChild(tooltip);
            
            // Trigger animation
            setTimeout(() => {{
                tooltip.style.opacity = '1';
            }}, 10);
            
            // Remove after 3 seconds
            setTimeout(() => {{
                if (tooltip.parentNode) {{
                    tooltip.style.opacity = '0';
                    setTimeout(() => {{
                        if (tooltip.parentNode) {{
                            tooltip.remove();
                        }}
                    }}, 300);
                }}
            }}, 3000);
        }}
        
        // Initialize any card-specific functionality
        document.addEventListener('DOMContentLoaded', function() {{
            // Initialize Anki button click handlers for mobile tooltips
            setTimeout(() => {{
                                 document.querySelectorAll('.anki-button').forEach(button => {{
                     button.addEventListener('click', function(e) {{
                         e.preventDefault(); // Prevent any default action
                         
                         // Add visual feedback
                         this.style.transform = 'scale(0.95)';
                         setTimeout(() => {{
                             this.style.transform = 'scale(1)';
                         }}, 150);
                         
                         const message = this.getAttribute('data-tooltip');
                         if (message) {{
                             showMobileTooltip(this, message);
                         }}
                     }});
                     
                     // Add CSS transition for the click effect
                     button.style.transition = 'transform 0.15s ease, background-color 0.2s ease, box-shadow 0.2s ease';
                 }});
            }}, 200);
            
            // Re-run any card-specific JavaScript after content loads
            setTimeout(() => {{
                // Initialize flip containers with proper event handling
                document.querySelectorAll('.flip-container').forEach(container => {{
                    // Remove any existing listeners to avoid duplicates
                    const newContainer = container.cloneNode(true);
                    container.parentNode.replaceChild(newContainer, container);
                    
                    newContainer.addEventListener('click', function() {{
                        this.classList.toggle('flipped');
                        const button = this.querySelector('.reveal-button');
                        if (button) {{
                            const isImage = this.classList.contains('flip-container-image');
                            const revealText = isImage ? 'Image' : 'English';
                            const hideText = isImage ? 'Hide Image' : 'Hide English';
                            
                            const svgIcon = `<svg class="reveal-icon" viewBox="0 0 24 24">
                                <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                            </svg>`;
                            
                            button.innerHTML = `${{svgIcon}} ${{this.classList.contains('flipped') ? hideText : revealText}}`;
                        }}
                    }});
                }});
                
                // Initialize click-to-copy for target text
                document.querySelectorAll('.target-text').forEach(el => {{
                    el.addEventListener('click', () => copyToClipboard(el));
                }});
                
                // Initialize target text processing for reading cards
                function processTargetText() {{
                    const containers = document.querySelectorAll('[id*="target-text-container"]');
                    containers.forEach(container => {{
                        const text = container.textContent.trim();
                        if (!text || container.querySelector('.hide-button')) return; // Already processed
                        
                        const hasSpaces = text.includes(' ');
                        
                        if (hasSpaces) {{
                            const words = text.split(' ');
                            const randomWord = words[Math.floor(Math.random() * words.length)];
                            
                            const html = words.map(word => {{
                                if (word === randomWord) {{
                                    return `<button onclick="this.classList.toggle('revealed')" 
                                            class="hide-button">${{word}}</button>`;
                                }}
                                return word;
                            }}).join(' ');
                            
                            container.innerHTML = html;
                        }} else {{
                            const chars = Array.from(text);
                            if (chars.length > 0) {{
                                const randomIndex = Math.floor(Math.random() * chars.length);
                                
                                const html = chars.map((char, index) => {{
                                    if (index === randomIndex) {{
                                        return `<button onclick="this.classList.toggle('revealed')" 
                                                class="hide-button">${{char}}</button>`;
                                    }}
                                    return char;
                                }}).join('');
                                
                                container.innerHTML = html;
                            }}
                        }}
                    }});
                }}
                
                processTargetText();
            }}, 100);
        }});
    </script>
</body>
</html>"""


def add_tutorial_card(cards_list, position, card_type, title, content):
    """
    Helper function to easily add or insert tutorial cards.

    Args:
        cards_list: The tutorial_cards list
        position: Where to insert (index, or -1 for end)
        card_type: Type of card ('text', 'example_*', etc.)
        title: Card title
        content: HTML content for the card
    """
    new_card = {"type": card_type, "title": title, "content": content}

    if position == -1:
        cards_list.append(new_card)
    else:
        cards_list.insert(position, new_card)


if __name__ == "__main__":
    import sys

    # Check if languages are provided as command line arguments
    if len(sys.argv) > 1:
        # Batch mode - generate for multiple languages
        languages = sys.argv[1:]  # All arguments after script name
        print(
            f"🌍 Batch mode: Generating tutorials for languages: {', '.join(languages)}"
        )

        results = generate_flashcard_tutorials_batch(languages)

        # Final summary
        successful_count = sum(1 for result in results.values() if result)
        if successful_count > 0:
            print(f"\n🎉 Successfully generated {successful_count} tutorials!")
            print("\n🎯 Next steps:")
            print("1. Test the public URLs to ensure they're accessible")
            print("2. Add redirects from your shop to these tutorials")
            print(
                "3. Customize tutorial cards by editing the tutorial_cards list in this file"
            )
        else:
            print("\n❌ No tutorials were generated successfully!")

    else:
        # Single mode - generate for default language
        print("🔥 Single mode: Generating tutorial with default settings...")
        output_file = generate_flashcard_tutorial()

        if output_file and output_file.startswith("gs://"):
            print(f"Tutorial uploaded to GCS: {output_file}")
            public_url = output_file.replace("gs://", "https://storage.googleapis.com/")
            print(f"🔗 Access at: {public_url}")
            print()
            print("🎯 Next steps:")
            print("1. Test the public URL to ensure it's accessible")
            print("2. Add a redirect from your shop to this tutorial")
            print(
                "3. Customize tutorial cards by editing the tutorial_cards list in this file"
            )
            print()
            print("💡 Tip: To generate for multiple languages, run:")
            print("    python flashcard_tutorial_generator.py Spanish French German")
        elif output_file:
            print(f"Tutorial generated locally: {output_file}")
            print("Manual upload needed to your GCS bucket to make it accessible!")
            print()
            print("🎯 Next steps:")
            print("1. Upload flashcard-tutorial.html to your public GCS bucket")
            print("2. Make it publicly accessible")
            print("3. Add a redirect from your shop to this tutorial")
            print(
                "4. Customize tutorial cards by editing the tutorial_cards list in this file"
            )
            print()
            print("💡 Tip: To generate for multiple languages, run:")
            print("    python flashcard_tutorial_generator.py Spanish French German")
        else:
            print("❌ Tutorial generation failed!")
            print()
            print("💡 Tip: To generate for multiple languages, run:")
            print("    python flashcard_tutorial_generator.py Spanish French German")
