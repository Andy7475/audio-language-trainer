# audio-language-trainer

## Caveat
A personal project - not intended for plug-and-play release. You are welcome to use what is here, but it will not work out of the box (see setup section later)

## Motivation
There is often a gap after completing a basic language course (like Section 1 on DuoLingo, or perhaps you've done a Foundation Michel Thomas method) - the advice is typically to start reading magazines and watching TV, but this is far too hard. I was wanting something that exposed me to some longer dialogue, rapidly increased by vocabulary, and also reinforced what I had learnt (or was still learning). A sort of 'stepping stone' towards watching a TV program perhaps, or going abroad and being more comfortable in the middle of a group where people are conversing - they won't pause after 3 or 4 words, they just keep talking amongst each other!

This is designed to fill that gap.

## Introduction

Given a vocab list and target language, it will produce practice material for you. This practice comes in two formats:

### Flash cards
I've used evidence-based techniques to make engaging flash cards:

✓ Phrases rather than isolated words (So you learn to use these in the right context - [Lexical Approach](https://en.wikipedia.org/wiki/Lexical_approach))

✓ Images and Audio for multiple encodings ([dual-encoding theory](https://en.wikipedia.org/wiki/Dual-coding_theory) of language learning - retention and recall boost)

✓ [Cloze deletion](https://en.wikipedia.org/wiki/Cloze_test) to encourage your brain to complete sentences (rather than giving you the answer as multiple choice)

✓ Additional, word-separated 'slow' audio to help with your pronunciation

✓ Handy hyperlinks and prompts for further study in [Wiktionary](https://www.wiktionary.org/) or an LLM of your choice

You will really start to notice the benefit about 200 phrases in, as you will have seen sufficient phrase overlap to start to see patterns.

### Long-form stories
We can pair up a set of flash cards with a story (that uses the same vocab). This lets you consolidate what you have learnt, and gives you practice in long-form listening.
I also produce a section of double-speed audio for the whole story, as research shows this can improve language parsing ability (your brain learns to separate words better) - but crucially this double-speed effect only works if you already know the vocab.

## Inputs
You need to update or provide your own (english) vocab list (a JSON file).
There are various notebooks to create material:
* create phrases
* create images for those phrases
* package those up into an Anki flashcard deck
* create a story off a selection of flashcards (embedded in both a webpage and a downloadable album with synchronised lyrics)
* publish the story to the cloud (so you can access it from the flashcard)

## Setup
Use the notebook which guides you through the process, but there is substantial setup required in terms of Google Cloud APIs,and FFMPEG (for audio generation) to configure if you want to run the code yourself. If you want an example lesson, happy to oblige!

Google Cloud:
* You will need an account and a project with the ID and Number, as well as know your regions for the LLM.

Google APIs used:
* Text to Speech
* Translate
* Vertex AI (and authorise Anthropic Sonnet 3.5 in the model garden)

Client:
* install FFMPEG for audio generation and fonts for the PDF system to work


# Acknowledgements
https://www.saysomethingin.com/en/home/ - heavily inspired by the approach of this company, DuoLingo and the Michel Thomas method.
