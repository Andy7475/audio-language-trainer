/**
 * ChallengeAPIHandlers.js
 *
 * Handles all OpenAI API interactions for the Challenge Viewer.
 * Includes conversation initialization, audio processing, and text-to-speech.
 */

const ChallengeAPIHandlers = {
    /**
     * Initialize a new conversation with the AI
     *
     * @param {string} challengeId - Unique identifier for the challenge
     * @param {string} systemPrompt - The system prompt to initialize the conversation
     * @param {string} apiKey - OpenAI API key
     * @param {Object} state - State setters object containing:
     *   - setIsProcessing
     *   - setStatusMessages
     *   - setConversationHistory
     *   - setTranscripts
     * @param {Function} textToSpeechCallback - Callback to play the AI's first response
     */
    async initializeConversation(challengeId, systemPrompt, apiKey, state, textToSpeechCallback) {
        const { setIsProcessing, setStatusMessages, setConversationHistory, setTranscripts } = state;

        if (!apiKey) {
            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Please enter an API key' }));
            return;
        }

        setIsProcessing(prev => ({ ...prev, [challengeId]: true }));
        setStatusMessages(prev => ({ ...prev, [challengeId]: 'Starting conversation...' }));

        try {
            const messages = [{ role: 'system', content: systemPrompt }];
            const response = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'gpt-4o',
                    messages: messages,
                    max_tokens: 150,
                    temperature: 0.8
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error?.message || 'Failed to start conversation');
            }

            const data = await response.json();
            const aiMessage = data.choices[0].message.content;

            setConversationHistory(prev => ({
                ...prev,
                [challengeId]: [
                    { role: 'system', content: systemPrompt },
                    { role: 'assistant', content: aiMessage }
                ]
            }));

            setTranscripts(prev => ({
                ...prev,
                [challengeId]: [{ type: 'ai', text: aiMessage }]
            }));

            await textToSpeechCallback(challengeId, aiMessage);

            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Ready - click Record to respond' }));
        } catch (error) {
            console.error('Error initializing conversation:', error);
            setStatusMessages(prev => ({ ...prev, [challengeId]: `Error: ${error.message}` }));
        } finally {
            setIsProcessing(prev => ({ ...prev, [challengeId]: false }));
        }
    },

    /**
     * Process recorded audio: transcribe, get AI response, and play response
     *
     * @param {string} challengeId - Unique identifier for the challenge
     * @param {Blob} audioBlob - The recorded audio blob
     * @param {string} apiKey - OpenAI API key
     * @param {string} languageCode - Language code for transcription (e.g., 'es-ES')
     * @param {Object} conversationHistory - Current conversation history for this challenge
     * @param {Object} state - State setters object
     * @param {Function} textToSpeechCallback - Callback to play the AI's response
     */
    async processAudio(challengeId, audioBlob, apiKey, languageCode, conversationHistory, state, textToSpeechCallback) {
        const { setIsProcessing, setStatusMessages, setConversationHistory, setTranscripts } = state;

        setIsProcessing(prev => ({ ...prev, [challengeId]: true }));

        try {
            // Step 1: Transcribe the audio
            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Transcribing...' }));
            const formData = new FormData();
            formData.append('file', audioBlob, 'audio.webm');
            formData.append('model', 'whisper-1');
            const whisperLangCode = languageCode.split('-')[0];
            formData.append('language', whisperLangCode);

            const transcriptionResponse = await fetch('https://api.openai.com/v1/audio/transcriptions', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${apiKey}` },
                body: formData
            });

            if (!transcriptionResponse.ok) {
                const error = await transcriptionResponse.json();
                throw new Error(error.error?.message || 'Transcription failed');
            }

            const transcriptionData = await transcriptionResponse.json();
            const userText = transcriptionData.text;

            setTranscripts(prev => ({
                ...prev,
                [challengeId]: [
                    ...(prev[challengeId] || []),
                    { type: 'user', text: userText }
                ]
            }));

            // Step 2: Get AI response
            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Getting response...' }));
            const messages = [
                ...(conversationHistory[challengeId] || []),
                { role: 'user', content: userText }
            ];

            const chatResponse = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'gpt-4o',
                    messages: messages,
                    max_tokens: 150,
                    temperature: 0.8
                })
            });

            if (!chatResponse.ok) {
                const error = await chatResponse.json();
                throw new Error(error.error?.message || 'Chat completion failed');
            }

            const chatData = await chatResponse.json();
            const aiMessage = chatData.choices[0].message.content;

            setConversationHistory(prev => ({
                ...prev,
                [challengeId]: [...messages, { role: 'assistant', content: aiMessage }]
            }));

            setTranscripts(prev => ({
                ...prev,
                [challengeId]: [
                    ...(prev[challengeId] || []),
                    { type: 'ai', text: aiMessage }
                ]
            }));

            // Step 3: Play the AI response
            await textToSpeechCallback(challengeId, aiMessage);

            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Ready - click Record to respond' }));
        } catch (error) {
            console.error('Error processing audio:', error);
            setStatusMessages(prev => ({ ...prev, [challengeId]: `Error: ${error.message}` }));
        } finally {
            setIsProcessing(prev => ({ ...prev, [challengeId]: false }));
        }
    },

    /**
     * Convert text to speech using OpenAI TTS
     *
     * @param {string} challengeId - Unique identifier for the challenge
     * @param {string} text - Text to convert to speech
     * @param {string} apiKey - OpenAI API key
     * @param {Object} audioElementRefs - Ref object to store audio elements
     * @param {Object} state - State setters object
     */
    async textToSpeech(challengeId, text, apiKey, audioElementRefs, state) {
        const { setStatusMessages, setIsPlaying, setIsProcessing } = state;

        try {
            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Generating speech...' }));

            // Sanitize and limit text length for TTS
            const sanitizedText = text.trim().substring(0, 4096);

            const response = await fetch('https://api.openai.com/v1/audio/speech', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'tts-1',
                    voice: 'nova',
                    input: sanitizedText,
                    speed: 0.9
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                let errorMessage = 'TTS failed';
                try {
                    const errorJson = JSON.parse(errorText);
                    errorMessage = errorJson.error?.message || errorMessage;
                } catch (e) {
                    errorMessage = errorText || errorMessage;
                }
                throw new Error(errorMessage);
            }

            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audioElement = new Audio(audioUrl);

            if (!audioElementRefs.current[challengeId]) {
                audioElementRefs.current[challengeId] = [];
            }
            audioElementRefs.current[challengeId].push({ audioUrl, audioElement });

            setIsPlaying(prev => ({ ...prev, [challengeId]: true }));
            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Playing response...' }));

            audioElement.onended = () => {
                setIsPlaying(prev => ({ ...prev, [challengeId]: false }));
                setStatusMessages(prev => ({ ...prev, [challengeId]: 'Ready - click Record to respond' }));
            };

            await audioElement.play();
        } catch (error) {
            console.error('Error with TTS:', error);
            setStatusMessages(prev => ({ ...prev, [challengeId]: `TTS Error: ${error.message}` }));
        } finally {
            setIsPlaying(prev => ({ ...prev, [challengeId]: false }));
            setIsProcessing(prev => ({ ...prev, [challengeId]: false }));
        }
    }
};
