/**
 * challenge_API.js
 * 
 * Shared API handler for all challenge pages.
 * Handles OpenAI API calls and state management.
 * Store this file on your GCS bucket and import it in all challenge HTML files.
 */

const ChallengeAPI = (() => {
    // Private state
    let apiKey = '';
    const state = {
        conversationHistory: {},
        transcripts: {},
        mediaRecorders: {},
        audioChunks: {},
        audioElements: {},
        isRecording: {},
        isProcessing: {}
    };

    // Private helper functions
    async function callOpenAI(endpoint, body, headers = {}) {
        const response = await fetch(`https://api.openai.com/v1/${endpoint}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                ...headers
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error?.message || 'API request failed');
        }

        return await response.json();
    }

    async function callOpenAIWithFormData(endpoint, formData) {
        const response = await fetch(`https://api.openai.com/v1/${endpoint}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`
            },
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error?.message || 'API request failed');
        }

        return await response.json();
    }

    async function textToSpeech(text) {
        const response = await fetch('https://api.openai.com/v1/audio/speech', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'tts-1',
                voice: 'nova',
                input: text.trim().substring(0, 4096),
                speed: 0.9
            })
        });

        if (!response.ok) {
            throw new Error('Text-to-speech failed');
        }

        return await response.blob();
    }

    // Public API
    return {
        /**
         * Set the OpenAI API key
         */
        setApiKey(key) {
            apiKey = key;
        },

        /**
         * Start a new challenge conversation
         * @param {string} challengeId - Unique ID for this challenge
         * @param {string} systemPrompt - The AI system prompt
         * @param {string} languageCode - Language code (e.g., 'sv' for Swedish)
         * @returns {Promise<Object>} - { transcripts: [...] }
         */
        async startChallenge(challengeId, systemPrompt, languageCode) {
            if (!apiKey) {
                throw new Error('Please set an API key first');
            }

            const messages = [{ role: 'system', content: systemPrompt }];
            const data = await callOpenAI('chat/completions', {
                model: 'gpt-4o',
                messages: messages,
                max_tokens: 150,
                temperature: 0.8
            });

            const aiMessage = data.choices[0].message.content;

            state.conversationHistory[challengeId] = [
                { role: 'system', content: systemPrompt },
                { role: 'assistant', content: aiMessage }
            ];

            state.transcripts[challengeId] = [{ type: 'ai', text: aiMessage }];

            // Play the AI's greeting
            const audioBlob = await textToSpeech(aiMessage);
            const audioUrl = URL.createObjectURL(audioBlob);
            const audioElement = new Audio(audioUrl);

            if (!state.audioElements[challengeId]) {
                state.audioElements[challengeId] = [];
            }
            state.audioElements[challengeId].push({ audioUrl, audioElement });

            await audioElement.play();

            return {
                transcripts: state.transcripts[challengeId]
            };
        },

        /**
         * Start recording audio
         * @param {string} challengeId - Unique ID for this challenge
         */
        async startRecording(challengeId) {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            state.audioChunks[challengeId] = [];
            const mediaRecorder = new MediaRecorder(stream);
            state.mediaRecorders[challengeId] = { recorder: mediaRecorder, stream: stream };

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    state.audioChunks[challengeId].push(event.data);
                }
            };

            mediaRecorder.start();
            state.isRecording[challengeId] = true;
        },

        /**
         * Stop recording and process the audio
         * @param {string} challengeId - Unique ID for this challenge
         * @param {string} languageCode - Language code for transcription
         * @returns {Promise<Object>} - { transcripts: [...] }
         */
        async stopRecording(challengeId, languageCode) {
            return new Promise((resolve, reject) => {
                const { recorder, stream } = state.mediaRecorders[challengeId];
                
                recorder.onstop = async () => {
                    try {
                        const audioBlob = new Blob(state.audioChunks[challengeId], { type: 'audio/webm' });
                        stream.getTracks().forEach(track => track.stop());
                        state.isRecording[challengeId] = false;

                        // Transcribe the audio
                        const formData = new FormData();
                        formData.append('file', audioBlob, 'audio.webm');
                        formData.append('model', 'whisper-1');
                        formData.append('language', languageCode.split('-')[0]);

                        const transcriptionData = await callOpenAIWithFormData('audio/transcriptions', formData);
                        const userText = transcriptionData.text;

                        state.transcripts[challengeId].push({ type: 'user', text: userText });

                        // Get AI response
                        const messages = [
                            ...state.conversationHistory[challengeId],
                            { role: 'user', content: userText }
                        ];

                        const chatData = await callOpenAI('chat/completions', {
                            model: 'gpt-4o',
                            messages: messages,
                            max_tokens: 150,
                            temperature: 0.8
                        });

                        const aiMessage = chatData.choices[0].message.content;

                        state.conversationHistory[challengeId] = [
                            ...messages,
                            { role: 'assistant', content: aiMessage }
                        ];

                        state.transcripts[challengeId].push({ type: 'ai', text: aiMessage });

                        // Play the AI response
                        const audioBlob2 = await textToSpeech(aiMessage);
                        const audioUrl = URL.createObjectURL(audioBlob2);
                        const audioElement = new Audio(audioUrl);

                        state.audioElements[challengeId].push({ audioUrl, audioElement });

                        await audioElement.play();

                        resolve({
                            transcripts: state.transcripts[challengeId]
                        });
                    } catch (error) {
                        reject(error);
                    }
                };

                recorder.stop();
            });
        },

        /**
         * Reset a challenge
         * @param {string} challengeId - Unique ID for this challenge
         */
        resetChallenge(challengeId) {
            // Stop and clean up audio
            if (state.audioElements[challengeId]) {
                state.audioElements[challengeId].forEach(({ audioUrl, audioElement }) => {
                    audioElement.pause();
                    URL.revokeObjectURL(audioUrl);
                });
                state.audioElements[challengeId] = [];
            }

            // Stop recording if active
            if (state.mediaRecorders[challengeId]) {
                const { recorder, stream } = state.mediaRecorders[challengeId];
                if (recorder.state !== 'inactive') {
                    recorder.stop();
                }
                stream.getTracks().forEach(track => track.stop());
            }

            // Reset state
            state.conversationHistory[challengeId] = [];
            state.transcripts[challengeId] = [];
            state.isRecording[challengeId] = false;
            state.isProcessing[challengeId] = false;
        },

        /**
         * Replay an AI audio response
         * @param {string} challengeId - Unique ID for this challenge
         * @param {number} transcriptIndex - Index in the transcript array
         */
        replayAudio(challengeId, transcriptIndex) {
            const transcript = state.transcripts[challengeId];
            let aiMessageIndex = -1;
            
            for (let i = 0; i <= transcriptIndex; i++) {
                if (transcript[i].type === 'ai') {
                    aiMessageIndex++;
                }
            }

            const audios = state.audioElements[challengeId];
            if (audios && audios[aiMessageIndex]) {
                const { audioElement } = audios[aiMessageIndex];
                audioElement.currentTime = 0;
                audioElement.play();
            }
        },

        /**
         * Get current state (for debugging)
         */
        getState() {
            return state;
        }
    };
})();