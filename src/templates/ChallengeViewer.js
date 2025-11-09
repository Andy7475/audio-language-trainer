const ChallengeViewer = ({ challengeData, title, targetLanguage, languageCode, collectionName, collectionRaw }) => {
    const [apiKey, setApiKey] = React.useState('');
    const [showAnswers, setShowAnswers] = React.useState({});
    const [expandedGroups, setExpandedGroups] = React.useState({});
    const [showPrompts, setShowPrompts] = React.useState({});
    const [editablePrompts, setEditablePrompts] = React.useState({});

    // Turn-based state
    const [conversationHistory, setConversationHistory] = React.useState({});
    const [isRecording, setIsRecording] = React.useState({});
    const [isProcessing, setIsProcessing] = React.useState({});
    const [isPlaying, setIsPlaying] = React.useState({});
    const [statusMessages, setStatusMessages] = React.useState({});
    const [transcripts, setTranscripts] = React.useState({});

    const mediaRecorderRefs = React.useRef({});
    const audioChunksRefs = React.useRef({});
    const audioElementRefs = React.useRef({});

    const story_folder = "story_" + title.replace(/\s+/g, '_').toLowerCase();

    const toggleGroup = (groupIndex) => {
        setExpandedGroups(prev => ({
            ...prev,
            [groupIndex]: !prev[groupIndex]
        }));
    };

    const togglePrompt = (challengeId, prompt) => {
        setShowPrompts(prev => ({ ...prev, [challengeId]: !prev[challengeId] }));
        if (!editablePrompts[challengeId]) {
            setEditablePrompts(prev => ({ ...prev, [challengeId]: prompt }));
        }
    };

    const copyPrompt = (challengeId) => {
        navigator.clipboard.writeText(editablePrompts[challengeId]);
    };

    const InfoPanel = () => {
        const [isExpanded, setIsExpanded] = React.useState(false);

        return React.createElement('div', { className: "info-panel" },
            React.createElement('div', {
                className: "info-panel-header",
                onClick: () => setIsExpanded(!isExpanded)
            },
                React.createElement('div', { className: "info-panel-title" },
                    React.createElement('span', { className: "info-panel-icon" }, "ℹ️"),
                    React.createElement('span', { className: "info-panel-text" },
                        "Turn-Based AI Speaking Feature"
                    )
                ),
                React.createElement('button', { className: "info-panel-toggle" },
                    isExpanded ? '▼' : '▶'
                )
            ),
            isExpanded && React.createElement('div', { className: "info-panel-content" },
                React.createElement('p', { className: "font-semibold" },
                    "🎯 Turn-based conversation practice - you control when to speak and listen!"
                ),
                React.createElement('p', { className: "font-semibold" },
                    "🔑 You will need your own OpenAI API key with permissions for Whisper, Chat Completions, and TTS."
                ),
                React.createElement('p', null, "For security and best practices:"),
                React.createElement('ul', { className: "info-panel-list" },
                    React.createElement('li', null, "Create a dedicated API key for this application"),
                    React.createElement('li', null, "Set usage limits to control costs"),
                    React.createElement('li', null, "Create the key in an isolated project"),
                    React.createElement('li', null,
                        "Review the ",
                        React.createElement('a', {
                            href: "https://github.com/Andy7475/audio-language-trainer",
                            className: "info-panel-link",
                            target: "_blank",
                            rel: "noopener noreferrer"
                        }, "open source code"),
                        " before proceeding if desired"
                    )
                ),
                React.createElement('p', null, "To get started:"),
                React.createElement('ol', { className: "info-panel-list" },
                    React.createElement('li', null,
                        "Visit ",
                        React.createElement('a', {
                            href: "https://platform.openai.com/api-keys",
                            className: "info-panel-link",
                            target: "_blank",
                            rel: "noopener noreferrer"
                        }, "OpenAI API Keys")
                    ),
                    React.createElement('li', null, "Create a new secret key"),
                    React.createElement('li', null, "Copy and paste the key above"),
                    React.createElement('li', null, "Click 'Start Challenge', then 'Record' to begin speaking")
                ),
                React.createElement('p', { className: "mt-2" },
                    "Estimated cost: ~$0.80 per 30-minute practice session"
                )
            )
        );
    };

    React.useEffect(() => {
        if (window.hideLoadingMessage) window.hideLoadingMessage();
    }, []);

    const initializeConversation = async (challengeId, systemPrompt) => {
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

            await textToSpeech(challengeId, aiMessage);

            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Ready - click Record to respond' }));
        } catch (error) {
            console.error('Error initializing conversation:', error);
            setStatusMessages(prev => ({ ...prev, [challengeId]: `Error: ${error.message}` }));
        } finally {
            setIsProcessing(prev => ({ ...prev, [challengeId]: false }));
        }
    };

    const startRecording = async (challengeId) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            audioChunksRefs.current[challengeId] = [];
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRefs.current[challengeId] = mediaRecorder;

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRefs.current[challengeId].push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRefs.current[challengeId], { type: 'audio/webm' });
                stream.getTracks().forEach(track => track.stop());
                await processAudio(challengeId, audioBlob);
            };

            mediaRecorder.start();
            setIsRecording(prev => ({ ...prev, [challengeId]: true }));
            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Recording... click Stop when done' }));
        } catch (error) {
            console.error('Error starting recording:', error);
            setStatusMessages(prev => ({ ...prev, [challengeId]: `Microphone error: ${error.message}` }));
        }
    };

    const stopRecording = (challengeId) => {
        if (mediaRecorderRefs.current[challengeId]) {
            mediaRecorderRefs.current[challengeId].stop();
            setIsRecording(prev => ({ ...prev, [challengeId]: false }));
            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Processing your speech...' }));
        }
    };

    const processAudio = async (challengeId, audioBlob) => {
        setIsProcessing(prev => ({ ...prev, [challengeId]: true }));

        try {
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

            await textToSpeech(challengeId, aiMessage);

            setStatusMessages(prev => ({ ...prev, [challengeId]: 'Ready - click Record to respond' }));
        } catch (error) {
            console.error('Error processing audio:', error);
            setStatusMessages(prev => ({ ...prev, [challengeId]: `Error: ${error.message}` }));
        } finally {
            setIsProcessing(prev => ({ ...prev, [challengeId]: false }));
        }
    };

    const textToSpeech = async (challengeId, text) => {
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

            if (!audioElementRefs.current[challengeId]) audioElementRefs.current[challengeId] = [];
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
    };

    const resetConversation = (challengeId) => {
        if (audioElementRefs.current[challengeId]) {
            audioElementRefs.current[challengeId].forEach(a => {
                a.audioElement.pause();
                URL.revokeObjectURL(a.audioUrl); // Clean up URLs on reset
            });
            audioElementRefs.current[challengeId] = [];
        }
        if (mediaRecorderRefs.current[challengeId]) {
            mediaRecorderRefs.current[challengeId].stop();
        }

        setConversationHistory(prev => ({ ...prev, [challengeId]: [] }));
        setTranscripts(prev => ({ ...prev, [challengeId]: [] }));
        setIsRecording(prev => ({ ...prev, [challengeId]: false }));
        setIsProcessing(prev => ({ ...prev, [challengeId]: false }));
        setIsPlaying(prev => ({ ...prev, [challengeId]: false }));
        setStatusMessages(prev => ({ ...prev, [challengeId]: '' }));
    };

    const toggleAnswer = (challengeId) => {
        setShowAnswers(prev => {
            const current = !!prev[challengeId];
            return { ...prev, [challengeId]: !current };
        });
    };

    const replayResponse = (challengeId, transcriptIndex) => {
        // Find the AI message index by counting AI messages up to this point
        const transcript = transcripts[challengeId];
        let aiMessageIndex = -1;
        for (let i = 0; i <= transcriptIndex; i++) {
            if (transcript[i].type === 'ai') {
                aiMessageIndex++;
            }
        }

        const audios = audioElementRefs.current[challengeId];
        if (audios && audios[aiMessageIndex]) {
            const { audioElement } = audios[aiMessageIndex];
            audioElement.currentTime = 0;
            audioElement.play();
        } else {
            setStatusMessages(prev => ({
                ...prev,
                [challengeId]: 'No audio available for this message.'
            }));
        }
    };

    const hasActiveConversation = (challengeId) => {
        return conversationHistory[challengeId] && conversationHistory[challengeId].length > 0;
    };

    return React.createElement('div', { className: 'app-container' },
        // Header
        React.createElement('header', { className: 'app-header' },
            React.createElement('div', { className: 'header-content' },
                React.createElement('div', { className: 'breadcrumb-nav' },
                    React.createElement('a', {
                        href: 'https://storage.googleapis.com/audio-language-trainer-stories/index.html',
                        className: 'breadcrumb-link'
                    }, 'All Languages'),
                    React.createElement('span', { className: 'breadcrumb-separator' }, '>'),
                    React.createElement('a', {
                        href: `https://storage.googleapis.com/audio-language-trainer-stories/${targetLanguage.toLowerCase()}/index.html`,
                        className: 'breadcrumb-link'
                    }, targetLanguage),
                    React.createElement('span', { className: 'breadcrumb-separator' }, '>'),
                    collectionName && React.createElement('a', {
                        href: `https://storage.googleapis.com/audio-language-trainer-stories/${targetLanguage.toLowerCase()}/${(collectionRaw || collectionName).toLowerCase()}/index.html`,
                        className: 'breadcrumb-link'
                    }, collectionName),
                    collectionName && React.createElement('span', { className: 'breadcrumb-separator' }, '>'),
                    React.createElement('a', {
                        href: `https://storage.googleapis.com/audio-language-trainer-stories/${targetLanguage.toLowerCase()}/${(collectionRaw || collectionName).toLowerCase()}/${story_folder}/${story_folder}.html`,
                        className: 'breadcrumb-link'
                    }, title),
                    React.createElement('span', { className: 'breadcrumb-separator' }, '>'),
                    React.createElement('span', { className: 'breadcrumb-current' }, 'Speaking Challenges')
                ),
                React.createElement('div', { className: 'api-key-controls' },
                    React.createElement('label', { className: 'api-key-label' }, 'OpenAI API Key:'),
                    React.createElement('input', {
                        type: 'password',
                        value: apiKey,
                        onChange: (e) => setApiKey(e.target.value),
                        className: 'api-key-input',
                        placeholder: 'sk-...'
                    })
                )
            )
        ),

        React.createElement(InfoPanel),

        // Main content
        React.createElement('main', { className: 'main-content' },
            challengeData.map((group, groupIndex) =>
                React.createElement('div', { key: groupIndex, className: 'content-card' },
                    React.createElement('div', {
                        className: 'content-card-header',
                        onClick: () => toggleGroup(groupIndex)
                    },
                        React.createElement('div', { className: 'content-card-title' },
                            React.createElement('h2', { className: 'content-card-heading' }, `Scenario ${groupIndex + 1}`),
                            React.createElement('span', null, expandedGroups[groupIndex] ? '▼' : '▶')
                        ),
                        React.createElement('p', {
                            className: 'content-card-description',
                            dangerouslySetInnerHTML: { __html: group.group_description }
                        })
                    ),
                    expandedGroups[groupIndex] && React.createElement('div', { className: 'content-card-body' },
                        group.variants.map((variant, variantIndex) => {
                            const challengeId = `${groupIndex}-${variantIndex}`;
                            const hasConversation = hasActiveConversation(challengeId);

                            return React.createElement('div', { key: variantIndex, className: 'variant-container' },
                                React.createElement('h3', { className: 'variant-title' }, variant.variant),
                                React.createElement('div', { className: 'button-group' },
                                    !hasConversation && React.createElement('button', {
                                        onClick: () => initializeConversation(
                                            challengeId,
                                            editablePrompts[challengeId] || variant.llm_prompt
                                        ),
                                        disabled: isProcessing[challengeId] || !apiKey,
                                        className: 'button primary'
                                    }, 'Start Challenge'),

                                    hasConversation && !isRecording[challengeId] && React.createElement('button', {
                                        onClick: () => startRecording(challengeId),
                                        disabled: isProcessing[challengeId] || isPlaying[challengeId],
                                        className: 'button primary'
                                    }, '🎤 Record'),

                                    hasConversation && isRecording[challengeId] && React.createElement('button', {
                                        onClick: () => stopRecording(challengeId),
                                        className: 'button danger'
                                    }, '⏹️ Stop Recording'),

                                    hasConversation && React.createElement('button', {
                                        onClick: () => resetConversation(challengeId),
                                        className: 'button danger'
                                    }, 'Reset'),

                                    React.createElement('button', {
                                        onClick: () => toggleAnswer(challengeId),
                                        className: 'button success'
                                    }, showAnswers[challengeId] ? 'Hide Answer' : 'Show Answer')
                                ),
                                React.createElement('div', { className: 'prompt-display' },
                                    React.createElement('button', {
                                        onClick: () => togglePrompt(challengeId, variant.llm_prompt),
                                        className: 'prompt-toggle'
                                    }, showPrompts[challengeId] ? 'Hide Prompt' : 'View/Edit Prompt'),

                                    showPrompts[challengeId] &&
                                    React.createElement('div', { className: 'mt-2' },
                                        React.createElement('textarea', {
                                            value: editablePrompts[challengeId],
                                            onChange: (e) => setEditablePrompts(prev => ({
                                                ...prev,
                                                [challengeId]: e.target.value
                                            })),
                                            className: 'prompt-textarea',
                                            rows: 4
                                        }),
                                        React.createElement('button', {
                                            onClick: () => copyPrompt(challengeId),
                                            className: 'prompt-copy-btn'
                                        }, 'Copy Prompt')
                                    )
                                ),
                                statusMessages[challengeId] &&
                                React.createElement('div', { className: 'status-display' }, statusMessages[challengeId]),
                                transcripts[challengeId] && transcripts[challengeId].length > 0 &&
                                React.createElement('div', {
                                    className: 'transcript-display',
                                    style: {
                                        marginTop: '1rem',
                                        padding: '1rem',
                                        backgroundColor: '#f3f4f6',
                                        borderRadius: '0.5rem',
                                        fontSize: '0.9rem',
                                        maxHeight: '400px',
                                        overflowY: 'auto'
                                    }
                                },
                                    React.createElement('h4', { style: { marginTop: 0, marginBottom: '1rem' } }, 'Conversation History:'),
                                    transcripts[challengeId].map((msg, idx) => {
                                        const isAI = msg.type === 'ai';
                                        return React.createElement('div', {
                                            key: idx,
                                            style: {
                                                marginBottom: '0.75rem',
                                                paddingBottom: '0.75rem',
                                                borderBottom: idx < transcripts[challengeId].length - 1 ? '1px solid #e5e7eb' : 'none'
                                            }
                                        },
                                            React.createElement('strong', null, isAI ? 'AI: ' : 'You: '),
                                            msg.text,
                                            isAI && React.createElement('button', {
                                                className: 'replay-btn',
                                                style: { marginLeft: '0.5rem', padding: '0.2rem 0.5rem', fontSize: '0.8rem' },
                                                onClick: () => replayResponse(challengeId, idx)
                                            }, '🔊 Replay')
                                        );
                                    })
                                ),
                                showAnswers[challengeId] &&
                                React.createElement('div', {
                                    className: 'answer-display',
                                    style: {
                                        marginTop: '1rem',
                                        padding: '1rem',
                                        backgroundColor: '#e8f5e9',
                                        borderRadius: '0.5rem',
                                        borderLeft: '4px solid #4caf50'
                                    }
                                },
                                    React.createElement('strong', null, 'Answer: '),
                                    variant.answer
                                )
                            );
                        })
                    )
                )
            )
        )
    );
};
