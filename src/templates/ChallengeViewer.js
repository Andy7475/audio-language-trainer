const ChallengeViewer = ({ challengeData, title, targetLanguage, collectionName, collectionRaw }) => {
    const [apiKey, setApiKey] = React.useState('');
    const [activeSessions, setActiveSessions] = React.useState({});
    const [showAnswers, setShowAnswers] = React.useState({});
    const [connectionStatus, setConnectionStatus] = React.useState({});
    const [expandedGroups, setExpandedGroups] = React.useState({});
    const [showPrompts, setShowPrompts] = React.useState({});
    const [editablePrompts, setEditablePrompts] = React.useState({});
    const audioRefs = React.useRef({});
    const wsRefs = React.useRef({});
    const mediaStreamRefs = React.useRef({});
    const audioContextRefs = React.useRef({});
    const playbackContextRefs = React.useRef({});
    const audioQueueRefs = React.useRef({});
    const isPlayingRefs = React.useRef({});

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
    
        return React.createElement('div', { 
            className: "info-panel" 
        },
            React.createElement('div', {
                className: "info-panel-header",
                onClick: () => setIsExpanded(!isExpanded)
            },
                React.createElement('div', { className: "info-panel-title" },
                    React.createElement('span', { className: "info-panel-icon" }, "ℹ️"),
                    React.createElement('span', { className: "info-panel-text" }, 
                        "Experimental AI Speaking Feature"
                    )
                ),
                React.createElement('button', { className: "info-panel-toggle" },
                    isExpanded ? '▼' : '▶'
                )
            ),
            
            isExpanded && React.createElement('div', { 
                className: "info-panel-content" 
            },
                React.createElement('p', { className: "font-semibold" },
                    "🧪 This is an experimental feature provided complimentary with no technical support."
                ),
                React.createElement('p', { className: "font-semibold" },
                    "🔑 You will need your own OpenAI account and API key with chat.completions permissions."
                ),
                
                React.createElement('p', null, "For security and best practices:"),
                React.createElement('ul', { className: "info-panel-list" },
                    React.createElement('li', null, "Create a dedicated API key for this application"),
                    React.createElement('li', null, "Disable auto top-up for this key"),
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
                    React.createElement('li', null, "Enable chat.completions permission"),
                    React.createElement('li', null, "Copy and paste the key above")
                ),
                
                React.createElement('p', { className: "mt-2" },
                    "For more information, see OpenAI's ",
                    React.createElement('a', {
                        href: "https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety",
                        className: "info-panel-link",
                        target: "_blank",
                        rel: "noopener noreferrer"
                    }, "API key safety best practices"),
                    "."
                )
            )
        );
    };
    
    React.useEffect(() => {
        if (window.hideLoadingMessage) {
            window.hideLoadingMessage();
        }
    }, []);

    const setupAudio = async (challengeId) => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 24000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            mediaStreamRefs.current[challengeId] = stream;
            const audioContext = new AudioContext({ sampleRate: 24000 });
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            processor.onaudioprocess = (e) => {
                if (wsRefs.current[challengeId]?.readyState === WebSocket.OPEN) {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmData = new Int16Array(inputData.length);
                    
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    
                    const message = {
                        type: 'input_audio_buffer.append',
                        audio: btoa(String.fromCharCode.apply(null, new Uint8Array(pcmData.buffer)))
                    };
                    
                    wsRefs.current[challengeId].send(JSON.stringify(message));
                }
            };
            
            return { audioContext, processor };
        } catch (error) {
            console.error('Audio setup error:', error);
            setConnectionStatus(prev => ({ ...prev, [challengeId]: `Audio setup error: ${error.message}` }));
            return null;
        }
    };

    const startSession = async (challengeId, defaultPrompt) => {
        const prompt = editablePrompts[challengeId] || defaultPrompt;
        if (!apiKey) {
            setConnectionStatus(prev => ({ ...prev, [challengeId]: 'Please enter an API key' }));
            return;
        }

        // Clean up any existing session
        await endSession(challengeId);

        try {
            setConnectionStatus(prev => ({ ...prev, [challengeId]: 'Setting up audio...' }));
            const audioSetup = await setupAudio(challengeId);
            if (!audioSetup) throw new Error('Audio setup failed');

            setConnectionStatus(prev => ({ ...prev, [challengeId]: 'Getting session token...' }));
            const tokenResponse = await fetch('https://create-realtime-chat-session-307643465852.europe-west2.run.app', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    api_key: apiKey,
                    model: 'gpt-4o-realtime-preview-2024-12-17',
                    voice: 'verse',
                    instructions: prompt,
                    modalities: ['text', 'audio'],
                    input_audio_format: 'pcm16',
                    output_audio_format: 'pcm16'
                })
            });

            if (!tokenResponse.ok) throw new Error('Failed to get session token');
            
            const data = await tokenResponse.json();
            const ephemeralKey = data.client_secret.value;

            const ws = new WebSocket(
                "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
                ["realtime", "openai-insecure-api-key." + ephemeralKey, "openai-beta.realtime-v1"]
            );

            wsRefs.current[challengeId] = ws;

            ws.onopen = () => {
                setConnectionStatus(prev => ({ ...prev, [challengeId]: 'Connected' }));
                setActiveSessions(prev => ({ ...prev, [challengeId]: true }));
            };

            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                console.log('Received:', data);

                if (data.type === 'response.audio.delta' && data.delta) {
                    try {
                        const pcmBuffer = base64ToArrayBuffer(data.delta);
                        const wavBuffer = createWavBuffer(pcmBuffer);
                        audioQueueRefs.current[challengeId] = audioQueueRefs.current[challengeId] || [];
                        audioQueueRefs.current[challengeId].push(wavBuffer);
                        playNextInQueue(challengeId);
                    } catch (error) {
                        console.error('Error processing audio:', error);
                    }
                }
            };

            ws.onclose = () => {
                setConnectionStatus(prev => ({ ...prev, [challengeId]: 'Disconnected' }));
                setActiveSessions(prev => ({ ...prev, [challengeId]: false }));
                endSession(challengeId);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setConnectionStatus(prev => ({ ...prev, [challengeId]: 'Connection error' }));
            };

        } catch (error) {
            console.error('Error:', error);
            setConnectionStatus(prev => ({ ...prev, [challengeId]: `Error: ${error.message}` }));
            await endSession(challengeId);
        }
    };

    const createWavBuffer = (pcmBuffer, sampleRate = 24000) => {
        const wavHeader = createWavHeader(pcmBuffer.byteLength / 2, sampleRate);
        const wavBuffer = new ArrayBuffer(wavHeader.byteLength + pcmBuffer.byteLength);
        new Uint8Array(wavBuffer).set(new Uint8Array(wavHeader), 0);
        new Uint8Array(wavBuffer).set(new Uint8Array(pcmBuffer), wavHeader.byteLength);
        return wavBuffer;
    };

    const createWavHeader = (length, sampleRate = 24000) => {
        const buffer = new ArrayBuffer(44);
        const view = new DataView(buffer);
        
        view.setUint8(0, 'R'.charCodeAt(0));
        view.setUint8(1, 'I'.charCodeAt(0));
        view.setUint8(2, 'F'.charCodeAt(0));
        view.setUint8(3, 'F'.charCodeAt(0));
        view.setUint32(4, 36 + length * 2, true);
        view.setUint8(8, 'W'.charCodeAt(0));
        view.setUint8(9, 'A'.charCodeAt(0));
        view.setUint8(10, 'V'.charCodeAt(0));
        view.setUint8(11, 'E'.charCodeAt(0));
        view.setUint8(12, 'f'.charCodeAt(0));
        view.setUint8(13, 'm'.charCodeAt(0));
        view.setUint8(14, 't'.charCodeAt(0));
        view.setUint8(15, ' '.charCodeAt(0));
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        view.setUint8(36, 'd'.charCodeAt(0));
        view.setUint8(37, 'a'.charCodeAt(0));
        view.setUint8(38, 't'.charCodeAt(0));
        view.setUint8(39, 'a'.charCodeAt(0));
        view.setUint32(40, length * 2, true);
        
        return buffer;
    };

    const base64ToArrayBuffer = (base64) => {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    };

    const playNextInQueue = async (challengeId) => {
        if (!audioQueueRefs.current[challengeId]?.length || isPlayingRefs.current[challengeId]) {
            return;
        }

        isPlayingRefs.current[challengeId] = true;
        const audioData = audioQueueRefs.current[challengeId].shift();

        try {
            if (!playbackContextRefs.current[challengeId]) {
                playbackContextRefs.current[challengeId] = new AudioContext({ sampleRate: 24000 });
            }

            const audioBuffer = await playbackContextRefs.current[challengeId].decodeAudioData(audioData);
            const source = playbackContextRefs.current[challengeId].createBufferSource();
            source.buffer = audioBuffer;
            source.connect(playbackContextRefs.current[challengeId].destination);
            
            source.onended = () => {
                isPlayingRefs.current[challengeId] = false;
                playNextInQueue(challengeId);
            };

            source.start();
        } catch (error) {
            console.error('Error playing audio:', error);
            isPlayingRefs.current[challengeId] = false;
            playNextInQueue(challengeId);
        }
    };

    const endSession = async (challengeId) => {
        if (wsRefs.current[challengeId]) {
            wsRefs.current[challengeId].close();
            delete wsRefs.current[challengeId];
        }
        
        if (mediaStreamRefs.current[challengeId]) {
            mediaStreamRefs.current[challengeId].getTracks().forEach(track => track.stop());
            delete mediaStreamRefs.current[challengeId];
        }

        if (audioContextRefs.current[challengeId]) {
            await audioContextRefs.current[challengeId].close();
            delete audioContextRefs.current[challengeId];
        }

        if (playbackContextRefs.current[challengeId]) {
            await playbackContextRefs.current[challengeId].close();
            delete playbackContextRefs.current[challengeId];
        }

        audioQueueRefs.current[challengeId] = [];
        isPlayingRefs.current[challengeId] = false;
        
        setActiveSessions(prev => ({ ...prev, [challengeId]: false }));
    };

    const toggleAnswer = (challengeId) => {
        setShowAnswers(prev => ({ ...prev, [challengeId]: !prev[challengeId] }));
    };

    return React.createElement('div', { className: 'app-container' },
        // Header with consistent breadcrumb navigation
        React.createElement('header', { className: 'app-header' },
            React.createElement('div', { className: 'header-content' },
                React.createElement('div', { className: 'breadcrumb-nav' },
                    React.createElement('a', {
                        href: 'https://storage.googleapis.com/audio-language-trainer-stories/index.html',
                        className: 'breadcrumb-link'
                    }, 'All Languages'),
                    React.createElement('span', { className: 'breadcrumb-separator' }, '>'),
                    React.createElement('a', {
                        href: `https://storage.googleapis.com/audio-language-trainer-stories/index.html#${targetLanguage.toLowerCase()}`,
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

        // Keep existing InfoPanel component
        React.createElement(InfoPanel),

        // Main content with grouped challenges
        React.createElement('main', { className: 'main-content' },
            challengeData.map((group, groupIndex) => 
                React.createElement('div', {
                    key: groupIndex,
                    className: 'content-card'
                },
                    // Group header
                    React.createElement('div', {
                        className: 'content-card-header',
                        onClick: () => toggleGroup(groupIndex)
                    },
                        React.createElement('div', { 
                            className: 'content-card-title'
                        },
                            React.createElement('h2', { 
                                className: 'content-card-heading'
                            }, `Scenario ${groupIndex + 1}`),
                            React.createElement('span', null, 
                                expandedGroups[groupIndex] ? '▼' : '▶'
                            )
                        ),
                        React.createElement('p', { 
                            className: 'content-card-description',
                            dangerouslySetInnerHTML: { __html: group.group_description }
                        })
                    ),
                    
                    // Variants section
                    expandedGroups[groupIndex] && React.createElement('div', { 
                        className: 'content-card-body'
                    },
                        group.variants.map((variant, variantIndex) => 
                            React.createElement('div', {
                                key: variantIndex,
                                className: 'variant-container'
                            },
                                React.createElement('h3', { 
                                    className: 'variant-title'
                                }, variant.variant),
                                
                                React.createElement('div', { 
                                    className: 'button-group'
                                },
                                    React.createElement('button', {
                                        onClick: () => startSession(
                                            `${groupIndex}-${variantIndex}`, 
                                            variant.llm_prompt
                                        ),
                                        disabled: activeSessions[`${groupIndex}-${variantIndex}`],
                                        className: `button ${
                                            activeSessions[`${groupIndex}-${variantIndex}`]
                                                ? 'secondary'
                                                : 'primary'
                                        }`
                                    }, activeSessions[`${groupIndex}-${variantIndex}`] 
                                        ? 'Session Active' 
                                        : 'Start Challenge'
                                    ),
                                    
                                    React.createElement('button', {
                                        onClick: () => endSession(`${groupIndex}-${variantIndex}`),
                                        disabled: !activeSessions[`${groupIndex}-${variantIndex}`],
                                        className: `button ${
                                            !activeSessions[`${groupIndex}-${variantIndex}`]
                                                ? 'secondary'
                                                : 'danger'
                                        }`
                                    }, 'End Session'),
                                    
                                    React.createElement('button', {
                                        onClick: () => toggleAnswer(`${groupIndex}-${variantIndex}`),
                                        className: 'button success'
                                    }, showAnswers[`${groupIndex}-${variantIndex}`] 
                                        ? 'Hide Answer' 
                                        : 'Show Answer'
                                    )
                                ),
                                React.createElement('div', { className: 'prompt-display' },
                                    React.createElement('button', {
                                        onClick: () => togglePrompt(`${groupIndex}-${variantIndex}`, variant.llm_prompt),
                                        className: 'prompt-toggle'
                                    }, showPrompts[`${groupIndex}-${variantIndex}`] ? 'Hide Prompt' : 'View/Edit Prompt'),
                                    
                                    showPrompts[`${groupIndex}-${variantIndex}`] && 
                                    React.createElement('div', { className: 'mt-2' },
                                        React.createElement('textarea', {
                                            value: editablePrompts[`${groupIndex}-${variantIndex}`],
                                            onChange: (e) => setEditablePrompts(prev => ({
                                                ...prev,
                                                [`${groupIndex}-${variantIndex}`]: e.target.value
                                            })),
                                            className: 'prompt-textarea',
                                            rows: 4
                                        }),
                                        React.createElement('button', {
                                            onClick: () => copyPrompt(`${groupIndex}-${variantIndex}`),
                                            className: 'prompt-copy-btn'
                                        }, 'Copy Prompt')
                                    )
                                ),
                                
                                connectionStatus[`${groupIndex}-${variantIndex}`] && 
                                    React.createElement('div', {
                                        className: 'status-display'
                                    }, connectionStatus[`${groupIndex}-${variantIndex}`]),
                                
                                showAnswers[`${groupIndex}-${variantIndex}`] && 
                                    React.createElement('div', {
                                        className: 'answer-display'
                                    },
                                        React.createElement('h4', { 
                                            className: 'answer-heading'
                                        }, 'Answer:'),
                                        React.createElement('p', null, variant.answer)
                                    )
                            )
                        )
                    )
                )
            )
        )
    );
};