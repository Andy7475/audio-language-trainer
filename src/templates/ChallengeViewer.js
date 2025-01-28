const ChallengeViewer = ({ challengeData, title, targetLanguage }) => {
    const [apiKey, setApiKey] = React.useState('');
    const [activeSessions, setActiveSessions] = React.useState({});
    const [showAnswers, setShowAnswers] = React.useState({});
    const [connectionStatus, setConnectionStatus] = React.useState({});
    
    const audioRefs = React.useRef({});
    const wsRefs = React.useRef({});
    const mediaStreamRefs = React.useRef({});
    const audioContextRefs = React.useRef({});
    const playbackContextRefs = React.useRef({});
    const audioQueueRefs = React.useRef({});
    const isPlayingRefs = React.useRef({});

    const InfoPanel = () => {
        const [isExpanded, setIsExpanded] = React.useState(false);
    
        return React.createElement('div', { 
            className: "mb-4 bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg" 
        },
            React.createElement('div', {
                className: "flex items-center justify-between cursor-pointer",
                onClick: () => setIsExpanded(!isExpanded)
            },
                React.createElement('div', { className: "flex items-center space-x-2" },
                    React.createElement('span', { className: "text-yellow-600 text-xl" }, "⚠️"),
                    React.createElement('span', { className: "font-medium text-yellow-800" }, 
                        "Important API Key Security Information"
                    )
                ),
                React.createElement('button', { className: "text-yellow-600" },
                    isExpanded ? '▼' : '▶'
                )
            ),
            
            isExpanded && React.createElement('div', { 
                className: "mt-4 text-sm text-yellow-700 space-y-2" 
            },
                React.createElement('p', { className: "font-semibold" },
                    "⚠️ This application requires an OpenAI API key with chat.completions write permissions."
                ),React.createElement('p', { className: "font-semibold" },
                    "⚠️ This tool is a personal project for my own use, but you are welcome to use it at your own risk."
                ),
                
                React.createElement('p', null, "For your security:"),
                React.createElement('ul', { className: "list-disc ml-6 space-y-1" },
                    React.createElement('li', null, "Create a separate API key just for this application"),
                    React.createElement('li', null, "Disable auto top-up for this key"),
                    React.createElement('li', null, "Create it in an isolated project"),
                    React.createElement('li', null, 
                        "Review the ",
                        React.createElement('a', {
                            href: "https://github.com/Andy7475/audio-language-trainer",
                            className: "text-blue-600 hover:underline",
                            target: "_blank",
                            rel: "noopener noreferrer"
                        }, "source code"),
                        " before proceeding. The code has an MIT license - this is not a commercial product, and pasting an API key into a random website is against security best practice."
                    )
                ),
                
                React.createElement('p', null, "To get started:"),
                React.createElement('ol', { className: "list-decimal ml-6 space-y-1" },
                    React.createElement('li', null, 
                        "Visit ",
                        React.createElement('a', {
                            href: "https://platform.openai.com/api-keys",
                            className: "text-blue-600 hover:underline",
                            target: "_blank",
                            rel: "noopener noreferrer"
                        }, "OpenAI API Keys")
                    ),
                    React.createElement('li', null, "Create a new secret key"),
                    React.createElement('li', null, "Enable chat.completions permission"),
                    React.createElement('li', null, "Copy and paste the key here")
                ),
                
                React.createElement('p', { className: "mt-4 text-xs" },
                    "For more information, see OpenAI's ",
                    React.createElement('a', {
                        href: "https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety",
                        className: "text-blue-600 hover:underline",
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

    const startSession = async (challengeId, prompt) => {
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

    return React.createElement('div', { className: 'min-h-screen bg-gray-100' },
        // Header with API key input
        React.createElement('header', { className: 'bg-blue-600 text-white p-4 sticky top-0 z-10' },
            React.createElement('div', { className: 'max-w-4xl mx-auto' },
                React.createElement('h1', { className: 'text-2xl font-bold mb-4' }, title),
                React.createElement('div', { className: 'flex items-center gap-4' },
                    React.createElement('label', { className: 'text-sm font-medium' }, 'OpenAI API Key:'),
                    React.createElement('input', {
                        type: 'password',
                        value: apiKey,
                        onChange: (e) => setApiKey(e.target.value),
                        className: 'flex-grow p-2 rounded text-black',
                        placeholder: 'sk-...'
                    })
                )
            ),
            
        ),
        React.createElement(InfoPanel),

        // Main content
        React.createElement('main', { className: 'max-w-4xl mx-auto p-4' },
            challengeData.map((challenge, index) => 
                React.createElement('div', {
                    key: index,
                    className: 'mb-6 bg-white rounded-lg shadow-md p-6'
                },
                    React.createElement('h2', { 
                        className: 'text-xl font-semibold mb-4'
                    }, challenge.challenge_description),
                    
                    React.createElement('div', { 
                        className: 'flex flex-col sm:grid sm:grid-cols-4 gap-2 sm:gap-4 mb-4 p-2 bg-gray-50 rounded-lg'
                    },
                        React.createElement('button', {
                            onClick: () => startSession(index, challenge.llm_prompt),
                            disabled: activeSessions[index],
                            className: `px-4 py-2 rounded ${
                                activeSessions[index]
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                            }`
                        }, activeSessions[index] ? 'Session Active' : 'Start Challenge'),
                        
                        React.createElement('button', {
                            onClick: () => endSession(index),
                            disabled: !activeSessions[index],
                            className: `px-4 py-2 rounded ${
                                !activeSessions[index]
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-red-600 hover:bg-red-700 text-white'
                            }`
                        }, 'End Session'),
                        
                        React.createElement('button', {
                            onClick: () => toggleAnswer(index),
                            className: 'px-4 py-2 rounded bg-green-600 hover:bg-green-700 text-white'
                        }, showAnswers[index] ? 'Hide Answer' : 'Show Answer')
                    ),
                    
                    connectionStatus[index] && React.createElement('div', {
                        className: 'mb-4 p-2 bg-gray-100 rounded'
                    }, connectionStatus[index]),
                    
                    showAnswers[index] && React.createElement('div', {
                        className: 'mt-4 p-4 bg-gray-50 rounded-lg'
                    },
                        React.createElement('h3', { 
                            className: 'font-medium mb-2'
                        }, 'Answer:'),
                        React.createElement('p', null, challenge.answer)
                    )
                )
            )
        )
    );
};