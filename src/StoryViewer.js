// StoryViewer component without JSX or imports
const StoryViewer = ({ storyData, targetLanguage, title }) => {
  const [activeSection, setActiveSection] = React.useState(null);
  const [isPlaying, setIsPlaying] = React.useState({});
  const [loopCount, setLoopCount] = React.useState(12); // Default 12 loops
  const [remainingLoops, setRemainingLoops] = React.useState(0);
  const [playbackMode, setPlaybackMode] = React.useState(null); // 'normal' or 'fast'
  const audioRef = React.useRef(null);
  const audioQueue = React.useRef([]);
  const fastAudioQueue = React.useRef([]);
  
  // Handle sequential playback for normal dialogue
  const playNextInQueue = () => {
    if (audioQueue.current.length > 0) {
      const nextAudio = audioQueue.current.shift();
      if (audioRef.current) {
        audioRef.current.src = `data:audio/mp3;base64,${nextAudio}`;
        audioRef.current.play();
      }
    } else {
      // Queue is empty, playback complete
      setIsPlaying({});
      setPlaybackMode(null);
    }
  };

  // Handle fast audio playback with looping
  const playNextFastAudio = () => {
    if (fastAudioQueue.current.length > 0) {
      // If we've played all sections once, decrement loop counter
      if (fastAudioQueue.current.length === Object.keys(storyData).length) {
        setRemainingLoops(prev => prev - 1);
      }

      const nextAudio = fastAudioQueue.current.shift();
      if (audioRef.current) {
        audioRef.current.src = `data:audio/mp3;base64,${nextAudio}`;
        audioRef.current.play();
      }
    } else {
      // Check if we should continue looping
      if (remainingLoops > 0) {
        // Reset queue for another loop
        const sections = Object.values(storyData);
        fastAudioQueue.current = sections.map(section => section.audio_data.fast_dialogue);
        playNextFastAudio();
      } else {
        // All loops complete
        setIsPlaying({});
        setPlaybackMode(null);
        setRemainingLoops(0);
      }
    }
  };

  React.useEffect(() => {
    // Add ended event listener for sequential playback
    if (audioRef.current) {
      const handleEnded = () => {
        if (playbackMode === 'normal') {
          playNextInQueue();
        } else if (playbackMode === 'fast') {
          playNextFastAudio();
        }
      };
      
      audioRef.current.addEventListener('ended', handleEnded);
      return () => {
        audioRef.current.removeEventListener('ended', handleEnded);
      };
    }
  }, [playbackMode]);

  const playAudio = (audioData) => {
    if (audioRef.current) {
      audioRef.current.src = `data:audio/mp3;base64,${audioData}`;
      audioRef.current.play();
    }
  };

  const playAllDialogue = (sectionIndex, dialogueAudio) => {
    setIsPlaying(prev => ({ ...prev, [sectionIndex]: true }));
    setPlaybackMode('normal');
    audioQueue.current = [...dialogueAudio];
    playNextInQueue();
  };

  const playFastAudio = (sectionIndex, loops = loopCount) => {
    setIsPlaying(prev => ({ ...prev, [sectionIndex]: true }));
    setPlaybackMode('fast');
    setRemainingLoops(loops - 1);
    
    // For single section loop
    const fastAudio = storyData[Object.keys(storyData)[sectionIndex]].audio_data.fast_dialogue;
    fastAudioQueue.current = Array(loops).fill(fastAudio);
    playNextFastAudio();
  };

  const playAllFastAudio = (loops = loopCount) => {
    setIsPlaying(prev => ({ ...prev, 'all': true }));
    setPlaybackMode('fast');
    setRemainingLoops(loops - 1);
    
    // Queue all sections in order
    const sections = Object.values(storyData);
    fastAudioQueue.current = sections.map(section => section.audio_data.fast_dialogue);
    playNextFastAudio();
  };

  const downloadM4A = (sectionIndex) => {
    const section = storyData[Object.keys(storyData)[sectionIndex]];
    if (!section.m4a_data) return;

    // Convert base64 to blob while preserving MIME type and metadata
    const byteCharacters = atob(section.m4a_data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'audio/x-m4a' });

    // Create filename from story part name
    const sectionName = Object.keys(storyData)[sectionIndex];
    const cleanName = sectionName.replace(/_/g, ' ').toLowerCase();
    const filename = `${cleanName}.m4a`;

    // Trigger download while preserving metadata
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    // Use .m4a extension to ensure media players recognize it properly
    a.type = 'audio/x-m4a';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return React.createElement('div', { className: 'min-h-screen bg-gray-100' },
    React.createElement('audio', { ref: audioRef, className: 'hidden' }),
    
    // Header with global controls
    React.createElement('header', { className: 'bg-blue-600 text-white p-4 sticky top-0 z-10' },
      React.createElement('div', { className: 'flex justify-between items-center' },
        React.createElement('h1', { className: 'text-xl font-bold' }, 
          title || 'Language Learning Story'
        ),
        React.createElement('div', { className: 'flex items-center gap-4' },
          React.createElement('div', { className: 'flex items-center gap-2' },
            React.createElement('label', { htmlFor: 'loopCount', className: 'text-sm' }, 
              'Loops:'
            ),
            React.createElement('input', {
              id: 'loopCount',
              type: 'number',
              min: 1,
              max: 50,
              value: loopCount,
              onChange: (e) => setLoopCount(Number(e.target.value)),
              className: 'w-16 px-2 py-1 text-black rounded'
            })
          ),
          React.createElement('button', {
            onClick: () => playAllFastAudio(loopCount),
            disabled: playbackMode === 'fast',
            className: `px-4 py-2 rounded-lg ${playbackMode === 'fast' ? 
              'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'} text-white`
          }, playbackMode === 'fast' ? `Playing (${remainingLoops + 1} loops left)` : 'Play All Fast')
        )
      )
    ),

    // Main content
    React.createElement('main', { className: 'max-w-4xl mx-auto p-4' },
      Object.entries(storyData).map(([sectionName, section], sectionIndex) =>
        React.createElement('div', {
          key: sectionName,
          className: 'mb-6 bg-white rounded-lg shadow-md'
        },
          // Section header button
          React.createElement('button', {
            onClick: () => setActiveSection(activeSection === sectionIndex ? null : sectionIndex),
            className: 'w-full p-4 flex items-center justify-between text-left bg-gray-50 rounded-t-lg hover:bg-gray-100'
          },
            React.createElement('h2', { className: 'text-lg font-semibold capitalize' },
              sectionName.replace(/_/g, ' ')
            ),
            React.createElement('span', null, 
              activeSection === sectionIndex ? '▼' : '▶'
            )
          ),

          // Section content
          activeSection === sectionIndex && React.createElement('div', { className: 'p-4' },
            // Story part image
            section.image_data && React.createElement('div', { 
              className: 'mb-4 rounded-lg overflow-hidden',
              style: { maxWidth: '100%', height: 'auto' }
            },
              React.createElement('img', {
                src: `data:image/jpeg;base64,${section.image_data}`,
                alt: `Story part ${sectionIndex + 1}`,
                className: 'w-full h-auto'
              })
            ),

            // Controls for dialogue playback
            React.createElement('div', { 
              className: 'grid grid-cols-3 gap-4 mb-4 p-2 bg-gray-50 rounded-lg'
            },
              React.createElement('button', {
                onClick: () => playAllDialogue(sectionIndex, section.audio_data.dialogue),
                disabled: playbackMode === 'normal',
                className: `px-4 py-2 rounded-lg ${playbackMode === 'normal' ? 
                  'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} text-white`
              }, playbackMode === 'normal' ? 'Playing...' : 'Play Dialogue'),
              
              React.createElement('button', {
                onClick: () => playFastAudio(sectionIndex, loopCount),
                disabled: playbackMode === 'fast',
                className: `px-4 py-2 rounded-lg ${playbackMode === 'fast' ? 
                  'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'} text-white`
              }, playbackMode === 'fast' ? 
                `Playing (${remainingLoops + 1} loops left)` : 
                'Play Fast Version'),
              
              section.m4a_data && React.createElement('button', {
                onClick: () => downloadM4A(sectionIndex),
                className: 'px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white'
              }, 'Download M4A')
            ),

            // Dialogue section
            section.translated_dialogue.map((utterance, index) =>
              React.createElement('div', {
                key: index,
                className: 'mb-4 p-3 bg-gray-50 rounded-lg'
              },
                React.createElement('div', { className: 'flex items-center justify-between' },
                  React.createElement('div', { className: 'flex-grow' },
                    React.createElement('p', { className: 'text-sm text-gray-600 mb-1' },
                      utterance.speaker
                    ),
                    React.createElement('p', { className: 'text-lg' },
                      createWiktionaryLinks(utterance.text)
                    ),
                    React.createElement('p', { className: 'mt-2 text-gray-600' },
                      section.dialogue[index].text
                    )
                  ),
                  section.audio_data?.dialogue[index] && React.createElement('button', {
                    onClick: () => playAudio(section.audio_data.dialogue[index]),
                    disabled: playbackMode !== null,
                    className: `p-2 rounded-full hover:bg-gray-200 ${
                      playbackMode !== null ? 'opacity-50 cursor-not-allowed' : ''
                    }`
                  }, '🔊')
                )
              )
            )
          )
        )
      )
    )
  );
};