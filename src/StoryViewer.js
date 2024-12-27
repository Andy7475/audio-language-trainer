const StoryViewer = ({ storyData, targetLanguage, title }) => {
  const createWiktionaryLinks = (text) => {
    return text.split(' ').map((word, index) => {
      const cleanWord = word.toLowerCase().replace(/[^\p{L}0-9]/gu, '');
      if (cleanWord) {
        return React.createElement(React.Fragment, { key: index },
          React.createElement('a', {
            href: `https://en.wiktionary.org/wiki/${encodeURIComponent(cleanWord)}#${targetLanguage}`,
            target: '_blank',
            rel: 'noopener noreferrer',
            className: 'text-blue-600 hover:underline'
          }, word),
          ' '
        );
      }
      return word + ' ';
    });
  };

  const [activeSection, setActiveSection] = React.useState(null);
  const [isPlaying, setIsPlaying] = React.useState({});
  const [loopCount, setLoopCount] = React.useState(12);
  const [remainingLoops, setRemainingLoops] = React.useState(0);
  const [playbackMode, setPlaybackMode] = React.useState(null);
  
  const audioRef = React.useRef(null);
  const audioQueue = React.useRef([]);
  const fastAudioQueue = React.useRef([]);
  const activeSectionAudio = React.useRef(null);

  const stopPlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    audioQueue.current = [];
    fastAudioQueue.current = [];
    activeSectionAudio.current = null;
    setIsPlaying({});
    setPlaybackMode(null);
    setRemainingLoops(0);
  };

  const playAudioData = (audioData) => {
    if (audioRef.current) {
      audioRef.current.src = `data:audio/mp3;base64,${audioData}`;
      audioRef.current.play().catch(error => {
        console.error('Audio playback error:', error);
        stopPlayback();
      });
    }
  };

  const playNextInQueue = () => {
    if (audioQueue.current.length > 0) {
      const nextAudio = audioQueue.current.shift();
      playAudioData(nextAudio);
    } else {
      setIsPlaying({});
      setPlaybackMode(null);
    }
  };

  const resetFastAudioQueue = (mode = 'single') => {
    if (mode === 'all') {
      // For playing all sections
      const sectionKeys = Object.keys(storyData);
      return sectionKeys.map((key, index) => ({
        audio: storyData[key].audio_data.fast_dialogue,
        isLastInLoop: index === sectionKeys.length - 1
      }));
    } else {
      // For playing single section
      return [{
        audio: activeSectionAudio.current,
        isLastInLoop: true
      }];
    }
  };
  
  const playFastAudio = (sectionIndex, loops = loopCount) => {
    stopPlayback();
    setIsPlaying(prev => ({ ...prev, [sectionIndex]: true }));
    setPlaybackMode('fast'); // Keep as 'fast', not 'all'
    setRemainingLoops(loops - 1); // Subtract 1 as we're about to play first loop
    
    const fastAudio = storyData[Object.keys(storyData)[sectionIndex]].audio_data.fast_dialogue;
    activeSectionAudio.current = fastAudio;
    
    // Initialize queue with first loop
    fastAudioQueue.current = [{
      audio: fastAudio,
      isLastInLoop: true
    }];
    
    playNextFastAudio();
  };
  
  const playNextFastAudio = () => {
    if (remainingLoops < 0 && fastAudioQueue.current.length === 0) {
      stopPlayback();
      return;
    }
  
    if (fastAudioQueue.current.length > 0) {
      const nextAudio = fastAudioQueue.current.shift();
      playAudioData(nextAudio.audio);
      
      if (nextAudio.isLastInLoop) {
        setRemainingLoops(prev => {
          const newCount = prev - 1;
          if (newCount >= 0) {
            // Important: Use the same mode as initial playback
            fastAudioQueue.current = resetFastAudioQueue(
              playbackMode === 'all' ? 'all' : 'single'
            );
          }
          return newCount;
        });
      }
    } else {
      stopPlayback();
    }
  };

  React.useEffect(() => {
    if (audioRef.current) {
      const handleEnded = () => {
        if (playbackMode === 'normal') {
          playNextInQueue();
        } else if (playbackMode === 'fast' || playbackMode === 'all') {
          playNextFastAudio();
        }
      };
      
      audioRef.current.addEventListener('ended', handleEnded);
      return () => {
        audioRef.current.removeEventListener('ended', handleEnded);
      };
    }
  }, [playbackMode]);

  const playAllDialogue = (sectionIndex, dialogueAudio) => {
    stopPlayback();
    setIsPlaying(prev => ({ ...prev, [sectionIndex]: true }));
    setPlaybackMode('normal');
    audioQueue.current = [...dialogueAudio];
    playNextInQueue();
  };

  

  const playAllFastAudio = (loops = loopCount) => {
    console.log('playAllFastAudio called with loops:', loops);
    console.log('Current story data keys:', Object.keys(storyData));
    console.log('Sample audio data from first section:', 
      storyData[Object.keys(storyData)[0]]?.audio_data?.fast_dialogue ? 'exists' : 'missing');
    
    stopPlayback();
    setIsPlaying(prev => ({ ...prev, 'all': true }));
    setPlaybackMode('all');
    setRemainingLoops(loops);
    
    const queue = resetFastAudioQueue('all');
    console.log('Queue created with length:', queue.length);
    fastAudioQueue.current = queue;
    
    playNextFastAudio();
  };

  const downloadM4A = (sectionIndex) => {
    const section = storyData[Object.keys(storyData)[sectionIndex]];
    if (!section.m4a_data) return;

    const byteCharacters = atob(section.m4a_data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'audio/x-m4a' });

    const sectionName = Object.keys(storyData)[sectionIndex];
    const cleanName = sectionName.replace(/_/g, ' ').toLowerCase();
    const filename = `${cleanName}.m4a`;

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
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
            disabled: playbackMode !== null,
            className: `px-4 py-2 rounded-lg ${
              playbackMode !== null
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700'
            } text-white`
          }, playbackMode !== null
            ? `Playing (${remainingLoops + 1} loops left)`
            : 'Play All Fast')
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
              className: 'mb-4 rounded-lg overflow-hidden'
            },
              React.createElement('img', {
                src: `data:image/jpeg;base64,${section.image_data}`,
                alt: `Story part ${sectionIndex + 1}`,
                className: 'w-full h-auto'
              })
            ),

            // Controls for dialogue playback
            React.createElement('div', { 
              className: 'grid grid-cols-4 gap-4 mb-4 p-2 bg-gray-50 rounded-lg'
            },
              React.createElement('button', {
                onClick: () => playAllDialogue(sectionIndex, section.audio_data.dialogue),
                disabled: playbackMode !== null,
                className: `px-4 py-2 rounded-lg ${
                  playbackMode !== null
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                } text-white`
              }, playbackMode === 'normal' ? 'Playing...' : 'Play Dialogue'),
              
              React.createElement('button', {
                onClick: () => playFastAudio(sectionIndex, loopCount),
                disabled: playbackMode !== null,
                className: `px-4 py-2 rounded-lg ${
                  playbackMode !== null
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-green-600 hover:bg-green-700'
                } text-white`
              }, playbackMode !== null
                ? `Playing (${remainingLoops + 1} loops left)`
                : 'Play Fast Version'),
                
              playbackMode && React.createElement('button', {
                onClick: stopPlayback,
                className: 'px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white'
              }, '■ Stop'),
              
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
                    onClick: () => playAudioData(section.audio_data.dialogue[index]),
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
