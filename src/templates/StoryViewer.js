const StoryViewer = ({ storyData, title, targetLanguage }) => {
  const [activeSection, setActiveSection] = React.useState(() => {
    // Check if there's a hash in the URL on initial load
    const hash = window.location.hash.slice(1);
    if (hash) {
      const sectionIndex = Object.keys(storyData).findIndex(name => name === hash);
      return sectionIndex >= 0 ? sectionIndex : null;
    }
    return null;
  });
  const [isPlaying, setIsPlaying] = React.useState({});
  const [loopCount, setLoopCount] = React.useState(12);
  const [remainingLoops, setRemainingLoops] = React.useState(0);
  const [playbackMode, setPlaybackMode] = React.useState(null);
  const [showCopyNotification, setShowCopyNotification] = React.useState(false);

  // for play all dialgoue
  const [isPlayingAll, setIsPlayingAll] = React.useState(false);
  const normalAudioQueue = React.useRef([]);
  const currentNormalAudioIndex = React.useRef(0);

  const audioRef = React.useRef(null);
  const audioQueue = React.useRef([]);
  const fastAudioQueue = React.useRef([]);
  const activeSectionAudio = React.useRef(null);

  React.useEffect(() => {
    // Hide loading message when component mounts
    if (window.hideLoadingMessage) {
      window.hideLoadingMessage();
    }
  }, []);

  const stopPlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.onended = null;
    }
    audioQueue.current = [];
    fastAudioQueue.current = [];
    activeSectionAudio.current = null;
    currentNormalAudioIndex.current = 0;
    setIsPlaying({});
    setPlaybackMode(null);
    setIsPlayingAll(false);
    setRemainingLoops(0);
  };

  const playNextNormalAudio = () => {
    if (audioQueue.current.length === 0) {
      setIsPlayingAll(false);
      setPlaybackMode(null);
      return;
    }
  
    const nextAudio = audioQueue.current.shift();
    playAudioData(nextAudio);
  };
  

  const playAllNormal = () => {
    if (isPlayingAll) {
      // Stop playback if already playing
      stopPlayback();
      setIsPlayingAll(false);
      return;
    }

    stopPlayback();
    setIsPlayingAll(true);
    setPlaybackMode('normal');

    audioQueue.current = resetNormalAudioQueue();
    playNextNormalAudio();
  };

  // 3. Function to play next audio in queue
  const playNextInNormalQueue = () => {
    if (!isPlayingAll || currentNormalAudioIndex.current >= normalAudioQueue.current.length) {
      setIsPlayingAll(false);
      return;
    }

    const current = normalAudioQueue.current[currentNormalAudioIndex.current];
    audioRef.current.src = `data:audio/mp3;base64,${current.audioData}`;
    audioRef.current.onended = () => {
      currentNormalAudioIndex.current++;
      playNextInNormalQueue();
    };
    audioRef.current.play().catch(error => {
      console.error('PlayAllNormal playback error:', error);
      setIsPlayingAll(false);
    });
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setShowCopyNotification(true);
      setTimeout(() => setShowCopyNotification(false), 1000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const copyPrompt = async (translatedPhrase, event) => {
    // Prevent the default link behavior initially
    if (event) {
      event.preventDefault();
    }

    // Create a temporary div to decode HTML entities
    const decoder = document.createElement('div');
    decoder.innerHTML = translatedPhrase;
    const decodedPhrase = decoder.textContent;

    const prompt = `Given this ${targetLanguage} phrase "${decodedPhrase}", please help me understand it, break down its grammar, and explain any idiomatic expressions.`;

    try {
      await navigator.clipboard.writeText(prompt);
      setShowCopyNotification(true);
      setTimeout(() => setShowCopyNotification(false), 1000);

      // Open claude.ai in a new tab after copying
      window.open('https://claude.ai', '_blank');
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
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

  const resetNormalAudioQueue = () => {
    const queue = [];
    Object.keys(storyData).forEach(part => {
      const dialogueAudios = storyData[part].audio_data?.dialogue || [];
      dialogueAudios.forEach(audioData => {
        queue.push(audioData); // Just base64 strings
      });
    });
    return queue;
  };

  const resetFastAudioQueue = (mode = 'single') => {
    if (mode === 'all') {
      const sectionKeys = Object.keys(storyData);
      return sectionKeys.map((key, index) => ({
        audio: storyData[key].audio_data.fast_dialogue,
        isLastInLoop: index === sectionKeys.length - 1
      }));
    } else {
      return [{
        audio: activeSectionAudio.current,
        isLastInLoop: true
      }];
    }
  };

  const playFastAudio = (sectionIndex, loops = loopCount) => {
    stopPlayback();
    setIsPlaying(prev => ({ ...prev, [sectionIndex]: true }));
    setPlaybackMode('fast');
    setRemainingLoops(loops - 1);

    const fastAudio = storyData[Object.keys(storyData)[sectionIndex]].audio_data.fast_dialogue;
    activeSectionAudio.current = fastAudio;

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
          playNextNormalAudio();
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
    stopPlayback();
    setIsPlaying(prev => ({ ...prev, 'all': true }));
    setPlaybackMode('all');
    setRemainingLoops(loops);

    const queue = resetFastAudioQueue('all');
    fastAudioQueue.current = queue;

    playNextFastAudio();
  };

  const renderWiktionaryLinks = (linksHtml) => {
    // Apply Tailwind styles to the links via a wrapper class
    const enhancedHtml = linksHtml.replace(/<a /g, '<a class="text-blue-600 hover:text-blue-800 underline hover:bg-blue-50 rounded px-0.5 transition-colors duration-150" ');
    return React.createElement('div', {
      className: 'text-lg mb-2 leading-relaxed',
      dangerouslySetInnerHTML: { __html: enhancedHtml }
    });
  };

  return React.createElement('div', { className: 'min-h-screen bg-gray-100' },
    // Add this as the first child element inside the main div
    showCopyNotification && React.createElement('div', {
      className: 'fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-75 text-white px-4 py-2 rounded-lg text-sm z-20'
    }, 'Copied!'),
    React.createElement('audio', { ref: audioRef, className: 'hidden' }),


    // Header with global controls
    React.createElement('header', { className: 'bg-blue-600 text-white p-4 sticky top-0 z-10' },
      React.createElement('div', { className: 'flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4' },
        React.createElement('div', { className: 'flex justify-between items-center' },
          // Left side - existing title hierarchy
          React.createElement('h1', { className: 'text-xl font-bold flex items-center gap-2' },
            React.createElement('a', {
              href: `https://storage.googleapis.com/audio-language-trainer-stories/index.html#${targetLanguage.toLowerCase()}`,
              className: 'hover:text-blue-500 transition-colors'
            }, `${targetLanguage} Stories`),
            React.createElement('span', { className: 'text-gray-400' }, '>'),
            title || 'Language Learning Story'
          ),
          // In StoryViewer.js, update the speaking challenges button
          React.createElement('a', {
            href: `https://storage.googleapis.com/audio-language-trainer-stories/${targetLanguage.toLowerCase()}/story_${title.toLowerCase().replace(/\s+/g, '_')}/challenges.html`,
            // Added ml-4 for left margin
            className: 'ml-4 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors text-sm flex items-center gap-2 font-medium'
          },
            React.createElement('span', null, '🎤'),
            'Try Speaking Practice'
          )
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
            onClick: () => {
              if (playbackMode === 'all') {
                stopPlayback();
              } else {
                playAllFastAudio(loopCount);
              }
            },
            className: `px-4 py-2 rounded-lg ${
              playbackMode === 'normal'
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700'
            } text-white`
          }, playbackMode === 'all' ? '■ Stop' : `Play All Fast`),

        React.createElement('button', {
          onClick: () => {
            if (playbackMode === 'normal') {
              stopPlayback();
            } else {
              playAllNormal();
            }
          },
          className: `px-4 py-2 rounded-lg mr-2 ${
            playbackMode === 'fast'
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-green-800 hover:bg-green-900'
          } text-white`
        }, playbackMode === 'normal' ? '■ Stop' : 'Play All'),),
        // Add this to the header section of StoryViewer
        React.createElement('div', { className: 'flex items-center gap-4' },
          React.createElement('a', {
            href: '/audio-language-trainer-stories/m4a_downloads.html',
            className: 'px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center gap-2'
          },
            React.createElement('span', { className: 'text-sm' }, '🎧'),
            'Download Audio Files'
          )
        )
      )
    ),

    // Main content
    React.createElement('main', { className: 'max-w-4xl mx-auto p-4' },
      Object.entries(storyData).map(([sectionName, section], sectionIndex) =>
        React.createElement('div', {
          key: sectionName,
          id: sectionName,
          className: 'mb-6 bg-white rounded-lg shadow-md scroll-mt-20'
        },
          // Section header with anchor
          React.createElement('a', {
            href: `#${sectionName}`,
            onClick: (e) => {
              e.preventDefault();
              const newState = activeSection === sectionIndex ? null : sectionIndex;
              setActiveSection(newState);
              if (newState !== null) {
                window.location.hash = sectionName;
                // Smooth scroll to the section
                const element = document.getElementById(sectionName);
                if (element) {
                  element.scrollIntoView({ behavior: 'smooth' });
                }
              } else {
                // Remove hash when closing section
                history.pushState('', document.title, window.location.pathname + window.location.search);
              }
            },
            className: 'w-full p-4 flex items-center justify-between text-left bg-gray-50 rounded-t-lg hover:bg-gray-100 block no-underline text-current'
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
              className: 'flex flex-col sm:grid sm:grid-cols-4 gap-2 sm:gap-4 mb-4 p-2 bg-gray-50 rounded-lg'
            },
              React.createElement('button', {
                onClick: () => playAllDialogue(sectionIndex, section.audio_data.dialogue),
                disabled: playbackMode !== null,
                className: `w-full px-4 py-3 sm:py-2 rounded-lg text-lg sm:text-base ${playbackMode !== null
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
                  } text-white`
              }, playbackMode === 'normal' ? 'Playing...' : 'Play Dialogue'),

              React.createElement('button', {
                onClick: () => playFastAudio(sectionIndex, loopCount),
                disabled: playbackMode !== null,
                className: `w-full px-4 py-3 sm:py-2 rounded-lg text-lg sm:text-base ${playbackMode !== null
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
                  } text-white`
              }, playbackMode !== null
                ? `Playing (${remainingLoops + 1} loops left)`
                : 'Play Fast Version'),

              playbackMode && React.createElement('button', {
                onClick: stopPlayback,
                className: 'w-full px-4 py-3 sm:py-2 rounded-lg text-lg sm:text-base bg-red-600 hover:bg-red-700 text-white'
              }, '■ Stop'),

            ),

            // Dialogue section
            section.translated_dialogue.map((utterance, index) =>
              React.createElement('div', {
                key: index,
                className: 'mb-4 p-3 bg-gray-50 rounded-lg'
              },
                React.createElement('div', { className: 'sm:flex sm:items-center sm:justify-between block' },
                  React.createElement('div', { className: 'flex-grow' },
                    React.createElement('p', { className: 'text-sm text-gray-600 mb-1' },
                      utterance.speaker
                    ),
                    renderWiktionaryLinks(utterance.wiktionary_links),
                    React.createElement('p', { className: 'mt-2 text-gray-600' },
                      section.dialogue[index].text
                    )
                  ),
                  React.createElement('div', { className: 'flex items-center gap-2 mt-2 sm:mt-0' },
                    section.audio_data?.dialogue[index] && React.createElement('button', {
                      onClick: () => playAudioData(section.audio_data.dialogue[index]),
                      disabled: playbackMode !== null,
                      className: `p-2 rounded-full hover:bg-gray-200 ${playbackMode !== null ? 'opacity-50 cursor-not-allowed' : ''
                        }`
                    }, '🔊'),
                    React.createElement('button', {
                      onClick: () => copyToClipboard(utterance.text),
                      className: 'p-2 rounded-full hover:bg-gray-200',
                      title: 'Copy phrase'
                    }, '📋'),
                    React.createElement('button', {
                      onClick: (e) => copyPrompt(utterance.text, e),
                      className: 'p-2 rounded-full hover:bg-gray-200',
                      title: 'Copy as prompt and open Claude'
                    }, '💡')
                  )
                )
              )
            )
          )
        )
      )

    )
  );
};