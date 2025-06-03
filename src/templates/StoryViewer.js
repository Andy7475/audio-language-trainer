const StoryViewer = ({ storyData, title, targetLanguage, collectionName, collectionRaw }) => {
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
  const [showEnglish, setShowEnglish] = React.useState(true);

  // for play all dialgoue
  const [isPlayingAll, setIsPlayingAll] = React.useState(false);
  const normalAudioQueue = React.useRef([]);
  const currentNormalAudioIndex = React.useRef(0);

  const audioRef = React.useRef(null);
  const audioQueue = React.useRef([]);
  const fastAudioQueue = React.useRef([]);
  const activeSectionAudio = React.useRef(null);

  const story_folder = "story_" + title.replace(/\s+/g, '_').toLowerCase();

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

  return React.createElement('div', { className: 'app-container' },
    // Add this as the first child element inside the main div
    showCopyNotification && React.createElement('div', {
      className: 'copy-notification'
    }, 'Copied!'),
    React.createElement('audio', { ref: audioRef, className: 'hidden' }),

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
          React.createElement('span', { className: 'breadcrumb-current' }, title)
        ),
        React.createElement('div', { className: 'controls-section' },
          React.createElement('div', { className: 'controls-group' },
            React.createElement('label', { htmlFor: 'loopCount', className: 'toggle-label' },
              'Loops:'
            ),
            React.createElement('input', {
              id: 'loopCount',
              type: 'number',
              min: 1,
              max: 50,
              value: loopCount,
              onChange: (e) => setLoopCount(Number(e.target.value)),
              className: 'loop-input'
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
            className: `button ${playbackMode === 'normal' ? 'secondary' : 'primary'}`
          }, playbackMode === 'all' ? '■ Stop' : `Play All Fast`),
          React.createElement('button', {
            onClick: () => {
              if (playbackMode === 'normal') {
                stopPlayback();
              } else {
                playAllNormal();
              }
            },
            className: `button ${playbackMode === 'fast' ? 'secondary' : 'primary'}`
          }, playbackMode === 'normal' ? '■ Stop' : 'Play All'),
          React.createElement('a', {
            href: `https://storage.googleapis.com/audio-language-trainer-stories/${targetLanguage.toLowerCase()}/${(collectionRaw || collectionName).toLowerCase()}/${story_folder}/challenges.html`,
            className: 'button'
          }, 'Speaking Challenges')
        )
      )
    ),

    // Main content
    React.createElement('main', { className: 'main-content' },
      Object.entries(storyData).map(([sectionName, section], sectionIndex) =>
        React.createElement('div', {
          key: sectionName,
          id: sectionName,
          className: 'card'
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
                const element = document.getElementById(sectionName);
                if (element) {
                  setTimeout(() => {
                    element.scrollIntoView({ 
                      behavior: 'smooth',
                      block: 'start'
                    });
                    window.scrollBy(0, -120);
                  }, 50);
                }
              } else {
                history.pushState('', document.title, window.location.pathname + window.location.search);
              }
            },
            className: 'section-anchor'
          },
            React.createElement('h2', { className: 'section-title' },
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
              className: 'section-image'
            },
              React.createElement('img', {
                src: `data:image/jpeg;base64,${section.image_data}`,
                alt: `Story part ${sectionIndex + 1}`,
                className: 'w-full h-auto'
              })
            ),

            // Controls for dialogue playback
            React.createElement('div', {
              className: 'button-grid mb-4 p-2 bg-gray-50 rounded-lg'
            },
              React.createElement('button', {
                onClick: () => playAllDialogue(sectionIndex, section.audio_data.dialogue),
                disabled: playbackMode !== null,
                className: `button ${playbackMode !== null ? 'secondary' : 'primary'}`
              }, playbackMode === 'normal' ? 'Playing...' : 'Play Dialogue'),

              React.createElement('button', {
                onClick: () => playFastAudio(sectionIndex, loopCount),
                disabled: playbackMode !== null,
                className: `button ${playbackMode !== null ? 'secondary' : 'primary'}`
              }, playbackMode !== null
                ? `Playing (${remainingLoops + 1} loops left)`
                : 'Play Fast Version'),

              playbackMode && React.createElement('button', {
                onClick: stopPlayback,
                className: 'button danger'
              }, '■ Stop'),

              // Add English text toggle here
              React.createElement('div', { className: 'toggle-controls' },
                React.createElement('label', { className: 'toggle-label' }, 'Show English:'),
                React.createElement('button', {
                  onClick: () => setShowEnglish(!showEnglish),
                  className: `button ${showEnglish ? 'primary' : 'secondary'}`
                }, showEnglish ? 'On' : 'Off')
              )
            ),

            // Dialogue section
            section.translated_dialogue.map((utterance, index) =>
              React.createElement('div', {
                key: index,
                className: 'dialogue-item'
              },
                React.createElement('div', { className: 'dialogue-layout' },
                  React.createElement('div', { className: 'dialogue-content' },
                    React.createElement('p', { className: 'dialogue-speaker' },
                      utterance.speaker
                    ),
                    React.createElement('p', { 
                      className: 'dialogue-text',
                      onClick: () => copyToClipboard(utterance.text)
                    },
                      utterance.text
                    ),
                    renderWiktionaryLinks(utterance.wiktionary_links),
                    showEnglish && React.createElement('p', { className: 'dialogue-english' },
                      section.dialogue[index].text
                    )
                  ),
                  React.createElement('div', { className: 'dialogue-controls' },
                    section.audio_data?.dialogue[index] && React.createElement('button', {
                      onClick: () => playAudioData(section.audio_data.dialogue[index]),
                      disabled: playbackMode !== null,
                      className: `button secondary ${playbackMode !== null ? 'opacity-50 cursor-not-allowed' : ''}`
                    }, '🔊'),
                    React.createElement('button', {
                      onClick: () => copyToClipboard(utterance.text),
                      className: 'button secondary',
                      title: 'Copy phrase'
                    }, '📋'),
                    React.createElement('button', {
                      onClick: (e) => copyPrompt(utterance.text, e),
                      className: 'button secondary',
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