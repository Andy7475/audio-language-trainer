// StoryViewer component without JSX or imports
const StoryViewer = ({ storyData, targetLanguage, title }) => {
  const [activeSection, setActiveSection] = React.useState(null);
  const [showTranslation, setShowTranslation] = React.useState({});
  const audioRef = React.useRef(null);

  const createWiktionaryLinks = (text) => {
    return text.split(' ').map((word, index) => {
      const cleanWord = word.toLowerCase().replace(/[^a-zA-Z0-9]/g, '');
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

  const playAudio = (audioData) => {
    if (audioRef.current) {
      audioRef.current.src = `data:audio/mp3;base64,${audioData}`;
      audioRef.current.play();
    }
  };

  const toggleTranslation = (index) => {
    setShowTranslation(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return React.createElement('div', { className: 'min-h-screen bg-gray-100' },
    React.createElement('audio', { ref: audioRef, className: 'hidden' }),
    
    // Header
    React.createElement('header', { className: 'bg-blue-600 text-white p-4 sticky top-0 z-10' },
      React.createElement('h1', { className: 'text-xl font-bold' }, title || 'Language Learning Story')
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

          // Section content (only shown when active)
          activeSection === sectionIndex && React.createElement('div', { className: 'p-4' },
            // Practice Phrases
            React.createElement('div', { className: 'mb-6' },
              React.createElement('h3', { className: 'text-lg font-semibold mb-4' }, 'Practice Phrases'),
              section.translated_phrase_list.map(([english, target], index) =>
                React.createElement('div', {
                  key: index,
                  className: 'mb-4 p-3 bg-gray-50 rounded-lg'
                },
                  React.createElement('div', { className: 'flex items-center justify-between' },
                    React.createElement('div', { className: 'flex-grow' },
                      React.createElement('p', { className: 'text-lg font-medium' },
                        createWiktionaryLinks(target)
                      ),
                      React.createElement('button', {
                        onClick: () => toggleTranslation(index),
                        className: 'text-sm text-blue-600 hover:underline mt-1'
                      }, `${showTranslation[index] ? 'Hide' : 'Show'} translation`),
                      showTranslation[index] && React.createElement('p', {
                        className: 'mt-2 text-gray-600'
                      }, english)
                    ),
                    section.audio_data?.phrases[index] && React.createElement('div', { className: 'flex gap-2' },
                      React.createElement('button', {
                        onClick: () => playAudio(section.audio_data.phrases[index].normal),
                        className: 'p-2 rounded-full hover:bg-gray-200',
                        title: 'Play normal speed'
                      }, '🔊'),
                      React.createElement('button', {
                        onClick: () => playAudio(section.audio_data.phrases[index].slow),
                        className: 'p-2 rounded-full hover:bg-gray-200',
                        title: 'Play slow speed'
                      }, '🐢')
                    )
                  )
                )
              )
            ),

            // Dialogue section
            section.translated_dialogue.length > 0 && React.createElement('div', null,
              React.createElement('h3', { className: 'text-lg font-semibold mb-4' }, 'Dialogue'),
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
                      className: 'p-2 rounded-full hover:bg-gray-200'
                    }, '🔊')
                  )
                )
              )
            )
          )
        )
      )
    ),

    // Footer
React.createElement('footer', { className: 'bg-gray-800 text-white p-4 mt-8' },
  React.createElement('div', { className: 'max-w-4xl mx-auto flex items-center justify-between' },
    React.createElement('p', { className: 'text-sm' }, 'Audio Language Trainer'),
    React.createElement('a', {
      href: 'https://github.com/Andy7475/audio-language-trainer',
      target: '_blank',
      rel: 'noopener noreferrer',
      className: 'flex items-center gap-2 bg-blue-600 px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors',
    }, [
      React.createElement('span', { className: 'text-lg' }, '🔗'),
      'View on GitHub'
    ])
  )
),
  );
};