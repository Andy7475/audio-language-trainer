import React, { useState, useRef } from 'react';
import { ChevronDown, ChevronRight, Play, Volume2, Book } from 'lucide-react';

const StoryViewer = ({ storyData, targetLanguage }) => {
  const [activeSection, setActiveSection] = useState(null);
  const [activePhraseIndex, setActivePhraseIndex] = useState(null);
  const [showTranslation, setShowTranslation] = useState({});
  const audioRef = useRef(null);

  // Create wiktionary links for a given text
  const createWiktionaryLinks = (text) => {
    return text.split(' ').map((word, index) => {
      const cleanWord = word.toLowerCase().replace(/[^a-zA-Z0-9]/g, '');
      if (cleanWord) {
        return (
          <React.Fragment key={index}>
            <a
              href={`https://en.wiktionary.org/wiki/${encodeURIComponent(cleanWord)}#${targetLanguage}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              {word}
            </a>
            {' '}
          </React.Fragment>
        );
      }
      return word + ' ';
    });
  };

  // Handle audio playback
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

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Hidden audio element for playback */}
      <audio ref={audioRef} className="hidden" />
      
      {/* Mobile-friendly header */}
      <header className="bg-blue-600 text-white p-4 sticky top-0 z-10">
        <h1 className="text-xl font-bold">Language Learning Story</h1>
      </header>

      {/* Main content area */}
      <main className="max-w-4xl mx-auto p-4">
        {Object.entries(storyData).map(([sectionName, section], sectionIndex) => (
          <div key={sectionName} className="mb-6 bg-white rounded-lg shadow-md">
            {/* Section header */}
            <button
              onClick={() => setActiveSection(activeSection === sectionIndex ? null : sectionIndex)}
              className="w-full p-4 flex items-center justify-between text-left bg-gray-50 rounded-t-lg hover:bg-gray-100"
            >
              <h2 className="text-lg font-semibold capitalize">
                {sectionName.replace(/_/g, ' ')}
              </h2>
              {activeSection === sectionIndex ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            </button>

            {/* Section content */}
            {activeSection === sectionIndex && (
              <div className="p-4">
                {/* Practice Phrases */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-4">Practice Phrases</h3>
                  {section.translated_phrase_list.map(([english, target], index) => (
                    <div key={index} className="mb-4 p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div className="flex-grow">
                          <p className="text-lg font-medium">{createWiktionaryLinks(target)}</p>
                          <button
                            onClick={() => toggleTranslation(index)}
                            className="text-sm text-blue-600 hover:underline mt-1"
                          >
                            {showTranslation[index] ? 'Hide' : 'Show'} translation
                          </button>
                          {showTranslation[index] && (
                            <p className="mt-2 text-gray-600">{english}</p>
                          )}
                        </div>
                        {section.audio_data?.phrases[index] && (
                          <div className="flex gap-2">
                            <button
                              onClick={() => playAudio(section.audio_data.phrases[index].normal)}
                              className="p-2 rounded-full hover:bg-gray-200"
                              title="Play normal speed"
                            >
                              <Volume2 className="w-5 h-5" />
                            </button>
                            <button
                              onClick={() => playAudio(section.audio_data.phrases[index].slow)}
                              className="p-2 rounded-full hover:bg-gray-200"
                              title="Play slow speed"
                            >
                              <Play className="w-5 h-5" />
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Dialogue */}
                <div>
                  <h3 className="text-lg font-semibold mb-4">Dialogue</h3>
                  {section.translated_dialogue.map((utterance, index) => (
                    <div key={index} className="mb-4 p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div className="flex-grow">
                          <p className="text-sm text-gray-600 mb-1">{utterance.speaker}</p>
                          <p className="text-lg">{createWiktionaryLinks(utterance.text)}</p>
                          <p className="mt-2 text-gray-600">{section.dialogue[index].text}</p>
                        </div>
                        {section.audio_data?.dialogue[index] && (
                          <button
                            onClick={() => playAudio(section.audio_data.dialogue[index])}
                            className="p-2 rounded-full hover:bg-gray-200"
                          >
                            <Volume2 className="w-5 h-5" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </main>

      {/* Footer with album link if available */}
      <footer className="bg-gray-800 text-white p-4 mt-8">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <p className="text-sm">Download the audio album for practice</p>
          <a 
            href="#" 
            className="flex items-center gap-2 bg-blue-600 px-4 py-2 rounded-lg hover:bg-blue-700"
            onClick={(e) => {
              e.preventDefault();
              // Handle album link - could open a modal with QR code or download link
            }}
          >
            <Book className="w-5 h-5" />
            <span>Get Album</span>
          </a>
        </div>
      </footer>
    </div>
  );
};

export default StoryViewer;