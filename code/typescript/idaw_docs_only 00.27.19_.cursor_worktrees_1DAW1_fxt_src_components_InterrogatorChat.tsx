import { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'system';
  message: string;
  timestamp: Date;
}

interface InterrogatorChatProps {
  onReady: (intent: any) => void;
}

export const InterrogatorChat: React.FC<InterrogatorChatProps> = ({ onReady }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoading(true);

    // Add user message to chat
    const newUserMessage: Message = {
      role: 'user',
      message: userMessage,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newUserMessage]);

    try {
      // Call interrogate API
      const response = await fetch('http://127.0.0.1:8000/interrogate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to interrogate');
      }

      const data = await response.json();

      // Update session ID
      if (data.session_id) {
        setSessionId(data.session_id);
      }

      // Add system response
      if (data.question) {
        const systemMessage: Message = {
          role: 'system',
          message: data.question,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, systemMessage]);
      }

      // Check if ready
      if (data.ready && data.intent) {
        setIsReady(true);
        onReady(data.intent);

        const readyMessage: Message = {
          role: 'system',
          message: data.message || 'Ready to generate music!',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, readyMessage]);
      }
    } catch (error) {
      console.error('Error in interrogation:', error);
      const errorMessage: Message = {
        role: 'system',
        message: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resetChat = () => {
    setMessages([]);
    setSessionId(null);
    setIsReady(false);
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '500px',
      border: '1px solid rgba(0, 0, 0, 0.1)',
      borderRadius: '8px',
      backgroundColor: '#fff',
      overflow: 'hidden'
    }}>
      {/* Chat header */}
      <div style={{
        padding: '12px',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        borderBottom: '1px solid rgba(0, 0, 0, 0.1)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h4 style={{ margin: 0, fontSize: '1em', fontWeight: 'bold' }}>
          💬 Interrogator
        </h4>
        <button
          onClick={resetChat}
          style={{
            padding: '4px 8px',
            fontSize: '0.8em',
            backgroundColor: 'transparent',
            border: '1px solid rgba(0, 0, 0, 0.2)',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Reset
        </button>
      </div>

      {/* Messages area */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '15px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        {messages.length === 0 && (
          <div style={{
            textAlign: 'center',
            color: '#666',
            fontStyle: 'italic',
            marginTop: '20px'
          }}>
            Start the conversation by describing how you're feeling...
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: '10px'
            }}
          >
            <div style={{
              maxWidth: '70%',
              padding: '10px 14px',
              borderRadius: '12px',
              backgroundColor: msg.role === 'user'
                ? 'rgba(99, 102, 241, 0.1)'
                : 'rgba(0, 0, 0, 0.05)',
              border: `1px solid ${msg.role === 'user'
                ? 'rgba(99, 102, 241, 0.3)'
                : 'rgba(0, 0, 0, 0.1)'}`,
              fontSize: '0.9em',
              lineHeight: '1.4'
            }}>
              {msg.message}
            </div>
          </div>
        ))}

        {isLoading && (
          <div style={{
            display: 'flex',
            justifyContent: 'flex-start'
          }}>
            <div style={{
              padding: '10px 14px',
              borderRadius: '12px',
              backgroundColor: 'rgba(0, 0, 0, 0.05)',
              fontSize: '0.9em',
              fontStyle: 'italic',
              color: '#666'
            }}>
              Thinking...
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div style={{
        padding: '12px',
        borderTop: '1px solid rgba(0, 0, 0, 0.1)',
        backgroundColor: 'rgba(0, 0, 0, 0.02)'
      }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isReady ? "Ready to generate! Type to continue..." : "Describe how you're feeling..."}
            disabled={isLoading}
            style={{
              flex: 1,
              padding: '10px',
              border: '1px solid rgba(0, 0, 0, 0.2)',
              borderRadius: '6px',
              fontSize: '0.9em',
              outline: 'none'
            }}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputValue.trim()}
            style={{
              padding: '10px 20px',
              backgroundColor: isLoading ? '#ccc' : '#6366f1',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              fontSize: '0.9em',
              fontWeight: 'bold'
            }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};
