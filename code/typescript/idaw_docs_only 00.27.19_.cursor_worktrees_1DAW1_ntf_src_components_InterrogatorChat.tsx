import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'assistant';
  message: string;
  timestamp?: Date;
}

interface InterrogatorChatProps {
  onReady: (intent: any) => void;
  apiUrl?: string;
}

export const InterrogatorChat: React.FC<InterrogatorChatProps> = ({ onReady, apiUrl = 'http://localhost:8000' }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Start conversation with opening question
  useEffect(() => {
    if (messages.length === 0) {
      const openingMessage: Message = {
        role: 'assistant',
        message: "Hi! I'm here to help you create music that captures your emotions. What are you feeling right now?",
        timestamp: new Date(),
      };
      setMessages([openingMessage]);
    }
  }, []);

  const sendMessage = async () => {
    if (!inputMessage.trim() || loading || ready) return;

    const userMessage: Message = {
      role: 'user',
      message: inputMessage.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await fetch(`${apiUrl}/interrogate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage.trim(),
          session_id: sessionId,
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Store session ID
        if (data.session_id) {
          setSessionId(data.session_id);
        }

        // Add assistant response
        const assistantMessage: Message = {
          role: 'assistant',
          message: data.question || data.message || 'I understand.',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMessage]);

        // Check if ready to generate
        if (data.ready && data.intent) {
          setReady(true);
          onReady(data.intent);
        }
      } else {
        throw new Error('API request failed');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        message: "I'm having trouble understanding. Could you try rephrasing that?",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      maxHeight: '500px',
      backgroundColor: 'rgba(0, 0, 0, 0.2)',
      borderRadius: '8px',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      overflow: 'hidden'
    }}>
      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '15px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: 'flex',
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: '8px'
            }}
          >
            <div
              style={{
                maxWidth: '70%',
                padding: '10px 15px',
                borderRadius: '12px',
                backgroundColor: msg.role === 'user'
                  ? 'rgba(76, 175, 80, 0.3)'
                  : 'rgba(255, 255, 255, 0.1)',
                color: '#ffffff',
                fontSize: '0.9em',
                lineHeight: '1.4',
                wordWrap: 'break-word'
              }}
            >
              {msg.message}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{
            display: 'flex',
            justifyContent: 'flex-start'
          }}>
            <div style={{
              padding: '10px 15px',
              borderRadius: '12px',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              color: 'rgba(255, 255, 255, 0.6)',
              fontSize: '0.9em'
            }}>
              <span>...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      {!ready && (
        <div style={{
          padding: '15px',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <div style={{
            display: 'flex',
            gap: '10px'
          }}>
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your response..."
              disabled={loading}
              style={{
                flex: 1,
                padding: '10px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '6px',
                color: '#ffffff',
                fontSize: '0.9em',
                outline: 'none'
              }}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !inputMessage.trim()}
              style={{
                padding: '10px 20px',
                backgroundColor: loading || !inputMessage.trim() ? 'rgba(76, 175, 80, 0.3)' : '#4caf50',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: loading || !inputMessage.trim() ? 'not-allowed' : 'pointer',
                fontSize: '0.9em',
                fontWeight: 'bold'
              }}
            >
              Send
            </button>
          </div>
        </div>
      )}

      {ready && (
        <div style={{
          padding: '15px',
          backgroundColor: 'rgba(76, 175, 80, 0.2)',
          borderTop: '1px solid rgba(76, 175, 80, 0.3)',
          textAlign: 'center',
          color: '#4caf50',
          fontWeight: 'bold'
        }}>
          ✓ Ready to generate music!
        </div>
      )}
    </div>
  );
};
