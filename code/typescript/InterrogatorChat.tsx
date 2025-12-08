import React, { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'system';
  content: string;
  timestamp: Date;
}

interface InterrogatorChatProps {
  onReady?: (intent: {
    base_emotion: string;
    intensity: string;
    specific_emotion?: string;
  }) => void;
  onSendMessage?: (message: string, sessionId?: string) => Promise<any>;
  initialMessage?: string;
}

export const InterrogatorChat: React.FC<InterrogatorChatProps> = ({
  onReady,
  onSendMessage,
  initialMessage,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [isReady, setIsReady] = useState(false);
  const [readyIntent, setReadyIntent] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (initialMessage && messages.length === 0) {
      handleSendMessage(initialMessage);
    }
  }, [initialMessage]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (messageText?: string) => {
    const text = messageText || inputValue.trim();
    if (!text || isLoading) return;

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send to API
      const result = await (onSendMessage || defaultSendMessage)(text, sessionId);

      if (result.session_id) {
        setSessionId(result.session_id);
      }

      if (result.ready) {
        setIsReady(true);
        setReadyIntent(result.intent);
        if (onReady && result.intent) {
          onReady(result.intent);
        }
        // Add ready message
        const readyMessage: Message = {
          role: 'system',
          content: result.message || 'Ready to generate music!',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, readyMessage]);
      } else if (result.question) {
        // Add system question
        const systemMessage: Message = {
          role: 'system',
          content: result.question,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, systemMessage]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'system',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const defaultSendMessage = async (
    message: string,
    sessionId?: string
  ): Promise<any> => {
    // Default implementation - should be overridden by parent
    const response = await fetch('http://localhost:8000/interrogate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    return response.json();
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '500px',
        border: '1px solid #ddd',
        borderRadius: '8px',
        backgroundColor: '#fff',
        overflow: 'hidden',
      }}
    >
      {/* Messages area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '15px',
          backgroundColor: '#f9f9f9',
        }}
      >
        {messages.length === 0 && (
          <div
            style={{
              textAlign: 'center',
              color: '#666',
              fontStyle: 'italic',
              marginTop: '20px',
            }}
          >
            Start a conversation about how you're feeling...
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              marginBottom: '15px',
              display: 'flex',
              flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
            }}
          >
            <div
              style={{
                maxWidth: '70%',
                padding: '10px 15px',
                borderRadius: '12px',
                backgroundColor:
                  msg.role === 'user' ? '#2196f3' : '#e0e0e0',
                color: msg.role === 'user' ? '#fff' : '#000',
                wordWrap: 'break-word',
              }}
            >
              <div>{msg.content}</div>
              <div
                style={{
                  fontSize: '0.7em',
                  opacity: 0.7,
                  marginTop: '5px',
                }}
              >
                {formatTime(msg.timestamp)}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-start',
              marginBottom: '15px',
            }}
          >
            <div
              style={{
                padding: '10px 15px',
                borderRadius: '12px',
                backgroundColor: '#e0e0e0',
                color: '#666',
              }}
            >
              <span style={{ animation: 'pulse 1.5s ease-in-out infinite' }}>
                ...
              </span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div
        style={{
          borderTop: '1px solid #ddd',
          padding: '15px',
          backgroundColor: '#fff',
        }}
      >
        {isReady && readyIntent && (
          <div
            style={{
              marginBottom: '10px',
              padding: '10px',
              backgroundColor: '#4caf50',
              color: 'white',
              borderRadius: '4px',
              fontSize: '0.9em',
            }}
          >
            ✅ Ready to generate! Emotion: {readyIntent.base_emotion} (
            {readyIntent.intensity}): {readyIntent.specific_emotion}
          </div>
        )}
        <div style={{ display: 'flex', gap: '10px' }}>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={
              isReady
                ? "Type to continue conversation..."
                : "Describe how you're feeling..."
            }
            disabled={isLoading}
            style={{
              flex: 1,
              padding: '10px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '0.9em',
            }}
          />
          <button
            onClick={() => handleSendMessage()}
            disabled={isLoading || !inputValue.trim()}
            style={{
              padding: '10px 20px',
              backgroundColor: '#2196f3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              opacity: isLoading || !inputValue.trim() ? 0.6 : 1,
            }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};
