import React, { useState, useEffect } from 'react';

interface RuleCategory {
  [technique: string]: string;
}

interface RuleBreakingOptions {
  harmony: RuleCategory;
  rhythm: RuleCategory;
  arrangement: RuleCategory;
  production: RuleCategory;
}

interface RuleBreakerProps {
  selectedEmotion?: string;
  onRuleSelected?: (rule: { category: string; technique: string; effect: string }) => void;
}

export const RuleBreaker: React.FC<RuleBreakerProps> = ({
  selectedEmotion,
  onRuleSelected,
}) => {
  const [rules, setRules] = useState<RuleBreakingOptions | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedRules, setSelectedRules] = useState<Set<string>>(new Set());
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);

  useEffect(() => {
    loadRules();
  }, []);

  useEffect(() => {
    if (selectedEmotion) {
      getSuggestions(selectedEmotion);
    }
  }, [selectedEmotion]);

  const loadRules = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/rules/breaking');
      const data = await response.json();
      if (data.success) {
        setRules(data.rules);
      }
    } catch (err) {
      setError('Failed to load rule-breaking options');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getSuggestions = async (emotion: string) => {
    try {
      const response = await fetch('http://localhost:8000/rules/suggest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emotion }),
      });
      const data = await response.json();
      if (data.success) {
        setSuggestions(data.suggestions);
      }
    } catch (err) {
      console.error('Failed to get suggestions:', err);
    }
  };

  const toggleRule = (category: string, technique: string, effect: string) => {
    const ruleKey = `${category}:${technique}`;
    const newSelected = new Set(selectedRules);

    if (newSelected.has(ruleKey)) {
      newSelected.delete(ruleKey);
    } else {
      newSelected.add(ruleKey);
      onRuleSelected?.({ category, technique, effect });
    }

    setSelectedRules(newSelected);
  };

  const categoryColors: { [key: string]: string } = {
    harmony: '#6366f1',
    rhythm: '#f59e0b',
    arrangement: '#10b981',
    production: '#ec4899',
  };

  const categoryIcons: { [key: string]: string } = {
    harmony: '🎹',
    rhythm: '🥁',
    arrangement: '🎼',
    production: '🎚️',
  };

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        Loading rule-breaking options...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', color: '#f44336' }}>
        {error}
        <button onClick={loadRules} style={{ marginLeft: '10px' }}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: '15px' }}>
      <div style={{ marginBottom: '15px' }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#333' }}>
          Rule-Breaking Techniques
        </h4>
        <p style={{ margin: 0, fontSize: '0.85em', color: '#666' }}>
          Sometimes breaking the rules creates the most emotional impact.
        </p>
      </div>

      {suggestions.length > 0 && (
        <div
          style={{
            marginBottom: '15px',
            padding: '10px',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            borderRadius: '8px',
            border: '1px solid rgba(99, 102, 241, 0.3)',
          }}
        >
          <div style={{ fontSize: '0.85em', fontWeight: 'bold', marginBottom: '8px' }}>
            Suggested for "{selectedEmotion}":
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {suggestions.map((s) => (
              <span
                key={s}
                style={{
                  padding: '4px 8px',
                  backgroundColor: '#6366f1',
                  color: 'white',
                  borderRadius: '4px',
                  fontSize: '0.8em',
                }}
              >
                {s.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {rules && Object.entries(rules).map(([category, techniques]) => (
        <div
          key={category}
          style={{
            marginBottom: '10px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            overflow: 'hidden',
          }}
        >
          <div
            onClick={() => setExpandedCategory(expandedCategory === category ? null : category)}
            style={{
              padding: '12px 15px',
              backgroundColor: categoryColors[category] || '#888',
              color: 'white',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontWeight: 'bold',
            }}
          >
            <span>
              {categoryIcons[category]} {category.charAt(0).toUpperCase() + category.slice(1)}
            </span>
            <span>{expandedCategory === category ? '▼' : '▶'}</span>
          </div>

          {expandedCategory === category && (
            <div style={{ padding: '10px' }}>
              {Object.entries(techniques as RuleCategory).map(([technique, effect]) => {
                const ruleKey = `${category}:${technique}`;
                const isSelected = selectedRules.has(ruleKey);
                const isSuggested = suggestions.includes(technique);

                return (
                  <div
                    key={technique}
                    onClick={() => toggleRule(category, technique, effect)}
                    style={{
                      padding: '10px',
                      marginBottom: '6px',
                      backgroundColor: isSelected
                        ? 'rgba(99, 102, 241, 0.2)'
                        : isSuggested
                        ? 'rgba(245, 158, 11, 0.1)'
                        : '#f9f9f9',
                      border: `2px solid ${
                        isSelected
                          ? '#6366f1'
                          : isSuggested
                          ? '#f59e0b'
                          : 'transparent'
                      }`,
                      borderRadius: '6px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <span style={{ fontWeight: 500 }}>
                        {technique.replace(/_/g, ' ')}
                      </span>
                      {isSelected && <span>✓</span>}
                      {isSuggested && !isSelected && (
                        <span style={{ fontSize: '0.75em', color: '#f59e0b' }}>
                          suggested
                        </span>
                      )}
                    </div>
                    <div
                      style={{
                        fontSize: '0.85em',
                        color: '#666',
                        marginTop: '4px',
                      }}
                    >
                      {effect}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ))}

      {selectedRules.size > 0 && (
        <div
          style={{
            marginTop: '15px',
            padding: '10px',
            backgroundColor: '#e8f5e9',
            borderRadius: '8px',
            border: '1px solid #4caf50',
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
            Selected Rules ({selectedRules.size}):
          </div>
          <div style={{ fontSize: '0.85em' }}>
            {Array.from(selectedRules).map((r) => (
              <span
                key={r}
                style={{
                  display: 'inline-block',
                  padding: '2px 6px',
                  margin: '2px',
                  backgroundColor: '#4caf50',
                  color: 'white',
                  borderRadius: '3px',
                }}
              >
                {r.split(':')[1].replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
