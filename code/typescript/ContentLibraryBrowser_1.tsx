import React, { useState, useMemo } from 'react';
import { ContentLibrary, LibraryItem, SmartCollection } from './ContentLibrary';

interface ContentLibraryBrowserProps {
  library: ContentLibrary;
  onItemSelect?: (item: LibraryItem) => void;
  onItemLoad?: (item: LibraryItem) => void;
}

export const ContentLibraryBrowser: React.FC<ContentLibraryBrowserProps> = ({
  library,
  onItemSelect,
  onItemLoad,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [minRating, setMinRating] = useState(0);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState<'name' | 'rating' | 'bpm' | 'date'>('name');
  const [showSmartCollections, setShowSmartCollections] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState('');

  const allTags = library.getAllTags();
  const collections = library.getCollections();

  // Search and filter
  const filteredItems = useMemo(() => {
    let results = library.search({
      type: selectedType !== 'all' ? [selectedType as any] : undefined,
      tags: selectedTags.length > 0 ? selectedTags : undefined,
      minRating,
      name: searchQuery || undefined,
    });

    // Sort
    results.sort((a, b) => {
      switch (sortBy) {
        case 'rating':
          return b.rating - a.rating;
        case 'bpm':
          return (b.metadata.bpm || 0) - (a.metadata.bpm || 0);
        case 'date':
          return 0; // Would use actual date
        default:
          return a.name.localeCompare(b.name);
      }
    });

    return results;
  }, [library, searchQuery, selectedType, selectedTags, minRating, sortBy]);

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const createSmartCollection = () => {
    if (!newCollectionName.trim()) return;

    const collection: SmartCollection = {
      id: `collection-${Date.now()}`,
      name: newCollectionName,
      query: {
        type: selectedType !== 'all' ? [selectedType as any] : undefined,
        tags: selectedTags.length > 0 ? selectedTags : undefined,
        minRating: minRating > 0 ? minRating : undefined,
      },
    };

    library.createSmartCollection(collection);
    setNewCollectionName('');
    setShowSmartCollections(false);
  };

  const loadCollection = (_collectionId: string) => {
    // Could set these as the current filter
    // const items = library.getSmartCollection(collectionId);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
          <h3 style={{ margin: 0, color: '#fff' }}>Content Library</h3>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setViewMode('grid')}
              style={{
                padding: '6px 12px',
                backgroundColor: viewMode === 'grid' ? '#6366f1' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.85em',
              }}
            >
              ⚏ Grid
            </button>
            <button
              onClick={() => setViewMode('list')}
              style={{
                padding: '6px 12px',
                backgroundColor: viewMode === 'list' ? '#6366f1' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.85em',
              }}
            >
              ☰ List
            </button>
          </div>
        </div>

        {/* Search */}
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search library..."
          style={{
            width: '100%',
            padding: '10px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '0.9em',
            marginBottom: '10px',
          }}
        />

        {/* Filters */}
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {/* Type Filter */}
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            style={{
              padding: '6px 12px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.85em',
            }}
          >
            <option value="all">All Types</option>
            <option value="loop">Loops</option>
            <option value="one-shot">One-Shots</option>
            <option value="instrument">Instruments</option>
            <option value="preset">Presets</option>
            <option value="effect-preset">Effect Presets</option>
            <option value="sound-pack">Sound Packs</option>
          </select>

          {/* Rating Filter */}
          <select
            value={minRating}
            onChange={(e) => setMinRating(Number(e.target.value))}
            style={{
              padding: '6px 12px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.85em',
            }}
          >
            <option value="0">Any Rating</option>
            <option value="1">1+ Stars</option>
            <option value="2">2+ Stars</option>
            <option value="3">3+ Stars</option>
            <option value="4">4+ Stars</option>
            <option value="5">5 Stars</option>
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            style={{
              padding: '6px 12px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.85em',
            }}
          >
            <option value="name">Sort by Name</option>
            <option value="rating">Sort by Rating</option>
            <option value="bpm">Sort by BPM</option>
            <option value="date">Sort by Date</option>
          </select>

          <button
            onClick={() => setShowSmartCollections(!showSmartCollections)}
            style={{
              padding: '6px 12px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.85em',
            }}
          >
            📁 Collections
          </button>
        </div>
      </div>

      {/* Smart Collections Panel */}
      {showSmartCollections && (
        <div
          style={{
            padding: '15px',
            backgroundColor: '#2a2a2a',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <div style={{ marginBottom: '10px', color: '#fff', fontWeight: 'bold' }}>Smart Collections</div>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
            <input
              type="text"
              value={newCollectionName}
              onChange={(e) => setNewCollectionName(e.target.value)}
              placeholder="Collection name..."
              style={{
                flex: 1,
                padding: '6px 12px',
                backgroundColor: '#1a1a1a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                fontSize: '0.85em',
              }}
            />
            <button
              onClick={createSmartCollection}
              style={{
                padding: '6px 12px',
                backgroundColor: '#4caf50',
                border: 'none',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.85em',
              }}
            >
              Create
            </button>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {collections.map((collection) => (
              <button
                key={collection.id}
                onClick={() => loadCollection(collection.id)}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#333',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '0.85em',
                }}
              >
                {collection.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Tags */}
      {allTags.length > 0 && (
        <div
          style={{
            padding: '10px 15px',
            backgroundColor: '#2a2a2a',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <div style={{ marginBottom: '8px', color: '#aaa', fontSize: '0.85em' }}>Tags:</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                style={{
                  padding: '4px 10px',
                  backgroundColor: selectedTags.includes(tag) ? '#6366f1' : '#333',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '0.8em',
                }}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Content Grid/List */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '15px',
        }}
      >
        {filteredItems.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
            No items found. Try adjusting your filters.
          </div>
        ) : viewMode === 'grid' ? (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: '15px',
            }}
          >
            {filteredItems.map((item) => (
              <LibraryItemCard
                key={item.id}
                item={item}
                library={library}
                onSelect={onItemSelect}
                onLoad={onItemLoad}
              />
            ))}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {filteredItems.map((item) => (
              <LibraryItemRow
                key={item.id}
                item={item}
                library={library}
                onSelect={onItemSelect}
                onLoad={onItemLoad}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div
        style={{
          padding: '10px 15px',
          backgroundColor: '#0f0f0f',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '0.85em',
          color: '#aaa',
        }}
      >
        <span>{filteredItems.length} items</span>
        <span>{selectedTags.length > 0 && `${selectedTags.length} tags selected`}</span>
      </div>
    </div>
  );
};

interface LibraryItemCardProps {
  item: LibraryItem;
  library: ContentLibrary;
  onSelect?: (item: LibraryItem) => void;
  onLoad?: (item: LibraryItem) => void;
}

const LibraryItemCard: React.FC<LibraryItemCardProps> = ({ item, library, onSelect, onLoad }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={() => onSelect?.(item)}
      style={{
        padding: '12px',
        backgroundColor: isHovered ? '#2a2a2a' : '#1a1a1a',
        border: `2px solid ${item.color || 'rgba(255, 255, 255, 0.1)'}`,
        borderRadius: '8px',
        cursor: 'pointer',
        transition: 'all 0.2s',
        position: 'relative',
      }}
    >
      {/* Color indicator */}
      <div
        style={{
          position: 'absolute',
          top: '8px',
          right: '8px',
          width: '12px',
          height: '12px',
          backgroundColor: item.color || '#6366f1',
          borderRadius: '50%',
          border: '2px solid #1a1a1a',
        }}
      />

      {/* Type badge */}
      <div
        style={{
          position: 'absolute',
          top: '8px',
          left: '8px',
          padding: '2px 6px',
          backgroundColor: '#6366f1',
          borderRadius: '4px',
          fontSize: '0.7em',
          color: '#fff',
          textTransform: 'uppercase',
        }}
      >
        {item.type}
      </div>

      {/* Preview area */}
      <div
        style={{
          width: '100%',
          height: '120px',
          backgroundColor: '#0a0a0a',
          borderRadius: '4px',
          marginBottom: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#666',
          fontSize: '0.8em',
        }}
      >
        {item.type === 'loop' && '🔁'}
        {item.type === 'one-shot' && '💥'}
        {item.type === 'instrument' && '🎹'}
        {item.type === 'preset' && '⚙️'}
        {item.type === 'effect-preset' && '🎛️'}
        {item.type === 'sound-pack' && '📦'}
      </div>

      {/* Name */}
      <div
        style={{
          color: '#fff',
          fontWeight: 'bold',
          fontSize: '0.9em',
          marginBottom: '6px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {item.name}
      </div>

      {/* Metadata */}
      <div style={{ fontSize: '0.75em', color: '#888', marginBottom: '8px' }}>
        {item.metadata.bpm && <div>BPM: {item.metadata.bpm}</div>}
        {item.metadata.key && <div>Key: {item.metadata.key}</div>}
        {item.metadata.genre && <div>Genre: {item.metadata.genre}</div>}
      </div>

      {/* Rating */}
      <div style={{ display: 'flex', gap: '2px', marginBottom: '8px' }}>
        {Array.from({ length: 5 }).map((_, i) => (
          <span key={i} style={{ color: i < item.rating ? '#ffeb3b' : '#444', fontSize: '0.9em' }}>
            ★
          </span>
        ))}
      </div>

      {/* Tags */}
      {item.tags.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '8px' }}>
          {item.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              style={{
                padding: '2px 6px',
                backgroundColor: '#333',
                borderRadius: '3px',
                fontSize: '0.7em',
                color: '#aaa',
              }}
            >
              {tag}
            </span>
          ))}
          {item.tags.length > 3 && (
            <span style={{ fontSize: '0.7em', color: '#666' }}>+{item.tags.length - 3}</span>
          )}
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: '6px' }}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onLoad?.(item);
          }}
          style={{
            flex: 1,
            padding: '6px',
            backgroundColor: '#4caf50',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
            fontWeight: 'bold',
          }}
        >
          Load
        </button>
        {!item.downloaded && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              library.addToDownloadQueue(item.id);
            }}
            style={{
              padding: '6px 12px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.8em',
            }}
            title="Add to download queue"
          >
            ⬇
          </button>
        )}
      </div>
    </div>
  );
};

const LibraryItemRow: React.FC<LibraryItemCardProps> = ({ item, library, onSelect, onLoad }) => {
  return (
    <div
      onClick={() => onSelect?.(item)}
      style={{
        padding: '12px',
        backgroundColor: '#1a1a1a',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '4px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '15px',
        transition: 'background-color 0.2s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = '#2a2a2a';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = '#1a1a1a';
      }}
    >
      {/* Color indicator */}
      <div
        style={{
          width: '4px',
          height: '40px',
          backgroundColor: item.color || '#6366f1',
          borderRadius: '2px',
        }}
      />

      {/* Type icon */}
      <div style={{ fontSize: '1.5em', width: '30px', textAlign: 'center' }}>
        {item.type === 'loop' && '🔁'}
        {item.type === 'one-shot' && '💥'}
        {item.type === 'instrument' && '🎹'}
        {item.type === 'preset' && '⚙️'}
        {item.type === 'effect-preset' && '🎛️'}
        {item.type === 'sound-pack' && '📦'}
      </div>

      {/* Name and metadata */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '4px' }}>{item.name}</div>
        <div style={{ fontSize: '0.85em', color: '#888' }}>
          {item.metadata.bpm && `BPM: ${item.metadata.bpm} • `}
          {item.metadata.key && `Key: ${item.metadata.key} • `}
          {item.metadata.genre && `Genre: ${item.metadata.genre}`}
        </div>
        {item.tags.length > 0 && (
          <div style={{ display: 'flex', gap: '4px', marginTop: '4px', flexWrap: 'wrap' }}>
            {item.tags.map((tag) => (
              <span
                key={tag}
                style={{
                  padding: '2px 6px',
                  backgroundColor: '#333',
                  borderRadius: '3px',
                  fontSize: '0.75em',
                  color: '#aaa',
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Rating */}
      <div style={{ display: 'flex', gap: '2px', marginRight: '15px' }}>
        {Array.from({ length: 5 }).map((_, i) => (
          <span key={i} style={{ color: i < item.rating ? '#ffeb3b' : '#444' }}>
            ★
          </span>
        ))}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onLoad?.(item);
          }}
          style={{
            padding: '6px 12px',
            backgroundColor: '#4caf50',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.85em',
            fontWeight: 'bold',
          }}
        >
          Load
        </button>
        {!item.downloaded && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              library.addToDownloadQueue(item.id);
            }}
            style={{
              padding: '6px 12px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.85em',
            }}
            title="Download"
          >
            ⬇
          </button>
        )}
      </div>
    </div>
  );
};
