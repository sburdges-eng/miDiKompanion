import React, { useState } from 'react';
import { CollaborationEngine, ProjectVersion, CollaborationSession, Comment } from './CollaborationEngine';

interface CollaborationPanelProps {
  engine: CollaborationEngine;
  projectId: string;
  currentUser: string;
  onVersionRestore?: (version: ProjectVersion) => void;
}

export const CollaborationPanel: React.FC<CollaborationPanelProps> = ({
  engine,
  projectId,
  currentUser,
  onVersionRestore,
}) => {
  const [activeTab, setActiveTab] = useState<'versions' | 'collaboration' | 'comments' | 'sharing'>('versions');
  const [versions] = useState(() => engine.getVersions(projectId));
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [newComment, setNewComment] = useState('');
  const [cloudStorageEnabled, setCloudStorageEnabled] = useState(false);
  const [cloudSyncEnabled, setCloudSyncEnabled] = useState(false);

  const session = sessionId ? engine.getSession(sessionId) : undefined;

  const createVersion = () => {
    engine.createVersion(projectId, currentUser, 'Manual save');
  };

  const restoreVersion = async (versionId: string) => {
    const version = versions.find((v) => v.id === versionId);
    if (version && onVersionRestore) {
      await engine.restoreVersion(projectId, versionId);
      onVersionRestore(version);
    }
  };

  const startCollaboration = () => {
    const newSessionId = engine.createSession(projectId, currentUser);
    setSessionId(newSessionId);
    setActiveTab('collaboration');
  };

  const addComment = () => {
    if (!sessionId || !newComment.trim()) return;

    engine.addComment(sessionId, {
      author: currentUser,
      text: newComment,
    });
    setNewComment('');
  };

  const formatDate = (date: Date): string => {
    return new Date(date).toLocaleString();
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
        <h3 style={{ margin: '0 0 15px 0', color: '#fff' }}>Collaboration</h3>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '8px', marginBottom: '15px' }}>
          {(['versions', 'collaboration', 'comments', 'sharing'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '8px 16px',
                backgroundColor: activeTab === tab ? '#6366f1' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.85em',
                fontWeight: activeTab === tab ? 'bold' : 'normal',
                textTransform: 'capitalize',
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Cloud Controls */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#aaa', fontSize: '0.85em' }}>
            <input
              type="checkbox"
              checked={cloudStorageEnabled}
              onChange={(e) => {
                setCloudStorageEnabled(e.target.checked);
                engine.enableCloudStorage(e.target.checked);
              }}
            />
            Cloud Storage
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#aaa', fontSize: '0.85em' }}>
            <input
              type="checkbox"
              checked={cloudSyncEnabled}
              onChange={(e) => {
                setCloudSyncEnabled(e.target.checked);
                engine.enableCloudSync(e.target.checked);
              }}
            />
            Cloud Sync
          </label>
          <button
            onClick={() => engine.syncProject(projectId)}
            disabled={!cloudSyncEnabled}
            style={{
              padding: '6px 12px',
              backgroundColor: cloudSyncEnabled ? '#4caf50' : '#666',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: cloudSyncEnabled ? 'pointer' : 'not-allowed',
              fontSize: '0.85em',
            }}
          >
            Sync Now
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '15px' }}>
        {activeTab === 'versions' && (
          <VersionsTab
            versions={versions}
            onCreateVersion={createVersion}
            onRestoreVersion={restoreVersion}
            formatDate={formatDate}
          />
        )}

        {activeTab === 'collaboration' && (
          <CollaborationTab
            engine={engine}
            sessionId={sessionId}
            session={session}
            currentUser={currentUser}
            onStartCollaboration={startCollaboration}
            onInvite={(userId, permission) => {
              if (sessionId) {
                engine.inviteCollaborator(sessionId, userId, permission);
              }
            }}
          />
        )}

        {activeTab === 'comments' && (
          <CommentsTab
            engine={engine}
            sessionId={sessionId}
            session={session}
            currentUser={currentUser}
            newComment={newComment}
            onCommentChange={setNewComment}
            onAddComment={addComment}
            onResolveComment={(commentId) => {
              if (sessionId) {
                engine.resolveComment(sessionId, commentId);
              }
            }}
            formatDate={formatDate}
          />
        )}

        {activeTab === 'sharing' && (
          <SharingTab
            engine={engine}
            projectId={projectId}
            onShareStem={async (stem) => {
              return await engine.shareStem(stem);
            }}
            onExportStems={async () => {
              return await engine.exportStems(projectId, 'wav');
            }}
            onExportMultitracks={async () => {
              return await engine.exportMultitracks(projectId);
            }}
            onExportPreview={async () => {
              return await engine.exportPreviewMix(projectId);
            }}
          />
        )}
      </div>
    </div>
  );
};

interface VersionsTabProps {
  versions: ProjectVersion[];
  onCreateVersion: () => void;
  onRestoreVersion: (versionId: string) => void;
  formatDate: (date: Date) => string;
}

const VersionsTab: React.FC<VersionsTabProps> = ({ versions, onCreateVersion, onRestoreVersion, formatDate }) => {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h4 style={{ margin: 0, color: '#fff' }}>Version History</h4>
        <button
          onClick={onCreateVersion}
          style={{
            padding: '8px 16px',
            backgroundColor: '#6366f1',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.85em',
            fontWeight: 'bold',
          }}
        >
          + Create Version
        </button>
      </div>

      {versions.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
          No versions saved yet. Create a version to track changes.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {versions.map((version) => (
            <div
              key={version.id}
              style={{
                padding: '15px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                <div>
                  <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '4px' }}>
                    Version {version.version}
                  </div>
                  <div style={{ fontSize: '0.85em', color: '#aaa' }}>
                    {formatDate(version.timestamp)} • {version.author}
                  </div>
                  {version.description && (
                    <div style={{ fontSize: '0.85em', color: '#888', marginTop: '4px' }}>
                      {version.description}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => onRestoreVersion(version.id)}
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
                  Restore
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

interface CollaborationTabProps {
  engine: CollaborationEngine;
  sessionId: string | null;
  session: CollaborationSession | undefined;
  currentUser: string;
  onStartCollaboration: () => void;
  onInvite: (userId: string, permission: 'read' | 'write' | 'admin') => void;
}

const CollaborationTab: React.FC<CollaborationTabProps> = ({
  sessionId,
  session,
  currentUser: _currentUser,
  onStartCollaboration,
  onInvite,
}) => {
  const [inviteUserId, setInviteUserId] = useState('');
  const [invitePermission, setInvitePermission] = useState<'read' | 'write' | 'admin'>('write');

  if (!sessionId || !session) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <div style={{ color: '#888', marginBottom: '20px' }}>No active collaboration session</div>
        <button
          onClick={onStartCollaboration}
          style={{
            padding: '12px 24px',
            backgroundColor: '#6366f1',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '1em',
            fontWeight: 'bold',
          }}
        >
          Start Collaboration Session
        </button>
      </div>
    );
  }

  return (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#fff' }}>Active Session</h4>

      {/* Participants */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ color: '#aaa', fontSize: '0.9em', marginBottom: '10px' }}>Participants:</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {session.participants.map((userId) => (
            <div
              key={userId}
              style={{
                padding: '10px',
                backgroundColor: '#2a2a2a',
                borderRadius: '4px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div>
                <span style={{ color: '#fff', fontWeight: 'bold' }}>{userId}</span>
                {userId === session.owner && (
                  <span style={{ marginLeft: '8px', color: '#ffeb3b', fontSize: '0.85em' }}>(Owner)</span>
                )}
                <span style={{ marginLeft: '8px', color: '#888', fontSize: '0.85em' }}>
                  {session.permissions.get(userId) || 'read'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Invite */}
      <div style={{ padding: '15px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
        <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '10px' }}>Invite Collaborator</div>
        <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
          <input
            type="text"
            value={inviteUserId}
            onChange={(e) => setInviteUserId(e.target.value)}
            placeholder="User ID or email"
            style={{
              flex: 1,
              padding: '8px',
              backgroundColor: '#1a1a1a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.9em',
            }}
          />
          <select
            value={invitePermission}
            onChange={(e) => setInvitePermission(e.target.value as any)}
            style={{
              padding: '8px',
              backgroundColor: '#1a1a1a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.9em',
            }}
          >
            <option value="read">Read</option>
            <option value="write">Write</option>
            <option value="admin">Admin</option>
          </select>
          <button
            onClick={() => {
              if (inviteUserId.trim()) {
                onInvite(inviteUserId, invitePermission);
                setInviteUserId('');
              }
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: '#4caf50',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.9em',
              fontWeight: 'bold',
            }}
          >
            Invite
          </button>
        </div>
      </div>

      {/* Session Status */}
      <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
        <div style={{ color: '#aaa', fontSize: '0.85em' }}>
          Session ID: <span style={{ color: '#fff', fontFamily: 'monospace' }}>{sessionId}</span>
        </div>
        <div style={{ color: '#aaa', fontSize: '0.85em', marginTop: '4px' }}>
          Status: <span style={{ color: session.isActive ? '#4caf50' : '#f44336' }}>
            {session.isActive ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>
    </div>
  );
};

interface CommentsTabProps {
  engine: CollaborationEngine;
  sessionId: string | null;
  session: CollaborationSession | undefined;
  currentUser: string;
  newComment: string;
  onCommentChange: (text: string) => void;
  onAddComment: () => void;
  onResolveComment: (commentId: string) => void;
  formatDate: (date: Date) => string;
}

const CommentsTab: React.FC<CommentsTabProps> = ({
  sessionId,
  session,
  currentUser,
  newComment,
  onCommentChange,
  onAddComment,
  onResolveComment,
  formatDate,
}) => {
  if (!sessionId || !session) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
        Start a collaboration session to add comments
      </div>
    );
  }

  const comments = session.comments || [];
  const unresolvedComments = comments.filter((c) => !c.resolved);
  const resolvedComments = comments.filter((c) => c.resolved);

  return (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#fff' }}>Comments & Annotations</h4>

      {/* Add Comment */}
      <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
        <textarea
          value={newComment}
          onChange={(e) => onCommentChange(e.target.value)}
          placeholder="Add a comment or annotation..."
          style={{
            width: '100%',
            minHeight: '80px',
            padding: '10px',
            backgroundColor: '#1a1a1a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '0.9em',
            resize: 'vertical',
            marginBottom: '10px',
          }}
        />
        <button
          onClick={onAddComment}
          disabled={!newComment.trim()}
          style={{
            padding: '8px 16px',
            backgroundColor: newComment.trim() ? '#6366f1' : '#666',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: newComment.trim() ? 'pointer' : 'not-allowed',
            fontSize: '0.9em',
            fontWeight: 'bold',
          }}
        >
          Add Comment
        </button>
      </div>

      {/* Unresolved Comments */}
      {unresolvedComments.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '10px' }}>
            Active Comments ({unresolvedComments.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {unresolvedComments.map((comment) => (
              <CommentItem
                key={comment.id}
                comment={comment}
                currentUser={currentUser}
                onResolve={() => onResolveComment(comment.id)}
                formatDate={formatDate}
              />
            ))}
          </div>
        </div>
      )}

      {/* Resolved Comments */}
      {resolvedComments.length > 0 && (
        <div>
          <div style={{ color: '#888', fontSize: '0.9em', marginBottom: '10px' }}>
            Resolved Comments ({resolvedComments.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {resolvedComments.map((comment) => (
              <CommentItem
                key={comment.id}
                comment={comment}
                currentUser={currentUser}
                onResolve={() => onResolveComment(comment.id)}
                formatDate={formatDate}
              />
            ))}
          </div>
        </div>
      )}

      {comments.length === 0 && (
        <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
          No comments yet. Add a comment to start the conversation.
        </div>
      )}
    </div>
  );
};

interface CommentItemProps {
  comment: Comment;
  currentUser: string;
  onResolve: () => void;
  formatDate: (date: Date) => string;
}

const CommentItem: React.FC<CommentItemProps> = ({ comment, currentUser, onResolve, formatDate }) => {
  return (
    <div
      style={{
        padding: '12px',
        backgroundColor: comment.resolved ? '#1a1a1a' : '#2a2a2a',
        borderRadius: '4px',
        border: `1px solid ${comment.resolved ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.1)'}`,
        opacity: comment.resolved ? 0.6 : 1,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
        <div>
          <span style={{ color: '#fff', fontWeight: 'bold' }}>{comment.author}</span>
          {comment.author === currentUser && (
            <span style={{ marginLeft: '8px', color: '#6366f1', fontSize: '0.85em' }}>(You)</span>
          )}
          <span style={{ marginLeft: '8px', color: '#888', fontSize: '0.85em' }}>
            {formatDate(comment.timestamp)}
          </span>
          {comment.position && (
            <span style={{ marginLeft: '8px', color: '#888', fontSize: '0.85em' }}>
              @ {comment.position.time.toFixed(2)}s
            </span>
          )}
        </div>
        {!comment.resolved && (
          <button
            onClick={onResolve}
            style={{
              padding: '4px 8px',
              backgroundColor: '#4caf50',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.75em',
            }}
          >
            Resolve
          </button>
        )}
        {comment.resolved && (
          <span style={{ color: '#4caf50', fontSize: '0.85em' }}>✓ Resolved</span>
        )}
      </div>
      <div style={{ color: '#ccc', fontSize: '0.9em', lineHeight: '1.5' }}>{comment.text}</div>
    </div>
  );
};

interface SharingTabProps {
  engine: CollaborationEngine;
  projectId: string;
  onShareStem: (stem: any) => Promise<string>;
  onExportStems: () => Promise<string[]>;
  onExportMultitracks: () => Promise<string>;
  onExportPreview: () => Promise<string>;
}

const SharingTab: React.FC<SharingTabProps> = ({
  engine,
  projectId,
  onShareStem: _onShareStem,
  onExportStems,
  onExportMultitracks,
  onExportPreview,
}) => {
  const [exporting, setExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState<string>('');

  const handleExport = async (exportFn: () => Promise<any>, name: string) => {
    setExporting(true);
    setExportStatus(`Exporting ${name}...`);
    try {
      await exportFn();
      setExportStatus(`${name} exported successfully!`);
      setTimeout(() => setExportStatus(''), 3000);
    } catch (error) {
      setExportStatus(`Error exporting ${name}`);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div>
      <h4 style={{ margin: '0 0 15px 0', color: '#fff' }}>Export & Sharing</h4>

      {exportStatus && (
        <div
          style={{
            padding: '10px',
            backgroundColor: exportStatus.includes('Error') ? '#f4433620' : '#4caf5020',
            border: `1px solid ${exportStatus.includes('Error') ? '#f44336' : '#4caf50'}`,
            borderRadius: '4px',
            marginBottom: '15px',
            color: exportStatus.includes('Error') ? '#f44336' : '#4caf50',
            fontSize: '0.9em',
          }}
        >
          {exportStatus}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {/* Stem Export */}
        <div style={{ padding: '15px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
          <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '8px' }}>Stem Export</div>
          <div style={{ fontSize: '0.85em', color: '#aaa', marginBottom: '10px' }}>
            Export individual tracks as separate audio files
          </div>
          <button
            onClick={() => handleExport(onExportStems, 'stems')}
            disabled={exporting}
            style={{
              padding: '8px 16px',
              backgroundColor: exporting ? '#666' : '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: exporting ? 'not-allowed' : 'pointer',
              fontSize: '0.9em',
              fontWeight: 'bold',
            }}
          >
            {exporting ? 'Exporting...' : 'Export Stems'}
          </button>
        </div>

        {/* Multitrack Export */}
        <div style={{ padding: '15px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
          <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '8px' }}>Multitrack Export</div>
          <div style={{ fontSize: '0.85em', color: '#aaa', marginBottom: '10px' }}>
            Export all tracks in a single archive
          </div>
          <button
            onClick={() => handleExport(onExportMultitracks, 'multitracks')}
            disabled={exporting}
            style={{
              padding: '8px 16px',
              backgroundColor: exporting ? '#666' : '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: exporting ? 'not-allowed' : 'pointer',
              fontSize: '0.9em',
              fontWeight: 'bold',
            }}
          >
            {exporting ? 'Exporting...' : 'Export Multitracks'}
          </button>
        </div>

        {/* Preview Mix */}
        <div style={{ padding: '15px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
          <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '8px' }}>Preview Mix</div>
          <div style={{ fontSize: '0.85em', color: '#aaa', marginBottom: '10px' }}>
            Export a rough mix for review
          </div>
          <button
            onClick={() => handleExport(onExportPreview, 'preview mix')}
            disabled={exporting}
            style={{
              padding: '8px 16px',
              backgroundColor: exporting ? '#666' : '#4caf50',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: exporting ? 'not-allowed' : 'pointer',
              fontSize: '0.9em',
              fontWeight: 'bold',
            }}
          >
            {exporting ? 'Exporting...' : 'Export Preview'}
          </button>
        </div>

        {/* Additional Export Options */}
        <div style={{ padding: '15px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
          <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '10px' }}>Additional Exports</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '8px' }}>
            <button
              onClick={() => handleExport(() => engine.exportSessionArchive(projectId), 'session archive')}
              disabled={exporting}
              style={{
                padding: '8px 12px',
                backgroundColor: exporting ? '#666' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '0.85em',
              }}
            >
              Session Archive
            </button>
            <button
              onClick={() => handleExport(() => engine.exportNotes(projectId), 'notes')}
              disabled={exporting}
              style={{
                padding: '8px 12px',
                backgroundColor: exporting ? '#666' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '0.85em',
              }}
            >
              Notes
            </button>
            <button
              onClick={() => handleExport(() => engine.exportTrackSheet(projectId), 'track sheet')}
              disabled={exporting}
              style={{
                padding: '8px 12px',
                backgroundColor: exporting ? '#666' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '0.85em',
              }}
            >
              Track Sheet
            </button>
            <button
              onClick={() => handleExport(() => engine.exportSessionInfo(projectId), 'session info')}
              disabled={exporting}
              style={{
                padding: '8px 12px',
                backgroundColor: exporting ? '#666' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: exporting ? 'not-allowed' : 'pointer',
                fontSize: '0.85em',
              }}
            >
              Session Info
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
